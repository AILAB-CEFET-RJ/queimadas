"""Baseline: STConvS2S com a receita do artigo do dataset nos 12 folds por ano, com teste único por fold.

Protocolo (o dos modelos multitemporais do artigo, `cfgs/*/wandb_table5.yaml` e `train.py` do repositório oficial):
cada fold oficial treina com dois anos, valida com um e testa com o quarto; o número reportado é a média e o desvio
do AP de teste entre os 12 folds. Cada run é `stconv_train.py --recipe paper` (lote 32, 10.000 passos, AdamW
lr 0,01, checkpoint pelo menor `val_loss` em recortes aumentados da validação, `center_crop_x32` no teste,
`pos_weight` = 1/taxa de fogo nas imagens de treino do fold, todas as features, semente 0).

O fold 2 vai primeiro: é o fold ilustrado na documentação do dataset e dá um primeiro número cedo. O executor foi
feito para rodar dias sem supervisão:

- é retomável em dois níveis: um fold com relatório completo é pulado, e um fold interrompido continua do `last.pt`
  da última época;
- uma falha transitória (driver, memória, antivírus segurando um arquivo) ganha novas tentativas com `--resume`;
  um fold que falha de vez fica registrado no resumo e os seguintes continuam;
- pede ao sistema operacional para não suspender enquanto roda (`prevent_sleep`) e grava a própria saída em
  `reports/logs/baseline_runner.log`, porque roda num processo sem console.

A normalização de cada fold é ajustada só nos eventos de treino dele, na primeira vez que ele roda.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Só módulos sem torch: o executor fica vivo por dias e não deve ocupar a RAM que o treino usa.
from scripts.core import (  # noqa: E402
    DEFAULT_DATA_ROOT, check_normalization, fit_normalization_for_split, resolve_split, split_years,
)
from scripts.io_utils import power_status, prevent_sleep, write_json  # noqa: E402
from scripts.run_registry import (  # noqa: E402
    REPORTS_DIR, RUNS_REPORT_DIR, TEST_LEDGER, ledger_entries, pending_test_report,
)

FOLD_ORDER = (2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11)
RUN_PREFIX = "baseline_paper_f"
DEFAULT_REPORT = REPORTS_DIR / "baseline_paper_summary.json"
TRAIN_SCRIPT = "scripts/stconv_train.py"
TRAIN_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 60
# O que um relatório precisa ter para contar como fold do baseline (um run de outra receita com o mesmo nome não entra).
EXPECTED_PROTOCOL = {"recipe": "paper", "eval_region": "crop32", "val_sampling": "train_crops", "split_kind": "temporal"}


def run_name(fold: int) -> str:
    return f"{RUN_PREFIX}{fold:02d}"


class Log:
    """Saída do executor no console e em arquivo (o processo desacoplado não tem console visível)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def ensure_normalization(data_root: Path, fold: int, log: Log) -> Path:
    """Arquivo de normalização do fold, ajustado só nos eventos de treino dele se ainda não existir."""
    path = data_root / "stats" / f"fold{fold}_train_normalization.json"
    spec = resolve_split(data_root / "hdf5", "temporal", fold)
    if path.exists():
        check_normalization(json.loads(path.read_text(encoding="utf-8")), spec)
        return path
    log(f"[normalização] fold {fold}: ajustando em {len(spec.events['train'])} eventos de treino")
    write_json(fit_normalization_for_split(data_root / "hdf5", spec), path)
    return path


def train_command(fold: int, args: argparse.Namespace, *, resume: bool) -> list[str]:
    """Linha de comando do treino de um fold com a receita do artigo."""
    command = [sys.executable, "-u", TRAIN_SCRIPT, "--recipe", "paper", "--fold", str(fold),
               "--run-name", run_name(fold), "--accumulate", str(args.accumulate),
               "--num-workers", str(args.num_workers), "--data-root", str(args.data_root)]
    return command + (["--resume"] if resume else [])


def test_command(fold: int) -> list[str]:
    return [sys.executable, "-u", TRAIN_SCRIPT, "--evaluate-test", "--run-name", run_name(fold)]


def execute(command: list[str], log_path: Path) -> int:
    """Roda um passo com a saída anexada ao log do fold e devolve o código de saída."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] $ {' '.join(command)}\n")
        stream.flush()
        return subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def read_report(fold: int, report_dir: Path = RUNS_REPORT_DIR) -> dict[str, Any] | None:
    path = report_dir / f"{run_name(fold)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def protocol_problems(report: dict[str, Any], fold: int) -> list[str]:
    """Divergências entre o relatório e o protocolo do baseline; vazio quando o fold pode entrar na média."""
    protocol = report.get("protocol", {})
    problems = [f"{key}={protocol.get(key)!r} (esperado {value!r})"
                for key, value in EXPECTED_PROTOCOL.items() if protocol.get(key) != value]
    if protocol.get("fold") != fold:
        problems.append(f"fold={protocol.get('fold')!r} (esperado {fold})")
    if report.get("kind") != "full":
        problems.append(f"kind={report.get('kind')!r} (esperado 'full')")
    return problems


def fold_entry(fold: int, report_dir: Path = RUNS_REPORT_DIR) -> dict[str, Any] | None:
    """Resumo de um fold a partir do relatório do run, ou `None` se ele ainda não terminou o treino."""
    report = read_report(fold, report_dir)
    if report is None:
        return None
    test = report.get("test")
    metrics = test["metrics"]["all"] if test else None
    persistence = metrics["baselines"]["persistence_all_detections_t"] if metrics else None
    repo = report.get("provenance", {}).get("repo", {})
    return {
        "fold": fold,
        "run_name": run_name(fold),
        "years": report["protocol"]["split"],
        "protocol_problems": protocol_problems(report, fold),
        "best_epoch": report["best_epoch"],
        "steps_done": report["protocol"]["optimization"]["steps_done"],
        "amp_skipped_steps": report["protocol"]["optimization"].get("amp_skipped_steps"),
        "best_val_loss": report.get("best_val_loss"),
        "val_ap": report["validation"]["all"]["model"]["average_precision"],
        "pos_weight": report["pos_weight"]["pos_weight"],
        "official_table_pos_weight": report["pos_weight"].get("official_table_pos_weight"),
        "test_ap": metrics["model"]["average_precision"] if metrics else None,
        "test_ap_ci95": (metrics.get("ap_bootstrap_events") or {}).get("ci95") if metrics else None,
        "test_prevalence": metrics["model"].get("prevalence") if metrics else None,
        "test_f2_at_val_threshold": metrics["model_at_validation_threshold"]["f2"] if metrics else None,
        "test_persistence_f2": persistence["f2"] if persistence else None,
        "elapsed_hours": round(report["elapsed_seconds"] / 3600, 2),
        "code_commit": repo.get("commit"),
        "code_dirty": repo.get("dirty"),
    }


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Média e desvio-padrão amostral (ddof = 1) do AP de teste, no total e por ano de teste.

    Os 12 folds não são independentes (4 anos de teste, 3 folds cada, treinos sobrepostos): o desvio mede sobretudo a
    variação entre anos e não é um erro-padrão. Os folds com teste em 2021 (0, 2 e 6) são os únicos cujo ano de teste
    não participou da EDA nem das decisões de codificação, e aparecem à parte.
    """
    valid = [e for e in entries if e.get("test_ap") is not None and not e.get("protocol_problems")]
    by_year: dict[int, list[float]] = {}
    for entry in valid:
        by_year.setdefault(split_years(entry["fold"])["test"][0], []).append(entry["test_ap"])
    overall = _mean_std([e["test_ap"] for e in valid])
    return {
        "metric": "AP de teste na vista `all`, que com eval_region=crop32 é a região center_crop_x32 do artigo",
        "n_folds_tested": overall["n"],
        "n_folds_expected": len(FOLD_ORDER),
        "test_ap_mean": overall["mean"],
        "test_ap_std": overall["std"],
        "by_test_year": {str(year): _mean_std(values) for year, values in sorted(by_year.items())},
        "folds_with_test_2021": _mean_std(by_year.get(2021, [])),
        "complete": overall["n"] == len(FOLD_ORDER),
        "reading": "o desvio entre folds não é erro-padrão; comparações entre modelos devem ser pareadas por fold",
    }


def write_summary(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    entries = [entry for fold in FOLD_ORDER if (entry := fold_entry(fold)) is not None]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "12 folds oficiais por ano, receita do artigo (stconv_train.py --recipe paper), teste único por fold",
        "fold_order": list(FOLD_ORDER),
        "folds": entries,
        "failures": state["failures"],
        "aggregate": summarize(entries),
        "started_at_utc": state["started"],
        "sleep_guard": state["guard"],
        "power_at_start": state["power"],
    }
    write_json(payload, path)
    return payload


def run_fold(fold: int, args: argparse.Namespace, log: Log) -> str | None:
    """Treina e testa um fold; devolve `None` se terminou ou o motivo da falha."""
    name = run_name(fold)
    fold_log = args.log_dir / f"{name}.log"
    ensure_normalization(args.data_root, fold, log)
    attempt = 0
    while read_report(fold) is None:
        attempt += 1
        resume = (args.data_root / "runs" / name / "last.pt").exists()
        log(f"[treino] fold {fold}, tentativa {attempt}{' (retomando do last.pt)' if resume else ''} → {fold_log}")
        code = execute(train_command(fold, args, resume=resume), fold_log)
        if code == 0:
            break
        if attempt >= TRAIN_ATTEMPTS:
            return f"treino falhou {attempt} vezes (último código {code}); veja {fold_log}"
        log(f"  treino saiu com código {code}; nova tentativa em {RETRY_PAUSE_SECONDS} s")
        time.sleep(RETRY_PAUSE_SECONDS)
    report = read_report(fold) or {}
    problems = protocol_problems(report, fold)
    if problems:
        return f"o relatório {name}.json não segue o protocolo do baseline: {'; '.join(problems)}"
    if "test" not in report:
        pending = pending_test_report(RUNS_REPORT_DIR / f"{name}.json")
        if ledger_entries(TEST_LEDGER, name) and not pending.exists():
            return ("o livro-razão registra uma avaliação do teste, mas o relatório não a tem; "
                    "inspecione antes de qualquer --force")
        log(f"[teste]  fold {fold}")
        for attempt in range(1, 3):
            code = execute(test_command(fold), fold_log)
            if code == 0 and "test" in (read_report(fold) or {}):
                break
            if attempt == 2:
                return f"avaliação do teste falhou (código {code}); veja {fold_log}"
            time.sleep(RETRY_PAUSE_SECONDS)
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda o baseline STConvS2S (receita do artigo) nos 12 folds.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--folds", type=int, nargs="*", default=list(FOLD_ORDER), help="Subconjunto, na ordem dada.")
    parser.add_argument("--accumulate", type=int, default=2,
                        help="Micro-lotes por passo; o lote efetivo continua 32 (2 = micro-lote 16, ~3,6 GiB de VRAM).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-dir", type=Path, default=REPORTS_DIR / "logs")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    # O executor e os filhos precisam enxergar o mesmo diretório, de qualquer lugar de onde forem lançados.
    if not args.data_root.is_absolute():
        args.data_root = (REPO_ROOT / args.data_root).resolve()
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    unknown = sorted(set(args.folds) - set(FOLD_ORDER))
    if unknown:
        raise SystemExit(f"Folds inexistentes: {unknown}.")
    log = Log(args.log_dir / "baseline_runner.log")
    state: dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "power": power_status(), "failures": {}, "guard": None}
    if state["power"]["on_ac"] is False:
        log("ATENÇÃO: o computador está na bateria. Ligue na tomada antes de deixar o treino rodando.")
    with prevent_sleep() as guard:
        state["guard"] = dict(guard)
        log(f"Início | folds {args.folds} | dados {args.data_root} | bloqueio de suspensão: "
            f"{'ativo' if guard['active'] else guard.get('note', 'inativo')}")
        for fold in args.folds:
            began = time.perf_counter()
            try:
                failure = run_fold(fold, args, log)
            except Exception as error:  # noqa: BLE001 - um fold não pode derrubar os outros
                failure = f"{type(error).__name__}: {error}"
            if failure:
                state["failures"][str(fold)] = failure
                log(f"  fold {fold} NÃO concluído: {failure}")
            else:
                state["failures"].pop(str(fold), None)
            aggregate = write_summary(args.report, state)["aggregate"]
            done = fold_entry(fold) or {}
            log(f"  fold {fold}: AP teste {done.get('test_ap')} | {round((time.perf_counter() - began) / 3600, 2)} h | "
                f"média até aqui {aggregate['test_ap_mean']} ({aggregate['n_folds_tested']} folds)")
    state["guard"] = {**state["guard"], "released": True}
    summary = write_summary(args.report, state)
    aggregate = summary["aggregate"]
    log(f"Fim | AP de teste {aggregate['test_ap_mean']} ± {aggregate['test_ap_std']} em {aggregate['n_folds_tested']} "
        f"folds | falhas: {sorted(state['failures']) or 'nenhuma'} | {args.report}")
    return summary


if __name__ == "__main__":
    main()
