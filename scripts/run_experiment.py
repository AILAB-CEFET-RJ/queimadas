"""Executor dos experimentos que comparam com o baseline: os mesmos 12 folds, outra configuração por vez.

É o `run_baseline.py` (retomável, novas tentativas com `--resume`, teste único por fold, resumo por ano de teste)
com o experimento escolhido num catálogo fixo (`EXPERIMENTS`), em vez de só a receita. Cada experimento tem
prefixo de run, log e resumo próprios, e um fold só entra na média se o relatório confirmar a receita, o protocolo
do baseline **e** a configuração do experimento (por exemplo `t_in = 1` com ajuste de teste 5).

Experimentos:

- `tin1_es` (2026-11-16): `paper_es` com 1 dia de entrada, comparável à linha de 1 dia da tabela 5 do
  WSTS. Única mudança em relação ao `baseline_es_*`: `--t-in 1`, mais `--test-adjustment 5`, que faz o teste pular os
  4 primeiros dias de cada evento, como o `n_leading_observations_test_adjustment` do oficial, para que os alvos de
  teste sejam os mesmos do modelo de 5 dias.

Proteções (2026-11-18, ver `gpu_guard.py`): treino e teste rodam com `--require-cuda` (sem cair na CPU em silêncio);
antes de cada tentativa o executor confere `nvidia-smi` e o torch, e um watchdog encerra o fold se o log ficar
`--stall-minutes` sem época nova. GPU indisponível ou fold travado param o executor inteiro, sem iniciar outros folds.

Uso (de qualquer pasta; no Linux, dentro de `tmux` ou `nohup`):

    python scripts/run_experiment.py --experiment tin1_es --data-root /caminho/wildfirespreadts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Só módulos sem torch: o executor fica vivo por dias e não deve ocupar a RAM que o treino usa.
from scripts import run_baseline as baseline  # noqa: E402
from scripts.core import DEFAULT_DATA_ROOT  # noqa: E402
from scripts.gpu_guard import (  # noqa: E402
    WATCHDOG_EXIT_CODE, GpuUnavailable, TrainingStalled, execute_with_watchdog, gpu_health,
)
from scripts.io_utils import power_status, prevent_sleep, write_json  # noqa: E402
from scripts.run_registry import REPORTS_DIR, RUNS_REPORT_DIR, TEST_LEDGER, ledger_entries, pending_test_report  # noqa: E402

FOLD_ORDER = baseline.FOLD_ORDER
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "tin1_es": {
        "description": "paper_es com T_in = 1 e ajuste de teste 5 (linha de 1 dia da tabela 5 do WSTS)",
        "script": "scripts/stconv_train.py",
        "recipe": "paper_es",
        "prefix": "tin1_es_f",
        "args": ["--t-in", "1", "--test-adjustment", "5"],
        "config": {"t_in": 1, "test_adjustment": 5},
        "summary": REPORTS_DIR / "tin1_es_summary.json",
        "runner_log": "tin1_es_runner.log",
        "paired_with": "baseline_es_summary.json",
        # 2026-11-18: o `tin1_es_f07` treinou e foi testado na CPU depois da queda da GPU0 (sem AMP, outro
        # determinismo). A avaliação foi anulada por decisão do Jota (entrada `annulment` no livro-razão, que é
        # append-only); pela regra do teste único o nome não pode ser retreinado, então o fold 7 roda como `f07b`.
        "run_name_overrides": {7: "tin1_es_f07b"},
    },
}


def run_name(fold: int, experiment: str) -> str:
    spec = EXPERIMENTS[experiment]
    return spec.get("run_name_overrides", {}).get(fold) or f"{spec['prefix']}{fold:02d}"


def train_command(fold: int, args: argparse.Namespace, *, resume: bool) -> list[str]:
    spec = EXPERIMENTS[args.experiment]
    command = [sys.executable, "-u", spec["script"], "--recipe", spec["recipe"], "--fold", str(fold),
               "--run-name", run_name(fold, args.experiment), "--accumulate", str(args.accumulate),
               "--num-workers", str(args.num_workers), "--data-root", str(args.data_root), *spec["args"]]
    command += [] if getattr(args, "allow_cpu", False) else ["--require-cuda"]
    return command + (["--resume"] if resume else [])


def test_command(fold: int, experiment: str, *, require_cuda: bool = True) -> list[str]:
    return [sys.executable, "-u", EXPERIMENTS[experiment]["script"], "--evaluate-test",
            *(["--require-cuda"] if require_cuda else []), "--run-name", run_name(fold, experiment)]


def check_gpu(args: argparse.Namespace, log: baseline.Log, step: str) -> None:
    """Para o executor (sem iniciar o passo) se o NVML ou o torch não enxergarem a GPU deste executor."""
    if args.allow_cpu:
        return
    ok, message = gpu_health()
    if not ok:
        raise GpuUnavailable(f"GPU indisponível antes de {step}: {message}")
    log(f"  [gpu] {message}")


def read_report(fold: int, experiment: str, report_dir: Path = RUNS_REPORT_DIR) -> dict[str, Any] | None:
    path = report_dir / f"{run_name(fold, experiment)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def protocol_problems(report: dict[str, Any], fold: int, experiment: str) -> list[str]:
    """Divergências do protocolo do baseline e da configuração do experimento; vazio quando o fold conta."""
    spec = EXPERIMENTS[experiment]
    problems = baseline.protocol_problems(report, fold, spec["recipe"])
    config = report.get("config", {})
    problems += [f"config.{key}={config.get(key)!r} (esperado {value!r})"
                 for key, value in spec["config"].items() if config.get(key) != value]
    test = report.get("test")
    if test is not None and "test_adjustment" in spec["config"]:
        expected = spec["config"]["test_adjustment"]
        got = (test.get("test_adjustment") or {}).get("adjustment")
        if got != expected:
            problems.append(f"test.test_adjustment={got!r} (esperado {expected!r})")
    return problems


def fold_entry(fold: int, experiment: str, report_dir: Path = RUNS_REPORT_DIR) -> dict[str, Any] | None:
    """Resumo de um fold, com os mesmos campos do baseline (para o pareamento por fold) e a configuração."""
    report = read_report(fold, experiment, report_dir)
    if report is None:
        return None
    test = report.get("test")
    metrics = test["metrics"]["all"] if test else None
    persistence = metrics["baselines"]["persistence_all_detections_t"] if metrics else None
    optimization = report["protocol"]["optimization"]
    repo = report.get("provenance", {}).get("repo", {})
    return {
        "fold": fold,
        "run_name": run_name(fold, experiment),
        "experiment": experiment,
        "recipe": EXPERIMENTS[experiment]["recipe"],
        "config": {key: report.get("config", {}).get(key) for key in EXPERIMENTS[experiment]["config"]},
        "years": report["protocol"]["split"],
        "protocol_problems": protocol_problems(report, fold, experiment),
        "best_epoch": report["best_epoch"],
        "epochs_done": optimization["epochs_done"],
        "steps_done": optimization["steps_done"],
        "stop_reason": optimization.get("stop_reason"),
        "amp_skipped_steps": optimization.get("amp_skipped_steps"),
        "best_val_loss": report.get("best_val_loss"),
        "val_ap": report["validation"]["all"]["model"]["average_precision"],
        "pos_weight": report["pos_weight"]["pos_weight"],
        "test_ap": metrics["model"]["average_precision"] if metrics else None,
        "test_ap_ci95": (metrics.get("ap_bootstrap_events") or {}).get("ci95") if metrics else None,
        "test_prevalence": metrics["model"].get("prevalence") if metrics else None,
        "test_f2_at_val_threshold": metrics["model_at_validation_threshold"]["f2"] if metrics else None,
        "test_persistence_f2": persistence["f2"] if persistence else None,
        "n_test_windows": test.get("n_test_windows") if test else None,
        "test_adjustment": test.get("test_adjustment") if test else None,
        "elapsed_hours": round(report["elapsed_seconds"] / 3600, 2),
        "code_commit": repo.get("commit"),
        "code_dirty": repo.get("dirty"),
    }


def write_summary(path: Path, state: dict[str, Any], experiment: str) -> dict[str, Any]:
    spec = EXPERIMENTS[experiment]
    entries = [entry for fold in FOLD_ORDER if (entry := fold_entry(fold, experiment)) is not None]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "description": spec["description"],
        "recipe": spec["recipe"],
        "extra_args": spec["args"],
        "paired_with": spec["paired_with"],
        "protocol": f"12 folds oficiais por ano, {spec['script']} --recipe {spec['recipe']} {' '.join(spec['args'])}, "
                    "teste único por fold",
        "fold_order": state.get("fold_order", list(FOLD_ORDER)),
        "folds": entries,
        "failures": state["failures"],
        "halted": state.get("halted"),
        "aggregate": baseline.summarize(entries),
        "started_at_utc": state["started"],
        "sleep_guard": state["guard"],
        "power_at_start": state["power"],
    }
    write_json(payload, path)
    return payload


def run_fold(fold: int, args: argparse.Namespace, log: baseline.Log) -> str | None:
    """Treina e testa um fold; devolve `None` se terminou ou o motivo da falha (mesma lógica do baseline)."""
    experiment = args.experiment
    name = run_name(fold, experiment)
    fold_log = args.log_dir / f"{name}.log"
    baseline.ensure_normalization(args.data_root, fold, log)
    attempt = 0
    while read_report(fold, experiment) is None:
        attempt += 1
        resume = (args.data_root / "runs" / name / "last.pt").exists()
        check_gpu(args, log, f"o treino do fold {fold} (tentativa {attempt})")
        log(f"[treino] fold {fold}, tentativa {attempt}{' (retomando do last.pt)' if resume else ''} → {fold_log}")
        code = execute_with_watchdog(train_command(fold, args, resume=resume), fold_log,
                                     stall_minutes=args.stall_minutes)
        if code == WATCHDOG_EXIT_CODE:
            raise TrainingStalled(f"fold {fold} sem época nova em {args.stall_minutes:g} min; processo encerrado "
                                  f"pelo watchdog (veja {fold_log})")
        if code == 0:
            break
        if attempt >= baseline.TRAIN_ATTEMPTS:
            return f"treino falhou {attempt} vezes (último código {code}); veja {fold_log}"
        pause = baseline.RETRY_PAUSES_SECONDS[attempt - 1]
        log(f"  treino saiu com código {code}; nova tentativa em {pause} s")
        time.sleep(pause)
    report = read_report(fold, experiment) or {}
    problems = [p for p in protocol_problems(report, fold, experiment) if not p.startswith("test.")]
    if problems:
        return f"o relatório {name}.json não segue o protocolo do experimento: {'; '.join(problems)}"
    if "test" not in report:
        pending = pending_test_report(RUNS_REPORT_DIR / f"{name}.json")
        if ledger_entries(TEST_LEDGER, name) and not pending.exists():
            return ("o livro-razão registra uma avaliação do teste, mas o relatório não a tem; "
                    "inspecione antes de qualquer --force")
        log(f"[teste]  fold {fold}")
        for attempt in range(1, 3):
            check_gpu(args, log, f"o teste do fold {fold}")
            code = baseline.execute(test_command(fold, experiment, require_cuda=not args.allow_cpu), fold_log)
            if code == 0 and "test" in (read_report(fold, experiment) or {}):
                break
            if attempt == 2:
                return f"avaliação do teste falhou (código {code}); veja {fold_log}"
            time.sleep(baseline.RETRY_PAUSE_SECONDS)
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda um experimento do catálogo nos 12 folds, pareado com o baseline.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--folds", type=int, nargs="*", default=list(FOLD_ORDER), help="Subconjunto, na ordem dada.")
    parser.add_argument("--accumulate", type=int, default=2,
                        help="Micro-lotes por passo; 2 (micro-lote 16) é o do baseline e deve ser mantido para parear.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-dir", type=Path, default=REPORTS_DIR / "logs")
    parser.add_argument("--report", type=Path, default=None, help="Padrão: o resumo do experimento.")
    parser.add_argument("--stall-minutes", type=float, default=45.0,
                        help="Watchdog: encerra o fold e o executor se o log ficar N min sem linha 'época' nova "
                             "(0 desliga). A época mais lenta na GPU levou ~8 min, a primeira inclui a carga dos dados.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Não exige CUDA nem checa a GPU antes de cada fold (só para depuração; runs na CPU "
                             "não são comparáveis aos da GPU).")
    args = parser.parse_args(argv)
    args.report = args.report or EXPERIMENTS[args.experiment]["summary"]
    if not args.data_root.is_absolute():
        args.data_root = (REPO_ROOT / args.data_root).resolve()
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    unknown = sorted(set(args.folds) - set(FOLD_ORDER))
    if unknown:
        raise SystemExit(f"Folds inexistentes: {unknown}.")
    log = baseline.Log(args.log_dir / EXPERIMENTS[args.experiment]["runner_log"])
    state: dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "power": power_status(), "failures": {}, "guard": None, "fold_order": list(args.folds)}
    if state["power"].get("on_ac") is False:
        log("ATENÇÃO: o computador está na bateria. Ligue na tomada antes de deixar o treino rodando.")
    with prevent_sleep() as guard:
        state["guard"] = dict(guard)
        log(f"Início | experimento {args.experiment} | folds {args.folds} | dados {args.data_root} | "
            f"bloqueio de suspensão: {'ativo' if guard['active'] else guard.get('note', 'inativo')}")
        for fold in args.folds:
            began = time.perf_counter()
            halt = None
            try:
                failure = run_fold(fold, args, log)
            except (GpuUnavailable, TrainingStalled) as error:
                # Sem GPU os folds seguintes também cairiam (ou rodariam na CPU): para o executor inteiro.
                failure = halt = f"{type(error).__name__}: {error}"
            except Exception as error:  # noqa: BLE001 - um fold não pode derrubar os outros
                failure = f"{type(error).__name__}: {error}"
            if halt:
                state["failures"][str(fold)] = failure
                state["halted"] = halt
                log(f"  PARADA: {halt}. Folds não iniciados: {args.folds[args.folds.index(fold) + 1:]}")
                write_summary(args.report, state, args.experiment)
                break
            if failure:
                state["failures"][str(fold)] = failure
                log(f"  fold {fold} NÃO concluído: {failure}")
            else:
                state["failures"].pop(str(fold), None)
            aggregate = write_summary(args.report, state, args.experiment)["aggregate"]
            done = fold_entry(fold, args.experiment) or {}
            log(f"  [{args.experiment}] fold {fold}: AP teste {done.get('test_ap')} | "
                f"{round((time.perf_counter() - began) / 3600, 2)} h | média até aqui {aggregate['test_ap_mean']} "
                f"({aggregate['n_folds_tested']} folds)")
    state["guard"] = {**state["guard"], "released": True}
    summary = write_summary(args.report, state, args.experiment)
    aggregate = summary["aggregate"]
    log(f"Fim{' (PARADO: ' + state['halted'] + ')' if state.get('halted') else ''} | experimento {args.experiment} | AP de teste {aggregate['test_ap_mean']} ± {aggregate['test_ap_std']} em "
        f"{aggregate['n_folds_tested']} folds | falhas: {sorted(state['failures']) or 'nenhuma'} | {args.report}")
    return summary


if __name__ == "__main__":
    main()
