"""Executa os 9 runs pré-registrados do STConvS2S, em sequência.

Configurações do pré-registro: principal, sem NDVI/EVI2 e sem os
canais GFS, cada uma com as seeds 42, 43 e 44. Cada run treina por `--epochs` e, com `--evaluate-test`,
avalia o teste **uma única vez** logo depois.

É retomável: um run cujo relatório já existe é pulado, então reexecutar o script depois de uma interrupção
continua de onde parou, sem retreinar nada.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import write_json
from scripts.core import PREREGISTERED_EPOCHS
from scripts.stconv_train import DEMO_REPORT_DIR, RUNS_REPORT_DIR, ledger_entries

TEST_LEDGER = REPO_ROOT / "reports" / "test_ledger.jsonl"
DEFAULT_REPORT = Path("reports/preregistered_runs.json")
CONFIGURATIONS = {"principal": [], "sem_veg": ["vegetation_index"], "sem_gfs": ["gfs"]}
SEEDS = (42, 43, 44)


def run_name(configuration: str, seed: int, split_kind: str = "temporal") -> str:
    """Nome do run, com sufixo do protocolo quando não é o temporal.

    O sufixo existe para que rodar os dois protocolos não sobrescreva relatórios nem torne o livro-razão
    ambíguo: `principal_s42` é o fold temporal e `principal_s42_st` o espaço-temporal.
    """
    suffix = "" if split_kind == "temporal" else "_st"
    return f"{configuration}_s{seed}{suffix}"


def python_executable() -> str:
    return sys.executable


def train_command(name: str, dropped: list[str], seed: int, args: argparse.Namespace) -> list[str]:
    command = [
        python_executable(), "-u", "scripts/stconv_train.py",
        "--run-name", name, "--seed", str(seed), "--epochs", str(args.epochs),
        "--num-workers", str(args.num_workers), "--split-kind", args.split_kind,
    ]
    if args.max_events is not None:
        command += ["--max-events", str(args.max_events)]
    if dropped:
        command += ["--drop-groups", *dropped]
    return command


def report_path(name: str, max_events: int | None) -> Path:
    directory = DEMO_REPORT_DIR if max_events is not None else RUNS_REPORT_DIR
    return directory / f"{name}.json"


def already_trained(name: str, max_events: int | None = None) -> bool:
    report = report_path(name, max_events)
    return report.exists() and "best_epoch" in json.loads(report.read_text(encoding="utf-8"))


def already_tested(name: str) -> bool:
    report = RUNS_REPORT_DIR / f"{name}.json"
    if report.exists() and "test" in json.loads(report.read_text(encoding="utf-8")):
        return True
    return bool(ledger_entries(TEST_LEDGER, name))


def execute(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        return subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda os 9 runs pré-registrados em sequência.")
    parser.add_argument("--epochs", type=int, default=PREREGISTERED_EPOCHS,
                        help="Orçamento pré-registrado (PREREGISTERED_EPOCHS), fixado antes de qualquer métrica.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split-kind", choices=("temporal", "spatiotemporal"), default="temporal",
                        help="Protocolo dos runs. No espaço-temporal a avaliação mede teste, só futuro e só "
                             "locais novos numa única passada, e os nomes dos runs ganham o sufixo `_st`.")
    parser.add_argument("--evaluate-train-cell", action="store_true",
                        help="Repassado à avaliação: mede também a célula de treino (dentro da amostra) para "
                             "fechar a interação dos deslocamentos.")
    parser.add_argument("--log-dir", type=Path, default=Path("reports/logs"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-test", action="store_true", help="Só treina; a avaliação do teste fica para depois.")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Demonstração: N eventos por ano. Os runs ficam inelegíveis para o teste.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    started = time.perf_counter()
    results = []
    for configuration, dropped in CONFIGURATIONS.items():
        for seed in SEEDS:
            name = run_name(configuration, seed, args.split_kind)
            log = args.log_dir / f"{name}.log"
            entry = {"run_name": name, "configuration": configuration, "drop_groups": dropped, "seed": seed}

            if already_trained(name, args.max_events):
                entry["training"] = "já existia"
            else:
                print(f"[treino] {name}", flush=True)
                began = time.perf_counter()
                code = execute(train_command(name, dropped, seed, args), log)
                entry["training"] = "ok" if code == 0 else f"falhou (código {code})"
                entry["training_seconds"] = round(time.perf_counter() - began, 1)
                if code != 0:
                    results.append(entry)
                    print(f"  ABORTADO: veja {log}", flush=True)
                    break

            if args.max_events is not None:
                entry["test"] = "não se aplica (demonstração)"
            elif args.skip_test:
                entry["test"] = "pulado"
            elif already_tested(name):
                entry["test"] = "já avaliado"
            else:
                print(f"[teste]  {name}", flush=True)
                evaluate = [python_executable(), "-u", "scripts/stconv_train.py",
                            "--evaluate-test", "--run-name", name]
                if args.evaluate_train_cell:
                    evaluate.append("--evaluate-train-cell")
                code = execute(evaluate, log)
                entry["test"] = "ok" if code == 0 else f"falhou (código {code})"
            results.append(entry)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "demonstração" if args.max_events is not None else "pré-registrado",
        "split_kind": args.split_kind,
        "max_events_per_year": args.max_events,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "configurations": {name: dropped for name, dropped in CONFIGURATIONS.items()},
        "seeds": list(SEEDS),
        "runs": results,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    write_json(report, args.report)
    done = sum(1 for r in results if r.get("training") in {"ok", "já existia"})
    print(f"\n{done}/{len(CONFIGURATIONS) * len(SEEDS)} runs treinados | relatório: {args.report}")
    return report


if __name__ == "__main__":
    main()
