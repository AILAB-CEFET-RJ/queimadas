"""Onde ficam os relatórios dos runs e o livro-razão do teste, sem importar torch.

Separado de `stconv_train.py` para que executores de longa duração (`run_baseline.py`, `run_preregistered.py`)
consultem o estado dos runs sem carregar o PyTorch na memória durante dias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
RUNS_REPORT_DIR = REPORTS_DIR / "stconv_runs"
DEMO_REPORT_DIR = REPORTS_DIR / "demo_runs"
TEST_LEDGER = REPORTS_DIR / "test_ledger.jsonl"


def ledger_entries(ledger: Path, run_name: str) -> list[dict[str, Any]]:
    """Linhas do livro-razão referentes a um run (avaliações anteriores do teste)."""
    if not ledger.exists():
        return []
    entries = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [entry for entry in entries if entry.get("run_name") == run_name]


def pending_test_report(report_path: Path) -> Path:
    """Relatório com o teste já avaliado, gravado antes do livro-razão e promovido depois dele."""
    return report_path.with_name(report_path.stem + ".test_pending.json")
