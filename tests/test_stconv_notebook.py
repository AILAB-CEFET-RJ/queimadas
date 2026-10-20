from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import build_stconv_notebook as builder
from scripts import stconv_train as train


def _run_report(run_name: str, *, seed: int, drop_groups: list[str], with_test: bool) -> dict:
    """Relatório mínimo com as chaves que o notebook lê, no mesmo formato de `run_training`."""
    rng = np.random.default_rng(seed)
    accumulator = train.EvaluationAccumulator()
    for event in range(4):
        labels = (rng.random((1, 1, 1, 8, 8)) < 0.2).astype(np.float32)
        logits = rng.normal(labels * 2.0, 1.0).astype(np.float32)
        fire = (rng.random(labels.shape) < 0.1).astype(np.float32)
        batch = {
            "y": labels, "fire_t": fire, "fire_last_t": fire, "fire_window": fire,
            "crop32": np.ones_like(labels), "mask": np.ones_like(labels), "event": np.array([event]),
        }
        accumulator.update(logits, batch)
    threshold_bin = accumulator.model["all"].summary()["best_threshold"]["bin"]
    metrics = accumulator.summary(threshold_bin, bootstrap=20)
    report = {
        "kind": "full",
        "eligible_for_test": True,
        "run_name": run_name,
        "best_epoch": 2,
        "n_train_windows": 120,
        "n_validation_windows": 40,
        "peak_vram_mib": 1458.6,
        "checkpoint_sha256": "a" * 64,
        "validation_threshold_bin": threshold_bin,
        "pos_weight": {"samples": 30, "positive": 458, "negative": 654902, "pos_weight": 1429.9},
        "history": [
            {"epoch": 1, "train_loss": 1.62, "val_ap": 0.0003, "val_ap_new_vs_t": 0.0002,
             "train_seconds": 6.1, "epoch_seconds": 22.6},
            {"epoch": 2, "train_loss": 1.49, "val_ap": 0.0005, "val_ap_new_vs_t": 0.0004,
             "train_seconds": 6.0, "epoch_seconds": 22.1},
        ],
        "config": {"seed": seed},
        "protocol": {
            "fold": 2,
            "split": {"train": [2018, 2020], "validation": [2019], "test": [2021]},
            "t_in": 5,
            "target": "fogo ativo binário em t+1",
            "readout": "logits[:, :, -1:] (convenção de leitura)",
            "train_sampling": "1 recorte/janela/época",
            "eval_tiling": "tiles 128 px com halo 8 px",
            "channels": ["ndvi", "wind_speed", "active_fire_binary"],
            "drop_groups": drop_groups,
            "views": train.VIEWS and {v: v for v in train.VIEWS},
            "baselines": train.BASELINES,
            "baseline_notes": "persistência 'última passagem' usa a hora máxima de t na imagem inteira",
            "determinism": {"seed": seed, "cudnn_deterministic": True},
        },
        "provenance": {
            "repo": {"commit": "0" * 40, "dirty": False},
            "stconvs2s": {"commit": "c" * 40},
            "torch": {"version": "2.13.0+cu126"},
        },
        "validation": metrics,
    }
    if with_test:
        report["test"] = {
            "evaluated_at_utc": "2026-10-01T00:00:00+00:00",
            "n_test_windows": 30,
            "threshold_bin_from_validation": threshold_bin,
            "metrics": metrics,
        }
    return report


class StconvNotebookBuilderTests(unittest.TestCase):
    def test_notebook_cells_are_valid_and_compile(self) -> None:
        notebook = builder.build()
        ids = [cell["id"] for cell in notebook.cells]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, cell["id"], "exec")

    def test_notebook_executes_on_minimal_runs(self) -> None:
        try:
            from nbclient import NotebookClient
        except ImportError:  # pragma: no cover
            self.skipTest("nbclient indisponível")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "reports" / "stconv_runs"
            runs_dir.mkdir(parents=True)
            # Um run completo com teste, um ablation e um smoke (que o notebook deve ignorar).
            for name, seed, dropped, tested in (
                ("principal_s42", 42, [], True),
                ("sem_veg_s42", 42, ["vegetation_index"], False),
            ):
                (runs_dir / f"{name}.json").write_text(
                    json.dumps(_run_report(name, seed=seed, drop_groups=dropped, with_test=tested), default=str),
                    encoding="utf-8",
                )
            (runs_dir / "smoke.json").write_text(json.dumps({"kind": "smoke"}), encoding="utf-8")
            ledger = root / "reports" / "test_ledger.jsonl"
            ledger.write_text(json.dumps({"run_name": "principal_s42", "test_ap": 0.12}) + "\n", encoding="utf-8")
            client = NotebookClient(
                builder.build(), timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}
            )
            try:
                client.execute()
            except Exception as error:  # noqa: BLE001
                message = str(error).lower()
                if "kernel" in message and "no such" in message:
                    self.skipTest(f"kernel indisponível: {error}")
                raise

    def test_builder_requires_a_full_run(self) -> None:
        try:
            from nbclient import NotebookClient
        except ImportError:  # pragma: no cover
            self.skipTest("nbclient indisponível")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports" / "stconv_runs").mkdir(parents=True)
            client = NotebookClient(
                builder.build(), timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}}
            )
            with self.assertRaises(Exception):
                client.execute()


if __name__ == "__main__":
    unittest.main()
