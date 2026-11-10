"""Testes da comparação pareada `paper` × `paper_es` (bootstrap de eventos e teste de determinismo do prefixo)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import compare_recipes as compare

BINS = 8


def _events(n: int, separation: float, seed: int) -> np.ndarray:
    """Contagens `[evento, classe, bin]`: positivos deslocados `separation` bins acima dos negativos."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((n, 2, BINS), dtype=np.int64)
    for event in range(n):
        positive = np.clip(np.round(rng.normal(3 + separation, 1.0, 40)), 0, BINS - 1).astype(int)
        negative = np.clip(np.round(rng.normal(3, 1.0, 400)), 0, BINS - 1).astype(int)
        counts[event, 0] = np.bincount(positive, minlength=BINS)
        counts[event, 1] = np.bincount(negative, minlength=BINS)
    return counts


def _report(losses: list[float], cuts: list[int], best: int) -> dict:
    return {
        "history": [{"epoch": i + 1, "val_loss": v, "validated": True} for i, v in enumerate(losses)],
        "protocol": {"optimization": {"lr_reductions": [{"after_epoch": c} for c in cuts]}},
        "best_epoch": best,
    }


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_predictions_give_a_zero_interval(self) -> None:
        counts = _events(20, 2.0, seed=1)
        result = compare.paired_delta_bootstrap(counts, counts.copy(), n_boot=200)
        self.assertEqual(result["point_coarse_bins"], 0.0)
        self.assertEqual(result["ci95"], [0.0, 0.0])
        self.assertFalse(result["excludes_zero"])

    def test_a_better_model_on_the_same_events_excludes_zero(self) -> None:
        worse, better = _events(30, 1.0, seed=2), _events(30, 3.0, seed=2)
        result = compare.paired_delta_bootstrap(worse, better, n_boot=200)
        self.assertGreater(result["point_coarse_bins"], 0)
        self.assertGreater(result["ci95"][0], 0)
        self.assertTrue(result["excludes_zero"])
        self.assertEqual(result["share_below_zero"], 0.0)

    def test_the_resampling_is_reproducible(self) -> None:
        a, b = _events(15, 1.0, seed=3), _events(15, 1.5, seed=4)
        self.assertEqual(compare.paired_delta_bootstrap(a, b, n_boot=100),
                         compare.paired_delta_bootstrap(a, b, n_boot=100))

    def test_counts_over_different_events_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            compare.paired_delta_bootstrap(_events(10, 1.0, seed=5), _events(11, 1.0, seed=5))


class PrefixCheckTests(unittest.TestCase):
    def test_equal_losses_up_to_the_first_cut_are_identical(self) -> None:
        paper = _report([1.0, 0.9, 0.8, 0.85, 0.9, 0.7], cuts=[], best=6)
        paper_es = _report([1.0, 0.9, 0.8, 0.85, 0.95, 0.99], cuts=[4], best=3)
        result = compare.prefix_check(paper, paper_es)
        self.assertEqual((result["up_to_epoch"], result["epochs_compared"]), (4, 4))
        self.assertTrue(result["identical"])

    def test_a_difference_before_the_cut_is_reported(self) -> None:
        paper = _report([1.0, 0.9, 0.8, 0.85], cuts=[], best=3)
        paper_es = _report([1.0, 0.9, 0.8001, 0.85], cuts=[4], best=3)
        result = compare.prefix_check(paper, paper_es)
        self.assertFalse(result["identical"])
        self.assertAlmostEqual(result["max_abs_val_loss_diff"], 1e-4)


class CompareFoldTests(unittest.TestCase):
    """`compare_fold` sobre relatórios, contagens e checkpoints sintéticos num diretório temporário."""

    def write_run(self, root: Path, recipe: str, *, events: list[str], counts: np.ndarray, weight: float) -> dict:
        import torch

        from scripts.io_utils import sha256_file
        from scripts.run_baseline import run_name

        name = run_name(0, recipe)
        run_dir = root / "runs" / name
        run_dir.mkdir(parents=True)
        np.savez_compressed(run_dir / "test.npz", all_events=np.arange(len(counts)), all_logit_histograms=counts)
        torch.save({"model": {"w": torch.full((2,), weight)}, "config": {"recipe": recipe}}, run_dir / "best.pt")
        report = {
            "run_name": name, "best_epoch": 3, "checkpoint": f"runs/{name}/best.pt",
            "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
            "history": [{"epoch": e, "val_loss": 1.0 / e, "validated": True} for e in range(1, 6)],
            "protocol": {"split": {"test": [2021]}, "optimization": {"lr_reductions": [{"after_epoch": 4}]}},
            "test": {"events": events, "per_event_counts": f"runs/{name}/test.npz",
                     "metrics": {"all": {"model": {"average_precision": 0.3 if recipe == "paper" else 0.25}}}},
        }
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / f"{name}.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    def test_same_checkpoint_gives_a_zero_delta_and_equal_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = _events(12, 2.0, seed=7)
            for recipe in ("paper", "paper_es"):
                self.write_run(root, recipe, events=["a", "b"], counts=counts, weight=1.0)
            entry = compare.compare_fold(0, root / "reports", root)
            self.assertTrue(entry["same_weights"])
            self.assertTrue(entry["same_test_counts"])
            self.assertEqual(entry["bootstrap"]["ci95"], [0.0, 0.0])
            self.assertTrue(entry["prefix"]["identical"])
            self.assertEqual(entry["test_year"], "2021")
            summary = compare.summarize([entry])
            self.assertEqual(summary["mean_delta_by_test_year"]["2021"]["n"], 1)

    def test_different_weights_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "paper", events=["a"], counts=_events(12, 2.0, seed=8), weight=1.0)
            self.write_run(root, "paper_es", events=["a"], counts=_events(12, 1.0, seed=8), weight=2.0)
            self.assertFalse(compare.compare_fold(0, root / "reports", root)["same_weights"])

    def test_tested_events_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = _events(12, 2.0, seed=9)
            self.write_run(root, "paper", events=["a"], counts=counts, weight=1.0)
            self.write_run(root, "paper_es", events=["b"], counts=counts, weight=1.0)
            with self.assertRaises(ValueError):
                compare.compare_fold(0, root / "reports", root)

    def test_a_checkpoint_that_is_not_the_tested_one_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = _events(12, 2.0, seed=10)
            self.write_run(root, "paper", events=["a"], counts=counts, weight=1.0)
            report = self.write_run(root, "paper_es", events=["a"], counts=counts, weight=1.0)
            (root / report["checkpoint"]).write_bytes(b"outro checkpoint")
            with self.assertRaises(ValueError):
                compare.compare_fold(0, root / "reports", root)

    def test_no_valid_resample_is_an_explicit_error(self) -> None:
        empty = np.zeros((4, 2, BINS), dtype=np.int64)
        empty[:, 1] = 10  # só negativos: o AP não é definido
        with self.assertRaises(ValueError):
            compare.paired_delta_bootstrap(empty, empty.copy(), n_boot=20)


if __name__ == "__main__":
    unittest.main()
