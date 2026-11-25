"""`T_in = 1` com ajuste de teste 5 (alvos de teste iguais aos do modelo de 5 dias) e o executor."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_experiment as experiment
from scripts import stconv_train as train
from scripts.stconv_dataset import WSTSWindowDataset
from tests.test_stconv_dataset import NORMALIZATION, _event

REPO_ROOT = Path(__file__).resolve().parents[1]
STCONV_AVAILABLE = (REPO_ROOT / ".stconvs2s" / "model" / "stconvs2s.py").exists()


def _targets(dataset: WSTSWindowDataset) -> list[tuple[int, int]]:
    """(evento, índice do dia-alvo) de cada janela: o alvo é o dia logo depois da entrada."""
    return [(event, start + dataset.t_in) for event, start in dataset.windows]


class TestAdjustmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.events = [_event(root, "fire_a", n_days=10, height=8, width=8, seed=1),
                       _event(root, "fire_b", n_days=7, height=8, width=8, seed=2)]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def dataset(self, t_in: int, adjustment: int) -> WSTSWindowDataset:
        skip = train.resolve_test_skip({"t_in": t_in, "test_adjustment": adjustment})["skip_initial"]
        return WSTSWindowDataset(self.events, NORMALIZATION, mode="eval", t_in=t_in, patch=8, halo=2,
                                 skip_initial=skip)

    def test_one_day_model_is_tested_on_the_same_target_days_as_the_five_day_model(self) -> None:
        five, one = self.dataset(5, 5), self.dataset(1, 5)
        self.assertEqual(_targets(one), _targets(five))
        self.assertEqual(_targets(five)[0], (0, 5))  # primeiro alvo = dia 5, nos dois
        five.close(), one.close()

    def test_without_adjustment_the_one_day_model_gets_four_extra_early_days_per_event(self) -> None:
        five, one = self.dataset(5, 0), self.dataset(1, 0)
        self.assertEqual(len(one.windows) - len(five.windows), 4 * len(self.events))
        self.assertTrue(set(_targets(five)) < set(_targets(one)))
        five.close(), one.close()

    def test_baseline_runs_without_the_key_skip_nothing(self) -> None:
        self.assertEqual(train.resolve_test_skip({"t_in": 5}), {"adjustment": 0, "skip_initial": 0})
        self.assertEqual(train.resolve_test_skip({"t_in": 5, "test_adjustment": 5})["skip_initial"], 0)
        self.assertEqual(train.resolve_test_skip({"t_in": 1, "test_adjustment": 5})["skip_initial"], 4)

    def test_flag_defaults_to_zero_and_rejects_an_adjustment_shorter_than_the_input(self) -> None:
        self.assertEqual(train.parse_args(["--recipe", "paper_es"]).test_adjustment, 0)
        args = train.parse_args(["--recipe", "paper_es", "--t-in", "1", "--test-adjustment", "5"])
        self.assertEqual((args.t_in, args.test_adjustment), (1, 5))
        with self.assertRaises(SystemExit):
            train.parse_args(["--recipe", "paper_es", "--t-in", "5", "--test-adjustment", "3"])

    def test_a_baseline_last_pt_without_the_key_still_resumes(self) -> None:
        current = vars(train.parse_args(["--recipe", "paper_es"]))
        legacy = {k: v for k, v in current.items() if k != "test_adjustment"}
        self.assertEqual(train.config_changes(legacy, current), [])
        changed = vars(train.parse_args(["--recipe", "paper_es", "--t-in", "1", "--test-adjustment", "5"]))
        self.assertIn("test_adjustment", train.config_changes(legacy, changed))


@unittest.skipUnless(STCONV_AVAILABLE, "clone .stconvs2s ausente")
class OneDayModelTests(unittest.TestCase):
    def test_stconvs2s_builds_and_backpropagates_with_a_single_input_day(self) -> None:
        import torch

        config = {"batch_size": 2, "patch": 16, "num_layers": 3, "hidden_dim": 8, "dropout": 0.0}
        model = train.build_model(6, 1, config, torch.device("cpu"))
        out = model(torch.randn(2, 6, 1, 16, 16))
        self.assertEqual(tuple(out.shape), (2, 1, 1, 16, 16))
        out[:, :, -1:].mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))


class ExperimentRunnerTests(unittest.TestCase):
    @staticmethod
    def report(**config: object) -> dict:
        return {"kind": "full", "config": config,
                "protocol": {"recipe": "paper_es", "eval_region": "crop32", "val_sampling": "train_crops",
                             "split_kind": "temporal", "fold": 3}}

    def test_command_is_the_baseline_one_plus_only_the_experiment_arguments(self) -> None:
        args = experiment.parse_args(["--experiment", "tin1_es"])
        command = experiment.train_command(3, args, resume=False)
        self.assertEqual(command[command.index("--recipe") + 1], "paper_es")
        self.assertEqual(command[command.index("--run-name") + 1], "tin1_es_f03")
        self.assertEqual(command[command.index("--t-in") + 1], "1")
        self.assertEqual(command[command.index("--test-adjustment") + 1], "5")
        self.assertEqual(command[command.index("--accumulate") + 1], "2")  # como o baseline
        self.assertIn("--resume", experiment.train_command(3, args, resume=True))
        self.assertEqual(experiment.test_command(3, "tin1_es")[-1], "tin1_es_f03")
        # Fold 7: a avaliação na CPU foi anulada e o nome antigo é intocável (teste único) → f07b.
        self.assertEqual(experiment.run_name(7, "tin1_es"), "tin1_es_f07b")
        self.assertEqual(experiment.test_command(7, "tin1_es")[-1], "tin1_es_f07b")
        self.assertTrue(experiment.run_name(7, "tin1_es").startswith(experiment.EXPERIMENTS["tin1_es"]["prefix"]))
        self.assertIn("--require-cuda", command)  # nunca cai na CPU em silêncio
        self.assertIn("--require-cuda", experiment.test_command(3, "tin1_es"))
        cpu = experiment.parse_args(["--experiment", "tin1_es", "--allow-cpu"])
        self.assertNotIn("--require-cuda", experiment.train_command(3, cpu, resume=False))

    def test_prefixes_and_summaries_never_collide_with_the_baseline(self) -> None:
        from scripts import run_baseline as baseline

        prefixes = [e["prefix"] for e in baseline.RECIPE_RUNS.values()] + [
            e["prefix"] for e in experiment.EXPERIMENTS.values()]
        self.assertFalse(any(a != b and a.startswith(b) for a in prefixes for b in prefixes))
        summaries = [e["summary"] for e in baseline.RECIPE_RUNS.values()] + [
            e["summary"] for e in experiment.EXPERIMENTS.values()]
        self.assertEqual(len(set(summaries)), len(summaries))

    def test_a_fold_counts_only_with_the_experiment_configuration(self) -> None:
        self.assertEqual(experiment.protocol_problems(self.report(t_in=1, test_adjustment=5), 3, "tin1_es"), [])
        self.assertTrue(experiment.protocol_problems(self.report(t_in=5, test_adjustment=0), 3, "tin1_es"))
        self.assertTrue(experiment.protocol_problems(self.report(t_in=1, test_adjustment=0), 3, "tin1_es"))
        tested = self.report(t_in=1, test_adjustment=5)
        tested["test"] = {"test_adjustment": {"adjustment": 0, "skip_initial": 0}}  # teste avaliado sem o ajuste
        self.assertTrue(experiment.protocol_problems(tested, 3, "tin1_es"))

    def test_failing_fold_is_retried_with_resume_and_the_next_fold_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--experiment", "tin1_es", "--log-dir", tmp, "--report", str(Path(tmp) / "s.json"),
                    "--folds", "2", "0", "--data-root", tmp]
            (Path(tmp) / "runs" / "tin1_es_f02").mkdir(parents=True)
            (Path(tmp) / "runs" / "tin1_es_f02" / "last.pt").write_bytes(b"")
            calls: list[list[str]] = []

            def failing(command: list[str], _log: Path) -> int:
                calls.append(command)
                return 1

            with mock.patch.object(experiment.baseline, "ensure_normalization"), \
                    mock.patch.object(experiment, "execute_with_watchdog",
                                      side_effect=lambda c, log, **_: failing(c, log)), \
                    mock.patch.object(experiment, "gpu_health", return_value=(True, "ok mock")), \
                    mock.patch.object(experiment, "read_report", return_value=None), \
                    mock.patch.object(experiment, "fold_entry", return_value=None), \
                    mock.patch.object(experiment.time, "sleep"), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = experiment.main(argv)
        attempts = experiment.baseline.TRAIN_ATTEMPTS
        self.assertEqual(len(calls), 2 * attempts)
        self.assertTrue(all("--resume" in c for c in calls[:attempts]))
        self.assertEqual(sorted(summary["failures"]), ["0", "2"])

    def test_runner_does_not_import_torch(self) -> None:
        code = "import sys; import scripts.run_experiment; print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
