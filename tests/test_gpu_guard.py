"""Proteções contra GPU perdida e treino travado (gpu_guard.py, run_experiment.py, --require-cuda). Sem GPU real."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import gpu_guard as guard
from scripts import run_experiment as experiment


def fake_runner(nvidia: tuple[int, str], probe: tuple[int, str]):
    def run(command: list[str], _timeout: float) -> tuple[int, str]:
        return nvidia if command[0] == "nvidia-smi" else probe
    return run


class GpuHealthTests(unittest.TestCase):
    def test_healthy_gpu_passes(self) -> None:
        ok, message = guard.gpu_health(runner=fake_runner(
            (0, "1, NVIDIA GeForce RTX 2080 Ti, 00000000:02:00.0"), (0, "ok NVIDIA GeForce RTX 2080 Ti 2.0")))
        self.assertTrue(ok)
        self.assertIn("RTX 2080 Ti", message)

    def test_nvml_failure_is_unhealthy(self) -> None:
        # Exatamente o que o cefet devolveu depois da queda da GPU0 em 17/11.
        ok, message = guard.gpu_health(runner=fake_runner(
            (255, "Unable to determine the device handle for GPU0000:01:00.0: Unknown Error"), (0, "ok x 2.0")))
        self.assertFalse(ok)
        self.assertIn("nvidia-smi", message)

    def test_torch_without_cuda_is_unhealthy(self) -> None:
        ok, message = guard.gpu_health(runner=fake_runner(
            (0, "1, RTX, 02:00.0"), (1, "UserWarning: Can't initialize NVML\ntorch.cuda.is_available() é False")))
        self.assertFalse(ok)
        self.assertIn("is_available", message)


class WatchdogTests(unittest.TestCase):
    def run_child(self, code: str, stall_minutes: float) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "fold.log"
            result = guard.execute_with_watchdog([sys.executable, "-u", "-c", code], log, stall_minutes=stall_minutes,
                                                 poll_seconds=0.1, grace_seconds=5)
            return result, log.read_text(encoding="utf-8")

    def test_stalled_step_is_terminated(self) -> None:
        code, text = self.run_child("import time; print('época 1 (passo 1)', flush=True); time.sleep(60)", 0.02)
        self.assertEqual(code, guard.WATCHDOG_EXIT_CODE)
        self.assertIn("[watchdog]", text)

    def test_progressing_step_is_left_alone(self) -> None:
        child = "import time\nfor i in range(8):\n    print(f'época {i}', flush=True); time.sleep(0.3)\n"
        code, text = self.run_child(child, 0.02)  # 1,2 s de tolerância; uma época a cada 0,3 s
        self.assertEqual(code, 0)
        self.assertNotIn("[watchdog]", text)

    def test_exit_code_is_passed_through_and_zero_disables(self) -> None:
        self.assertEqual(self.run_child("import sys; sys.exit(3)", 0.02)[0], 3)
        self.assertEqual(self.run_child("import time; time.sleep(1.5)", 0)[0], 0)


class RunnerHaltTests(unittest.TestCase):
    def run_main(self, *, health: tuple[bool, str], step_code: int) -> tuple[dict, list[list[str]], str]:
        calls: list[list[str]] = []

        def step(command: list[str], _log: Path, **_: object) -> int:
            calls.append(command)
            return step_code

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "s.json"
            argv = ["--experiment", "tin1_es", "--log-dir", tmp, "--report", str(report), "--folds", "2", "0",
                    "--data-root", tmp]
            with mock.patch.object(experiment.baseline, "ensure_normalization"), \
                    mock.patch.object(experiment, "execute_with_watchdog", side_effect=step), \
                    mock.patch.object(experiment, "gpu_health", return_value=health), \
                    mock.patch.object(experiment, "read_report", return_value=None), \
                    mock.patch.object(experiment, "fold_entry", return_value=None), \
                    mock.patch.object(experiment.time, "sleep"), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = experiment.main(argv)
            runner_log = (Path(tmp) / "tin1_es_runner.log").read_text(encoding="utf-8")
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["halted"], summary["halted"])
        return summary, calls, runner_log

    def test_unhealthy_gpu_stops_the_runner_before_any_training(self) -> None:
        summary, calls, runner_log = self.run_main(health=(False, "nvidia-smi falhou"), step_code=0)
        self.assertEqual(calls, [])
        self.assertIn("GpuUnavailable", summary["halted"])
        self.assertEqual(sorted(summary["failures"]), ["2"])  # o fold 0 nem começou
        self.assertIn("Folds não iniciados: [0]", runner_log)

    def test_watchdog_stops_the_runner_without_retrying(self) -> None:
        summary, calls, _ = self.run_main(health=(True, "ok"), step_code=guard.WATCHDOG_EXIT_CODE)
        self.assertEqual(len(calls), 1)
        self.assertIn("TrainingStalled", summary["halted"])
        self.assertIn("--require-cuda", calls[0])


class RequireCudaTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from scripts import stconv_train
        except ImportError as error:  # pragma: no cover - ambiente sem torch
            self.skipTest(str(error))
        self.train = stconv_train

    def test_without_cuda_require_cuda_aborts_instead_of_using_the_cpu(self) -> None:
        args = self.train.parse_args(["--recipe", "paper_es", "--require-cuda"])
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(SystemExit) as raised:
                self.train.select_device(args)
            self.assertIn("--require-cuda", str(raised.exception))
            self.assertEqual(self.train.select_device(self.train.parse_args(["--recipe", "paper_es"])).type, "cpu")

    def test_require_cuda_conflicts_with_cpu_and_does_not_block_resume(self) -> None:
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.train.parse_args(["--recipe", "paper_es", "--require-cuda", "--cpu"])
        self.assertIn("require_cuda", self.train.RESUME_VOLATILE)  # last.pt sem a flag continua retomável

    def test_piml_trainer_inherits_the_flag(self) -> None:
        from scripts import stconv_train_piml as piml

        _, rest = piml.split_argv(["--recipe", "paper_es", "--require-cuda", "--run-name", "piml_x"])
        self.assertTrue(self.train.parse_args(rest).require_cuda)


if __name__ == "__main__":
    unittest.main()
