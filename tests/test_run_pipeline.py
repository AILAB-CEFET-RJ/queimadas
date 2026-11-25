"""Pipeline completo num comando (run_pipeline.py): estado das etapas, comandos e pacote de entrega. Sem rede nem GPU."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scripts import run_baseline as baseline
from scripts import run_pipeline as pipeline


def write_events(root: Path, n: int, *, sidecars: bool = True) -> None:
    for index in range(n):
        target = root / "hdf5" / str(2018 + index % 4) / f"fire_{index}.hdf5"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        if sidecars:
            target.with_suffix(".json").write_text("{}", encoding="utf-8")


def full_manifest(path: Path) -> Path:
    path.write_text(json.dumps({"full_run": True, "counts_match_zenodo": True, "n_events": pipeline.EXPECTED_EVENTS,
                                "n_images": pipeline.EXPECTED_IMAGES}), encoding="utf-8")
    return path


class ConversionStatusTests(unittest.TestCase):
    def test_complete_needs_all_events_sidecars_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "m.json"
            write_events(root, pipeline.EXPECTED_EVENTS)
            self.assertFalse(pipeline.conversion_status(root, manifest)["complete"])  # sem manifesto
            self.assertTrue(pipeline.conversion_status(root, full_manifest(manifest))["complete"])

    def test_partial_or_leftover_tmp_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = full_manifest(root / "m.json")
            write_events(root, pipeline.EXPECTED_EVENTS - 1)
            status = pipeline.conversion_status(root, manifest)
            self.assertFalse(status["complete"])
            self.assertEqual(status["with_sidecar"], pipeline.EXPECTED_EVENTS - 1)
            write_events(root, pipeline.EXPECTED_EVENTS)
            (root / "hdf5" / "2018" / "x.hdf5.tmp").write_bytes(b"")
            self.assertFalse(pipeline.conversion_status(root, manifest)["complete"])

    def test_manifest_of_partial_run_does_not_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_events(root, pipeline.EXPECTED_EVENTS)
            manifest = root / "m.json"
            manifest.write_text(json.dumps({"full_run": False, "counts_match_zenodo": None}), encoding="utf-8")
            self.assertFalse(pipeline.conversion_status(root, manifest)["complete"])


class DiskTests(unittest.TestCase):
    def test_need_shrinks_with_what_already_exists(self) -> None:
        empty = {"complete": False, "with_sidecar": 0}
        no_zip = {"size_bytes": None, "partial_bytes": 0}
        fresh = pipeline.disk_needed_gib(empty, no_zip)
        self.assertAlmostEqual(fresh, pipeline.ZIP_GIB + pipeline.HDF5_GIB + pipeline.MARGIN_GIB, places=1)
        half = pipeline.disk_needed_gib({"complete": False, "with_sidecar": pipeline.EXPECTED_EVENTS // 2},
                                        {"size_bytes": int(pipeline.ZIP_GIB * 2**30), "partial_bytes": 0})
        self.assertLess(half, pipeline.HDF5_GIB / 2 + pipeline.MARGIN_GIB + 0.2)
        self.assertEqual(pipeline.disk_needed_gib({"complete": True, "with_sidecar": 607}, no_zip), pipeline.MARGIN_GIB)

    def test_zip_verified_only_with_md5_manifest_and_same_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / pipeline.ZIP_NAME).write_bytes(b"abc")
            manifest = root / "d.json"
            self.assertFalse(pipeline.zip_status(root, manifest)["verified"])
            manifest.write_text(json.dumps({"md5_verified": True, "files": [
                {"key": pipeline.ZIP_NAME, "size_bytes": 3}]}), encoding="utf-8")
            self.assertTrue(pipeline.zip_status(root, manifest)["verified"])
            (root / "raw" / pipeline.ZIP_NAME).write_bytes(b"abcd")
            self.assertFalse(pipeline.zip_status(root, manifest)["verified"])


class CommandTests(unittest.TestCase):
    def test_experiment_uses_the_existing_runner_with_its_defaults(self) -> None:
        args = pipeline.parse_args(["--train", "tin1_es", "--folds", "3", "7", "--data-root", "/dados"])
        command = pipeline.train_command(args)
        self.assertIn("run_experiment.py", command[2])
        for flag, value in (("--experiment", "tin1_es"), ("--accumulate", "2"), ("--num-workers", "4"),
                            ("--stall-minutes", "45")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(command[command.index("--folds") + 1:command.index("--folds") + 3], ["3", "7"])
        self.assertNotIn("--allow-cpu", command)

    def test_baseline_recipe_goes_to_run_baseline_unchanged(self) -> None:
        args = pipeline.parse_args(["--train", "paper_es", "--folds", "2"])
        command = pipeline.train_command(args)
        self.assertIn("run_baseline.py", command[2])
        self.assertEqual(command[command.index("--recipe") + 1], "paper_es")
        self.assertNotIn("--stall-minutes", command)

    def test_default_folds_are_the_runner_order(self) -> None:
        self.assertEqual(pipeline.parse_args([]).folds, list(baseline.FOLD_ORDER))
        with self.assertRaises(SystemExit):
            pipeline.parse_args(["--folds", "12"])
        with self.assertRaises(SystemExit):
            pipeline.parse_args(["--train", "none", "--smoke"])

    def test_smoke_keeps_the_experiment_config_and_never_tests(self) -> None:
        args = pipeline.parse_args(["--train", "tin1_es", "--smoke"])
        command = pipeline.smoke_command(args, "smoke", Path("r.json"))
        self.assertEqual(command[command.index("--recipe") + 1], "paper_es")
        self.assertEqual(command[command.index("--t-in") + 1], "1")
        self.assertIn("--max-events", command)
        self.assertIn("--require-cuda", command)
        self.assertNotIn("--evaluate-test", command)

    def test_run_names_follow_the_catalog_including_overrides(self) -> None:
        self.assertEqual(pipeline.run_names("tin1_es", [3, 7]), ["tin1_es_f03", "tin1_es_f07b"])
        self.assertEqual(pipeline.run_names("paper_es", [2]), ["baseline_es_f02"])


class GpuProbeTests(unittest.TestCase):
    def probe(self, stdout: str, code: int = 0) -> dict:
        runner = mock.Mock(return_value=subprocess.CompletedProcess([], code, stdout=stdout, stderr="boom"))
        return pipeline.probe_torch(runner=runner)

    def test_unsupported_architecture_points_to_cuda130(self) -> None:
        probe = self.probe("PROBE " + json.dumps({
            "torch": "2.13.0+cu126", "cuda_build": "12.6", "cuda_available": True,
            "device": "NVIDIA GeForce RTX 5090", "capability": "sm_120", "arch_list": ["sm_86", "sm_90"],
            "arch_supported": False, "amp_conv3d": "RuntimeError: no kernel image is available"}))
        problems = pipeline.gpu_problems(probe)
        self.assertEqual(len(problems), 1)
        self.assertIn("requirements-cuda130.txt", problems[0])

    def test_healthy_probe_has_no_problem(self) -> None:
        probe = self.probe("noise\nPROBE " + json.dumps({"cuda_available": True, "arch_supported": True,
                                                         "amp_conv3d": "ok"}))
        self.assertEqual(pipeline.gpu_problems(probe), [])

    def test_crash_and_cpu_only_are_problems(self) -> None:
        self.assertIn("falhou", pipeline.gpu_problems(self.probe("", code=1))[0])
        self.assertIn("não enxerga CUDA", pipeline.gpu_problems({"cuda_available": False, "torch": "2.13.0+cpu"})[0])


class DataStepTests(unittest.TestCase):
    def test_complete_conversion_skips_download_and_conversion(self) -> None:
        args = pipeline.parse_args(["--train", "none"])
        log = mock.Mock()
        with mock.patch.object(pipeline, "conversion_status", return_value={"complete": True, "hdf5_files": 607}), \
                mock.patch.object(pipeline, "run_step") as step:
            pipeline.ensure_data(args, log)
        step.assert_not_called()

    def test_download_then_convert_and_failures_stop(self) -> None:
        args = pipeline.parse_args(["--train", "none"])
        states = iter([{"complete": False, "with_sidecar": 0}, {"complete": True, "hdf5_files": 607}])
        with mock.patch.object(pipeline, "conversion_status", side_effect=lambda *_: next(states)), \
                mock.patch.object(pipeline, "zip_status", return_value={"verified": False}), \
                mock.patch.object(pipeline, "run_step", return_value=0) as step:
            pipeline.ensure_data(args, mock.Mock())
        scripts = [call.args[0][2] for call in step.call_args_list]
        self.assertEqual([Path(s).name for s in scripts], ["download.py", "convert_hdf5.py"])
        with mock.patch.object(pipeline, "conversion_status", return_value={"complete": False, "with_sidecar": 0}), \
                mock.patch.object(pipeline, "zip_status", return_value={"verified": False}), \
                mock.patch.object(pipeline, "run_step", return_value=1):
            with self.assertRaises(pipeline.PipelineError):
                pipeline.ensure_data(args, mock.Mock())

    def test_delete_zip_only_after_complete_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = pipeline.parse_args(["--train", "none", "--delete-zip", "--data-root", tmp])
            zip_path = Path(tmp) / "raw" / pipeline.ZIP_NAME
            zip_path.parent.mkdir()
            zip_path.write_bytes(b"z")
            with mock.patch.object(pipeline, "conversion_status", return_value={"complete": False, "with_sidecar": 0}), \
                    mock.patch.object(pipeline, "zip_status", return_value={"verified": True, "size_bytes": 1}), \
                    mock.patch.object(pipeline, "run_step", return_value=1):
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.ensure_data(args, mock.Mock())
            self.assertTrue(zip_path.exists())
            with mock.patch.object(pipeline, "conversion_status", return_value={"complete": True, "hdf5_files": 607}):
                pipeline.ensure_data(args, mock.Mock())
            self.assertFalse(zip_path.exists())


class BundleAndLockTests(unittest.TestCase):
    def test_bundle_collects_summary_fold_reports_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            runs, logs = reports / "stconv_runs", reports / "logs"
            runs.mkdir()
            logs.mkdir()
            (runs / "tin1_es_f03.json").write_text("{}", encoding="utf-8")
            (runs / "tin1_es_f06.json").write_text("{}", encoding="utf-8")  # fora dos folds pedidos
            (logs / "tin1_es_f03.log").write_text("época 1", encoding="utf-8")
            (reports / "tin1_es_summary.json").write_text("{}", encoding="utf-8")
            (reports / "test_ledger.jsonl").write_text("", encoding="utf-8")
            args = pipeline.parse_args(["--train", "tin1_es", "--folds", "3"])
            patched = {**pipeline.experiment.EXPERIMENTS["tin1_es"], "summary": reports / "tin1_es_summary.json"}
            with mock.patch.multiple(pipeline, REPORTS_DIR=reports, RUNS_REPORT_DIR=runs,
                                     TEST_LEDGER=reports / "test_ledger.jsonl", REPO_ROOT=reports.parent), \
                    mock.patch.dict(pipeline.experiment.EXPERIMENTS, {"tin1_es": patched}):
                target = pipeline.bundle(args, mock.Mock(), "20261125_1200")
            with zipfile.ZipFile(target) as archive:
                names = set(archive.namelist())
        self.assertEqual(names, {"tin1_es_summary.json", "test_ledger.jsonl", "stconv_runs/tin1_es_f03.json",
                                 "logs/tin1_es_f03.log"})

    def test_second_instance_is_refused_and_lock_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with pipeline.SingleInstance(root):
                with self.assertRaises(pipeline.PipelineError):
                    with pipeline.SingleInstance(root):
                        pass
            self.assertFalse((root / ".pipeline.lock").exists())
            (root / ".pipeline.lock").write_text("999999999", encoding="utf-8")  # pid morto: trava velha
            with pipeline.SingleInstance(root):
                pass


if __name__ == "__main__":
    unittest.main()
