"""Testes da receita do artigo: aumento de dados com ângulos exatos, região `crop32` e treino por passos."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import core as core
from scripts import run_baseline as baseline
from scripts import stconv_train as train
from scripts.stconv_dataset import WSTSWindowDataset, augment_patch, center_crop32_bounds
from tests.test_stconv_dataset import NORMALIZATION, _event

REPO_ROOT = Path(__file__).resolve().parents[1]

SIZE = 16
COMBINATIONS = list(itertools.product((False, True), (False, True), range(4)))


def _plane(dc: float, dr: float) -> np.ndarray:
    rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    return dc * cols + dr * rows


def _gradient(z: np.ndarray) -> tuple[float, float]:
    d_row, d_col = np.gradient(z)
    return float(d_col.mean()), float(d_row.mean())


def _angle_diff(a: float, b: float, period: float) -> float:
    return abs((a - b + period / 2) % period - period / 2)


class AugmentationGeometryTests(unittest.TestCase):
    """Prova geométrica: o canal de ângulo transformado tem de descrever o campo transformado.

    Um plano de elevação cuja descida aponta para o aspect θ (norte verdadeiro, grade girada de γ) passa pelo flip
    ou rotação; o aspect recalculado do plano transformado tem de coincidir com o decodificado do sen/cos. Vale
    para as 16 combinações de flips e rotações e para γ ≠ 0, que é o caso que a fórmula do oficial erra.
    """

    def test_bearing_channels_follow_the_field(self) -> None:
        pairs = {"bearing": [(0, 1)], "axial": []}
        for gamma, theta in itertools.product((-13.28, 0.0, 7.5), (0.0, 35.0, 200.0, 333.0)):
            theta_grid = np.deg2rad(theta + gamma)
            # Descida na grade, em (d_coluna, d_linha) com a linha crescendo para o sul.
            down_c, down_r = np.sin(theta_grid), -np.cos(theta_grid)
            x = np.zeros((3, 1, SIZE, SIZE), dtype=np.float64)
            x[0], x[1] = np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))
            x[2, 0] = -_plane(down_c, down_r)
            for hflip, vflip, rotations in COMBINATIONS:
                with self.subTest(gamma=gamma, theta=theta, hflip=hflip, vflip=vflip, rotations=rotations):
                    out, _ = augment_patch(x, {}, pairs, gamma, hflip=hflip, vflip=vflip, rotations=rotations)
                    up_c, up_r = _gradient(out[2, 0])
                    expected = np.degrees(np.arctan2(-up_c, up_r)) - gamma  # descida = −subida; − γ volta ao norte
                    decoded = np.degrees(np.arctan2(out[0].mean(), out[1].mean()))
                    self.assertLess(_angle_diff(decoded, expected, 360.0), 1e-6)

    def test_axial_channels_follow_the_field(self) -> None:
        pairs = {"bearing": [], "axial": [(0, 1)]}
        for gamma, phi in itertools.product((-13.28, 0.0, 7.5), (-80.0, -10.0, 0.0, 45.0, 89.0)):
            phi_grid = np.deg2rad(phi - gamma)  # anti-horário desde leste: azimute + γ = ângulo − γ
            x = np.zeros((3, 1, SIZE, SIZE), dtype=np.float64)
            x[0], x[1] = np.sin(np.deg2rad(2 * phi)), np.cos(np.deg2rad(2 * phi))
            x[2, 0] = _plane(np.cos(phi_grid), -np.sin(phi_grid))
            for hflip, vflip, rotations in COMBINATIONS:
                with self.subTest(gamma=gamma, phi=phi, hflip=hflip, vflip=vflip, rotations=rotations):
                    out, _ = augment_patch(x, {}, pairs, gamma, hflip=hflip, vflip=vflip, rotations=rotations)
                    g_c, g_r = _gradient(out[2, 0])
                    expected = np.degrees(np.arctan2(-g_r, g_c)) + gamma
                    decoded = np.degrees(np.arctan2(out[0].mean(), out[1].mean())) / 2
                    self.assertLess(_angle_diff(decoded, expected, 180.0), 1e-6)

    def test_gamma_zero_reproduces_the_official_formulas(self) -> None:
        pairs = {"bearing": [(0, 1)], "axial": []}
        theta = 30.0
        x = np.zeros((2, 1, 4, 4))
        x[0], x[1] = np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))
        official = {(True, False, 0): 360 - theta, (False, True, 0): 180 - theta, (False, False, 1): theta - 90}
        for (hflip, vflip, rotations), expected in official.items():
            out, _ = augment_patch(x, {}, pairs, 0.0, hflip=hflip, vflip=vflip, rotations=rotations)
            decoded = np.degrees(np.arctan2(out[0, 0, 0, 0], out[1, 0, 0, 0]))
            self.assertLess(_angle_diff(decoded, expected, 360.0), 1e-9)

    def test_fields_move_with_the_input_and_missing_angles_stay_zero(self) -> None:
        pairs = {"bearing": [(0, 1)], "axial": []}
        x = np.zeros((3, 1, 4, 4), dtype=np.float32)
        x[2, 0, 0, 1] = 1.0  # marcador na entrada
        fields = {"y": np.zeros((1, 1, 4, 4), dtype=np.float32)}
        fields["y"][0, 0, 0, 1] = 1.0  # o mesmo pixel no alvo
        for hflip, vflip, rotations in COMBINATIONS:
            out, moved = augment_patch(x, fields, pairs, 9.0, hflip=hflip, vflip=vflip, rotations=rotations)
            np.testing.assert_array_equal(out[2, 0], moved["y"][0, 0])
            self.assertEqual(float(np.abs(out[:2]).max()), 0.0)  # sen = cos = 0 (sem dado) continua zero
        self.assertEqual(fields["y"][0, 0, 0, 1], 1.0)  # a entrada não é alterada no lugar


class DatasetRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_crop32_region_covers_only_the_center_crop_once(self) -> None:
        event = _event(self.root, "fire_crop", n_days=6, height=70, width=45, seed=3)
        dataset = WSTSWindowDataset([event], NORMALIZATION, mode="eval", t_in=5, patch=24, halo=4,
                                    eval_region="crop32")
        (r0, r1), (c0, c1) = center_crop32_bounds(70), center_crop32_bounds(45)
        covered = np.zeros((70, 45), dtype=int)
        for index in range(len(dataset)):
            item, sample = dataset[index], dataset.samples[index]
            valid = item["mask"][0, 0].numpy().astype(bool)
            rows, cols = np.nonzero(valid)
            covered[rows + sample.row - 4, cols + sample.col - 4] += 1
            self.assertTrue(item["crop32"][0, 0].numpy()[valid].all())
        dataset.close()
        expected = np.zeros_like(covered)
        expected[r0:r1, c0:c1] = 1
        np.testing.assert_array_equal(covered, expected)

    def test_crop32_model_sees_nothing_outside_the_crop(self) -> None:
        event = _event(self.root, "fire_outside", n_days=6, height=70, width=45, seed=4)
        (r0, _), (c0, _) = center_crop32_bounds(70), center_crop32_bounds(45)
        dataset = WSTSWindowDataset([event], NORMALIZATION, mode="eval", t_in=5, patch=24, halo=4,
                                    eval_region="crop32")
        first = next(i for i, s in enumerate(dataset.samples) if (s.row, s.col) == (r0, c0))
        x = dataset[first]["x"].numpy()
        dataset.close()
        # O halo do primeiro tile cai fora do recorte: tem de ser padding (zero), não a imagem vizinha.
        self.assertEqual(float(np.abs(x[:, :, :4, :]).max()), 0.0)
        self.assertEqual(float(np.abs(x[:, :, :, :4]).max()), 0.0)

    def test_augmented_training_keeps_shapes_mask_and_target(self) -> None:
        event = _event(self.root, "fire_aug", n_days=6, height=40, width=40, seed=5)
        plain = WSTSWindowDataset([event], NORMALIZATION, mode="train", t_in=5, patch=32, halo=4)
        augmented = WSTSWindowDataset([event], NORMALIZATION, mode="train", t_in=5, patch=32, halo=4, augment=True)
        a, b = plain[0], augmented[0]
        plain.close(), augmented.close()
        self.assertEqual(a["x"].shape, b["x"].shape)
        self.assertEqual(int(a["mask"].sum()), int(b["mask"].sum()))
        self.assertEqual(int(a["y"].sum()), int(b["y"].sum()))
        with self.assertRaises(ValueError):
            WSTSWindowDataset([event], NORMALIZATION, mode="eval", augment=True)


class RecipeTests(unittest.TestCase):
    def test_paper_recipe_matches_the_official_configuration(self) -> None:
        args = train.parse_args(["--recipe", "paper"])
        self.assertEqual((args.batch_size, args.max_steps, args.lr, args.weight_decay, args.seed),
                         (32, 10_000, 1e-2, 1e-2, 0))
        self.assertEqual((args.augment, args.eval_region, args.checkpoint_metric), (True, "crop32", "val_loss"))
        self.assertEqual((args.val_sampling, args.pos_weight_source), ("train_crops", "fire_rate"))
        self.assertTrue(args.include_forecast_precip)  # "todas as features"
        self.assertIsNone(args.epochs)

    def test_default_recipe_keeps_the_pre_registered_values(self) -> None:
        args = train.parse_args([])
        self.assertEqual((args.epochs, args.batch_size, args.lr, args.weight_decay, args.seed), (20, 8, 1e-3, 1e-4, 42))
        self.assertEqual((args.augment, args.eval_region, args.checkpoint_metric), (False, "full", "val_ap"))
        self.assertFalse(args.include_forecast_precip)

    def test_explicit_arguments_win_and_accumulation_must_divide_the_batch(self) -> None:
        args = train.parse_args(["--recipe", "paper", "--accumulate", "2", "--no-augment", "--max-steps", "5"])
        self.assertEqual((args.accumulate, args.augment, args.max_steps, args.batch_size), (2, False, 5, 32))
        with self.assertRaises(SystemExit):
            train.parse_args(["--recipe", "paper", "--accumulate", "3"])


class BaselineRunnerTests(unittest.TestCase):
    def test_all_twelve_folds_once_with_fold_two_first(self) -> None:
        self.assertEqual(sorted(baseline.FOLD_ORDER), list(range(12)))
        self.assertEqual(baseline.FOLD_ORDER[0], 2)

    def test_resume_flag_recipe_and_absolute_data_root_in_the_training_command(self) -> None:
        args = baseline.parse_args([])
        self.assertTrue(args.data_root.is_absolute())  # filho e executor enxergam o mesmo diretório
        command = baseline.train_command(5, args, resume=True)
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--recipe") + 1], "paper")
        self.assertEqual(command[command.index("--run-name") + 1], "baseline_paper_f05")
        self.assertNotIn("--resume", baseline.train_command(5, args, resume=False))

    def test_summary_uses_only_valid_tested_folds_sample_std_and_groups_by_test_year(self) -> None:
        entries = [{"fold": 2, "test_ap": 0.30, "protocol_problems": []},   # teste 2021
                   {"fold": 0, "test_ap": 0.40, "protocol_problems": []},   # teste 2021
                   {"fold": 1, "test_ap": 0.20, "protocol_problems": []},   # teste 2020
                   {"fold": 3, "test_ap": 0.90, "protocol_problems": ["recipe='x'"]},
                   {"fold": 4, "test_ap": None, "protocol_problems": []}]
        summary = baseline.summarize(entries)
        self.assertEqual(summary["n_folds_tested"], 3)
        self.assertAlmostEqual(summary["test_ap_mean"], 0.30)
        self.assertAlmostEqual(summary["test_ap_std"], float(np.std([0.30, 0.40, 0.20], ddof=1)))
        self.assertEqual(summary["folds_with_test_2021"]["n"], 2)
        self.assertAlmostEqual(summary["by_test_year"]["2020"]["mean"], 0.20)
        self.assertFalse(summary["complete"])

    def test_report_of_another_recipe_does_not_count(self) -> None:
        report = {"kind": "full", "protocol": {"recipe": "pre_registered", "eval_region": "crop32",
                                                "val_sampling": "train_crops", "split_kind": "temporal", "fold": 2}}
        self.assertTrue(baseline.protocol_problems(report, 2))
        report["protocol"]["recipe"] = "paper"
        self.assertEqual(baseline.protocol_problems(report, 2), [])
        self.assertTrue(baseline.protocol_problems(report, 3))

    def test_failing_fold_is_retried_with_resume_then_recorded_and_the_next_fold_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--log-dir", tmp, "--report", str(Path(tmp) / "s.json"), "--folds", "2", "0", "--data-root", tmp]
            (Path(tmp) / "runs" / "baseline_paper_f02").mkdir(parents=True)
            (Path(tmp) / "runs" / "baseline_paper_f02" / "last.pt").write_bytes(b"")
            calls: list[list[str]] = []

            def failing(command: list[str], _log: Path) -> int:
                calls.append(command)
                return 1

            with mock.patch.object(baseline, "ensure_normalization"), \
                    mock.patch.object(baseline, "execute", side_effect=failing), \
                    mock.patch.object(baseline, "read_report", return_value=None), \
                    mock.patch.object(baseline, "fold_entry", return_value=None), \
                    mock.patch.object(baseline.time, "sleep"), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = baseline.main(argv)
        self.assertEqual(len(calls), 2 * baseline.TRAIN_ATTEMPTS)  # o fold 2 falhou e o fold 0 ainda rodou
        self.assertTrue(all("--resume" in c for c in calls[: baseline.TRAIN_ATTEMPTS]))
        self.assertEqual(sorted(summary["failures"]), ["0", "2"])

    def test_runner_does_not_import_torch(self) -> None:
        code = "import sys; import scripts.run_baseline; print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")


class TrainingLoopTests(unittest.TestCase):
    """O laço de época com um modelo mínimo em CPU (o STConvS2S real não é necessário para contar passos)."""

    @staticmethod
    def _setup(seed: int = 0) -> tuple:
        import torch

        torch.manual_seed(seed)
        model = torch.nn.Conv3d(2, 1, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        return model, optimizer, scaler

    @staticmethod
    def _batches(n: int) -> list[dict]:
        import torch

        generator = torch.Generator().manual_seed(1)
        return [{"x": torch.randn(2, 2, 3, 4, 4, generator=generator),
                 "y": (torch.rand(2, 1, 1, 4, 4, generator=generator) > 0.7).float(),
                 "mask": torch.ones(2, 1, 1, 4, 4)} for _ in range(n)]

    @staticmethod
    def _run(batches: list[dict], accumulate: int, max_steps: int | None, model, optimizer, scaler) -> dict:
        import torch

        return train.train_one_epoch(model, batches, optimizer, scaler, device=torch.device("cpu"), use_amp=False,
                                     accumulate=accumulate, pos_weight=3.0, step=0, max_steps=max_steps)

    def test_steps_with_accumulation_and_leftover(self) -> None:
        result = self._run(self._batches(5), 2, None, *self._setup())
        self.assertEqual(result["step"], 3)  # 2 passos completos + 1 com o micro-lote que sobrou
        self.assertTrue(result["epoch_complete"])

    def test_max_steps_stops_mid_epoch_without_an_extra_step(self) -> None:
        result = self._run(self._batches(6), 2, 1, *self._setup())
        self.assertEqual((result["step"], result["batches"], result["epoch_complete"]), (1, 2, False))
        exact = self._run(self._batches(4), 2, 2, *self._setup())
        self.assertTrue(exact["epoch_complete"])  # parar no último lote ainda conta como época completa

    def test_leftover_step_uses_the_mean_gradient_of_what_is_left(self) -> None:
        import torch

        batches = self._batches(3)
        model, optimizer, scaler = self._setup()
        self._run(batches, 2, None, model, optimizer, scaler)
        reference, ref_optimizer, _ = self._setup()
        for group in ([batches[0], batches[1]], [batches[2]]):  # o que um lote parcial faria sem acumulação
            ref_optimizer.zero_grad()
            losses = [train.masked_bce_with_logits(reference(b["x"])[:, :, -1:], b["y"], b["mask"], 3.0) for b in group]
            (sum(losses) / len(losses)).backward()
            ref_optimizer.step()
        for got, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(got, expected)

    def test_config_fingerprint_ignores_only_infrastructure(self) -> None:
        base = {"lr": 0.01, "accumulate": 2, "num_workers": 4, "resume": True}
        self.assertEqual(train.config_fingerprint(base), {"lr": 0.01, "accumulate": 2})


class DurabilityTests(unittest.TestCase):
    def test_atomic_write_keeps_the_old_file_when_the_writer_fails(self) -> None:
        from scripts.io_utils import atomic_write

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text("antigo", encoding="utf-8")

            def broken(stream) -> None:
                stream.write(b"meio")
                raise RuntimeError("queda")

            with self.assertRaises(RuntimeError):
                atomic_write(path, broken)
            self.assertEqual(path.read_text(encoding="utf-8"), "antigo")
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["a.json"])

    def test_interrupted_test_evaluation_is_promoted_not_repeated(self) -> None:
        from scripts.run_registry import pending_test_report

        with tempfile.TemporaryDirectory() as tmp:
            report, ledger = Path(tmp) / "r.json", Path(tmp) / "ledger.jsonl"
            stamp = "2026-11-02T10:00:00+00:00"
            report.write_text(json.dumps({"kind": "full", "eligible_for_test": True}), encoding="utf-8")
            staged = {"kind": "full", "test": {"evaluated_at_utc": stamp,
                                               "metrics": {"all": {"model": {"average_precision": 0.3}}}}}
            pending_test_report(report).write_text(json.dumps(staged), encoding="utf-8")
            ledger.write_text(json.dumps({"run_name": "r", "evaluated_at_utc": stamp}) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                train.main(["--evaluate-test", "--run-name", "r", "--report", str(report), "--ledger", str(ledger)])
            self.assertIn("test", json.loads(report.read_text(encoding="utf-8")))
            self.assertFalse(pending_test_report(report).exists())

    @unittest.skipUnless(sys.platform == "win32", "bloqueio de suspensão só no Windows")
    def test_prevent_sleep_releases_even_on_error(self) -> None:
        import ctypes

        from scripts import io_utils

        calls: list[int] = []
        fake = mock.Mock()
        fake.kernel32.SetThreadExecutionState.side_effect = lambda flags: calls.append(flags) or 1
        with mock.patch.object(ctypes, "windll", fake):
            with self.assertRaises(RuntimeError):
                with io_utils.prevent_sleep() as status:
                    self.assertTrue(status["active"])
                    raise RuntimeError("falha no meio do treino")
        self.assertEqual(calls, [0x80000001, 0x80000000])  # pedido e liberação


if __name__ == "__main__":
    unittest.main()
