from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import core as core
from scripts import stconv_train as train
from scripts.stconv_dataset import (
    WSTSWindowDataset,
    center_crop32_bounds,
    choose_training_crop,
    last_pass_mask,
    receptive_radius,
    tile_origins,
)


def _event(root: Path, name: str, n_days: int, height: int, width: int, seed: int, year: int = 2018) -> core.EventRef:
    rng = np.random.default_rng(seed)
    frames = rng.normal(10.0, 2.0, size=(n_days, core.N_RAW_CHANNELS, height, width)).astype(np.float32)
    frames[:, core.LANDCOVER_CHANNEL] = 9.0
    frames[:, core.ACTIVE_FIRE_CHANNEL] = 0.0
    # Fogo "marcador" só no último dia: não pode aparecer em nenhuma entrada.
    frames[-1, core.ACTIVE_FIRE_CHANNEL, height // 2, width // 2] = 21.0
    # Marcador de entrada fixo no canto (0, 0) em todos os dias anteriores.
    frames[: n_days - 1, core.ACTIVE_FIRE_CHANNEL, 0, 0] = 9.0
    path = root / str(year) / f"{name}.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with core.open_h5(path, "w") as handle:
        data = handle.create_dataset("data", data=frames, chunks=(1, core.N_RAW_CHANNELS, height, width))
        data.attrs["year"] = year
        data.attrs["fire_name"] = name
        data.attrs["img_dates"] = [f"{year}-07-{d + 1:02d}" for d in range(n_days)]
        data.attrs["lnglat"] = [-120.0, 38.0]
    return core.read_event_ref(path)


NORMALIZATION = {"channels": {core.CHANNEL_NAMES[c]: {"mean": 10.0, "std": 2.0} for c in core.CONTINUOUS_CHANNELS}}


class DatasetContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_train_sample_shapes_mask_and_padding(self) -> None:
        event = _event(self.root, "fire_small", n_days=6, height=20, width=30, seed=0)
        dataset = WSTSWindowDataset([event], NORMALIZATION, mode="train", t_in=5, patch=32, halo=4)
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        n_channels = len(core.input_channel_names())
        self.assertEqual(tuple(item["x"].shape), (n_channels, 5, 32, 32))
        for key in ("y", "mask", "fire_t", "fire_last_t", "fire_window", "crop32"):
            self.assertEqual(tuple(item[key].shape), (1, 1, 32, 32))
        self.assertEqual(int(item["mask"].sum()), 20 * 30)
        self.assertEqual(int(item["y"].sum()), 1)
        self.assertEqual(float(item["mask"][item["y"].bool()].min()), 1.0)
        self.assertGreater(int(((item["mask"] == 1) & (item["y"] == 0)).sum()), 0)
        dataset.close()

    def test_target_day_never_leaks_into_inputs(self) -> None:
        event = _event(self.root, "fire_leak", n_days=6, height=16, width=16, seed=1)
        dataset = WSTSWindowDataset([event], NORMALIZATION, mode="eval", t_in=5, patch=24, halo=4)
        fire_channel = core.input_channel_names().index("active_fire_binary")
        seen_input = torch.zeros(16, 16)
        seen_target = torch.zeros(16, 16)
        for index in range(len(dataset)):
            item = dataset[index]
            sample = dataset.samples[index]
            r0, c0 = sample.row - 4, sample.col - 4
            valid = item["mask"][0, 0].bool()
            rows, cols = torch.nonzero(valid, as_tuple=True)
            seen_input[rows + r0, cols + c0] += item["x"][fire_channel].sum(0)[rows, cols]
            seen_target[rows + r0, cols + c0] += item["y"][0, 0][rows, cols]
            self.assertTrue(torch.equal(item["fire_t"][0, 0], item["x"][fire_channel, -1]))
        self.assertEqual(float(seen_target[8, 8]), 1.0)
        self.assertEqual(float(seen_input[8, 8]), 0.0)
        self.assertEqual(float(seen_input[0, 0]), 5.0)
        dataset.close()

    def test_eval_tiles_reassemble_full_image(self) -> None:
        event = _event(self.root, "fire_tiles", n_days=6, height=37, width=50, seed=2)
        dataset = WSTSWindowDataset([event], NORMALIZATION, mode="eval", t_in=5, patch=24, halo=4)
        core_size = 24 - 8
        self.assertEqual(len(dataset), len(tile_origins(37, core_size)) * len(tile_origins(50, core_size)))
        full_x = core.transform_frames(core.read_days(event.path, 0, 5), NORMALIZATION)
        assembled = np.zeros_like(full_x)
        coverage = np.zeros((37, 50))
        for index in range(len(dataset)):
            item = dataset[index]
            sample = dataset.samples[index]
            valid = item["mask"][0, 0].numpy().astype(bool)
            rows, cols = np.nonzero(valid)
            assembled[:, :, rows + sample.row - 4, cols + sample.col - 4] = item["x"].numpy()[:, :, rows, cols]
            coverage[rows + sample.row - 4, cols + sample.col - 4] += 1
        np.testing.assert_array_equal(coverage, 1)
        np.testing.assert_allclose(assembled, full_x)
        dataset.close()

    def test_windows_skip_date_gaps(self) -> None:
        event = _event(self.root, "fire_gap", n_days=8, height=8, width=8, seed=3)
        with core.open_h5(event.path, "a") as handle:
            handle["data"].attrs["img_dates"] = [f"2018-07-{d:02d}" for d in (1, 2, 3, 4, 5, 6, 8, 9)]
        event = core.read_event_ref(event.path)
        dataset = WSTSWindowDataset([event], NORMALIZATION, mode="train", t_in=5, patch=8, halo=2)
        self.assertEqual(dataset.windows, [(0, 0)])

    def test_drop_groups_remove_channels_and_flags(self) -> None:
        event = _event(self.root, "fire_drop", n_days=6, height=8, width=8, seed=4)
        dataset = WSTSWindowDataset([event], NORMALIZATION, mode="train", t_in=5, patch=8, halo=2, drop_groups=("vegetation_index", "gfs"))
        names = dataset.channel_names
        for removed in ("ndvi", "evi2", "missing_vegetation_index", "forecast_wind_speed", "forecast_wind_direction_sin2", "missing_gfs"):
            self.assertNotIn(removed, names)
        self.assertEqual(dataset[0]["x"].shape[0], len(names))
        with self.assertRaises(ValueError):
            core.input_channel_names(drop_groups=("topography",))
        dataset.close()

    def test_training_crop_prefers_target_fire(self) -> None:
        fire_next = np.zeros((64, 64), dtype=np.uint8)
        fire_next[:10, :10] = 1
        rows = [choose_training_crop(np.zeros((5, 64, 64)), fire_next, 16, np.random.default_rng(s))[0] for s in range(20)]
        self.assertLess(np.mean(rows), 24)

    def test_last_pass_and_center_crop_helpers(self) -> None:
        hours = np.array([[0, 9, 21], [21, 20, np.nan]], dtype=np.float32)
        np.testing.assert_array_equal(last_pass_mask(hours), [[0, 0, 1], [1, 0, 0]])
        np.testing.assert_array_equal(last_pass_mask(np.zeros((2, 2))), 0)
        self.assertEqual(center_crop32_bounds(300), (6, 294))
        self.assertEqual(center_crop32_bounds(64), (0, 64))
        self.assertEqual(receptive_radius(3), 4)


class LossAndMetricTests(unittest.TestCase):
    def test_masked_pixels_do_not_contribute(self) -> None:
        logits = torch.tensor([[[[[5.0, -5.0]]]]])
        target = torch.tensor([[[[[0.0, 0.0]]]]])
        mask = torch.tensor([[[[[0.0, 1.0]]]]])
        loss = train.masked_bce_with_logits(logits, target, mask, pos_weight=10.0)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(-5.0), torch.tensor(0.0))
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_logit_histogram_ap_is_accurate_for_low_scores(self) -> None:
        from sklearn.metrics import average_precision_score

        rng = np.random.default_rng(0)
        labels = rng.random(200000) < 1e-3
        logits = np.where(labels, rng.normal(-7, 1, labels.size), rng.normal(-12, 1, labels.size))
        histogram = train.LogitHistogram()
        histogram.update(logits[:100000], labels[:100000])
        histogram.update(logits[100000:], labels[100000:])
        self.assertAlmostEqual(histogram.summary()["average_precision"], average_precision_score(labels, logits), delta=5e-4)

    def test_threshold_bin_reproduces_best_point(self) -> None:
        rng = np.random.default_rng(1)
        labels = rng.random(5000) < 0.05
        logits = rng.normal(np.where(labels, 1.0, -1.0), 1.0)
        histogram = train.LogitHistogram()
        histogram.update(logits, labels)
        summary = histogram.summary()
        at_best = histogram.at_bin(summary["best_threshold"]["bin"])
        self.assertAlmostEqual(at_best["precision"], summary["precision_at_best"])
        self.assertAlmostEqual(at_best["recall"], summary["recall_at_best"])
        self.assertAlmostEqual(at_best["f2"], summary["best_f2"])

    def test_accumulator_views_baselines_and_events(self) -> None:
        accumulator = train.EvaluationAccumulator()
        shape = (2, 1, 1, 1, 4)
        batch = {
            "y": np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=np.float32).reshape(shape),
            "fire_t": np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32).reshape(shape),
            "fire_last_t": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(shape),
            "fire_window": np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32).reshape(shape),
            "crop32": np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32).reshape(shape),
            "mask": np.ones(shape, dtype=np.float32),
            "event": np.array([0, 1]),
        }
        logits = np.array([3, 2, -3, -2, 1, -1, -4, -5], dtype=np.float32).reshape(shape)
        accumulator.update(logits, batch)
        summary = accumulator.summary()
        persistence = summary["all"]["baselines"]["persistence_all_detections_t"]
        self.assertEqual((persistence["tp"], persistence["fp"], persistence["fn"]), (1, 1, 2))
        self.assertEqual(summary["all"]["baselines"]["persistence_last_pass_t"]["tp"], 1)
        self.assertEqual(summary["crop32"]["model"]["n_positive"], 2)
        self.assertEqual(summary["new_vs_t"]["model"]["n_positive"], 2)
        self.assertEqual(summary["new_vs_window"]["model"]["n_positive"], 1)
        self.assertAlmostEqual(summary["all"]["model"]["average_precision"], 1.0)
        self.assertEqual(sorted(accumulator.per_event["all"]), [0, 1])
        self.assertIsNone(accumulator.bootstrap_ap("all", n_boot=10))

    def test_pos_weight_requires_positives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            event = _event(Path(tmp), "fire_none", n_days=6, height=8, width=8, seed=5)
            with core.open_h5(event.path, "a") as handle:
                handle["data"][-1, core.ACTIVE_FIRE_CHANNEL] = 0.0
            dataset = WSTSWindowDataset([event], NORMALIZATION, mode="train", t_in=5, patch=8, halo=2)
            with self.assertRaises(ValueError):
                train.estimate_pos_weight(dataset, 5, 0)
            dataset.close()


class ShiftDecompositionTests(unittest.TestCase):
    """A decomposição é o que transforma um AP de teste em "de onde vem a perda"; não pode inverter de sinal."""

    @staticmethod
    def _cells(**aps: float) -> dict:
        return {cell: {"metrics": {"all": {"model": {"average_precision": value}}}, "in_sample": cell == "train"}
                for cell, value in aps.items()}

    def test_cada_custo_compara_a_celula_certa(self) -> None:
        out = train.shift_decomposition(self._cells(test=0.20, future_only=0.30, space_only=0.26))
        # Lugar novo custa o que se ganha ao voltar para a geografia de treino, no mesmo ano de teste.
        self.assertAlmostEqual(0.10, out["cost_of_new_places"])
        # Período novo custa o que se ganha ao voltar ao período de treino, nos mesmos blocos de teste.
        self.assertAlmostEqual(0.06, out["cost_of_new_period"])
        self.assertIsNone(out["interaction"])
        self.assertIn("lugar novo", out["reading"])

    def test_interacao_so_aparece_com_a_celula_de_treino_e_vem_com_aviso(self) -> None:
        out = train.shift_decomposition(
            self._cells(test=0.20, future_only=0.30, space_only=0.26, train=0.40))
        self.assertAlmostEqual(0.30 + 0.26 - 0.40 - 0.20, out["interaction"])
        self.assertIn("dentro da amostra", out["interaction_note"])
        self.assertTrue(out["average_precision"]["train_cell_in_sample"])

    def test_sem_cenarios_a_decomposicao_nao_inventa_numeros(self) -> None:
        out = train.shift_decomposition(self._cells(test=0.20))
        self.assertIsNone(out["cost_of_new_places"])
        self.assertIsNone(out["cost_of_new_period"])
        self.assertIsNone(out["interaction"])


def _fake_ref(key: str, year: int) -> core.EventRef:
    """Referência de evento sem arquivo: basta para checar qual célula vai para qual dataset."""
    return core.EventRef(year=year, fire_name=key, path=Path(f"/fake/{year}/{key}.hdf5"), n_days=10,
                         height=64, width=64,
                         dates=tuple(f"{year}-07-{d + 1:02d}" for d in range(10)), lnglat=(-120.0, 38.0))


class ProtocolTests(unittest.TestCase):
    """O que estes testes guardam: nenhum evento de célula de avaliação entra em treino ou validação.

    O treino **enumera** os metadados de todos os eventos — é o que permite calcular os blocos espaciais e as
    datas —, então a garantia não pode ser "não listar o ano de teste". A garantia verificável é mais forte e
    está aqui: os datasets que o treino constrói recebem apenas as células de treino e de validação, nos dois
    protocolos, e a normalização é conferida contra as chaves da célula de treino.
    """

    def _run_until_datasets(self, root: Path, spec: core.SplitSpec, extra: list[str]) -> list[list[str]]:
        received: list[list[str]] = []

        def fake_dataset(events, *_args, **_kwargs):
            received.append([e.key for e in events])
            raise RuntimeError("parar após construir o dataset")

        manifest = root / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        with mock.patch.object(train, "resolve_split", return_value=spec), \
             mock.patch.object(train, "WSTSWindowDataset", side_effect=fake_dataset), \
             mock.patch.object(train, "check_resources"), mock.patch.object(train, "HDF5_MANIFEST", manifest):
            with self.assertRaises(RuntimeError):
                train.main(["--data-root", str(root), "--cpu", "--report", str(root / "r.json"),
                            "--ledger", str(root / "ledger.jsonl"), *extra])
        return received

    def _write_normalization(self, root: Path, name: str, spec: core.SplitSpec) -> None:
        (root / "stats").mkdir(exist_ok=True)
        payload = {**NORMALIZATION, "split_kind": spec.kind, "fold": spec.fold,
                   "fit_years": sorted({e.year for e in spec.events["train"]}),
                   "train_event_keys": sorted(e.key for e in spec.events["train"])}
        (root / "stats" / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_temporal_training_datasets_never_receive_test_year_events(self) -> None:
        spec = core.SplitSpec(
            kind="temporal", fold=2,
            events={"train": [_fake_ref("a", 2018), _fake_ref("b", 2020)],
                    "validation": [_fake_ref("c", 2019)],
                    "test": [_fake_ref("d", 2021)]},
            metadata={"years": {"train": [2018, 2020], "validation": [2019], "test": [2021]}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_normalization(root, "fold2_train_normalization.json", spec)
            received = self._run_until_datasets(root, spec, [])
        self.assertEqual([["2018/a", "2020/b"]], received)
        self.assertNotIn("2021/d", [key for keys in received for key in keys])

    def test_spatiotemporal_training_datasets_only_receive_train_and_validation_cells(self) -> None:
        spec = core.SplitSpec(
            kind="spatiotemporal", fold=core.SPATIOTEMPORAL_FOLD,
            events={"train": [_fake_ref("a", 2019), _fake_ref("b", 2020)],
                    "validation": [_fake_ref("c", 2018)],
                    "test": [_fake_ref("d", 2021)],
                    "future_only": [_fake_ref("e", 2021)],
                    "space_only": [_fake_ref("f", 2020)]},
            metadata={"years": {"train": [2019, 2020], "validation": [2018], "test": [2021]}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_normalization(
                root, f"spatiotemporal_fold{core.SPATIOTEMPORAL_FOLD}_train_normalization.json", spec)
            definition = root / "definition.json"
            definition.write_text(json.dumps({"assignment": {}}), encoding="utf-8")
            received = self._run_until_datasets(
                root, spec, ["--split-kind", "spatiotemporal", "--split-definition", str(definition)])
        flat = [key for keys in received for key in keys]
        self.assertEqual([["2019/a", "2020/b"]], received)
        for key in ("2021/d", "2021/e", "2020/f"):
            self.assertNotIn(key, flat)

    def test_spatiotemporal_requires_the_versioned_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stats").mkdir()
            name = f"spatiotemporal_fold{core.SPATIOTEMPORAL_FOLD}_train_normalization.json"
            (root / "stats" / name).write_text(json.dumps(NORMALIZATION), encoding="utf-8")
            # Sem o manifesto o treino para antes, por outro motivo — e o teste passaria só onde ele existe.
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with mock.patch.object(train, "check_resources"), mock.patch.object(train, "HDF5_MANIFEST", manifest):
                with self.assertRaises(SystemExit) as raised:
                    train.main(["--data-root", str(root), "--cpu", "--split-kind", "spatiotemporal",
                                "--split-definition", str(root / "ausente.json"),
                                "--report", str(root / "r.json"), "--ledger", str(root / "l.jsonl")])
        self.assertIn("make_splits.py", str(raised.exception))

    def test_normalization_of_the_other_protocol_is_rejected(self) -> None:
        spec = core.SplitSpec(
            kind="spatiotemporal", fold=core.SPATIOTEMPORAL_FOLD,
            events={"train": [_fake_ref("a", 2019)], "validation": [_fake_ref("c", 2018)],
                    "test": [_fake_ref("d", 2021)], "future_only": [], "space_only": []},
            metadata={"years": {"train": [2019], "validation": [2018], "test": [2021]}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stats").mkdir()
            # Normalização do fold temporal: anos coerentes, conjunto de eventos errado.
            name = f"spatiotemporal_fold{core.SPATIOTEMPORAL_FOLD}_train_normalization.json"
            (root / "stats" / name).write_text(
                json.dumps({**NORMALIZATION, "fold": 2, "fit_years": [2018, 2020],
                            "train_event_keys": ["2018/outro"]}), encoding="utf-8")
            definition = root / "definition.json"
            definition.write_text(json.dumps({"assignment": {}}), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with mock.patch.object(train, "resolve_split", return_value=spec), \
                 mock.patch.object(train, "check_resources"), \
                 mock.patch.object(train, "HDF5_MANIFEST", manifest):
                with self.assertRaises(ValueError) as raised:
                    train.main(["--data-root", str(root), "--cpu", "--split-kind", "spatiotemporal",
                                "--split-definition", str(definition),
                                "--report", str(root / "r.json"), "--ledger", str(root / "l.jsonl")])
        self.assertIn("Faltando", str(raised.exception))

    def test_smoke_cannot_evaluate_test_and_halo_must_cover_receptive_field(self) -> None:
        with self.assertRaises(SystemExit):
            train.main(["--evaluate-test", "--run-name", "x", "--max-windows", "3"])
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(train, "check_resources"):
                with self.assertRaises(SystemExit):
                    train.main(["--data-root", tmp, "--cpu", "--halo", "2", "--num-layers", "3",
                                "--report", str(Path(tmp) / "r.json"), "--ledger", str(Path(tmp) / "l.jsonl")])

    def _tested_run(self, root: Path, *, in_report: bool, in_ledger: bool) -> tuple[Path, Path]:
        checkpoint = root / "runs" / "r1" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"pesos")
        report = root / "r1.json"
        payload = {"kind": "full", "eligible_for_test": True, "data_root": str(root), "checkpoint": "runs/r1/best.pt",
                   "checkpoint_sha256": "0" * 64}
        if in_report:
            payload["test"] = {"evaluated_at_utc": "2026-09-30T00:00:00+00:00"}
        report.write_text(json.dumps(payload), encoding="utf-8")
        ledger = root / "ledger.jsonl"
        if in_ledger:
            ledger.write_text(json.dumps({"run_name": "r1", "test_ap": 0.1}) + "\n", encoding="utf-8")
        return report, ledger

    def test_ledger_blocks_reevaluation_even_without_test_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report, ledger = self._tested_run(Path(tmp), in_report=False, in_ledger=True)
            with self.assertRaises(SystemExit) as raised:
                train.main(["--evaluate-test", "--run-name", "r1", "--report", str(report), "--ledger", str(ledger)])
            self.assertIn("já foi avaliado", str(raised.exception))

    def test_checkpoint_hash_is_verified_before_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report, ledger = self._tested_run(Path(tmp), in_report=False, in_ledger=False)
            with self.assertRaises(SystemExit) as raised:
                train.main(["--evaluate-test", "--run-name", "r1", "--report", str(report), "--ledger", str(ledger)])
            self.assertIn("SHA-256", str(raised.exception))

    def test_tested_run_is_never_retrained_even_with_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, ledger = self._tested_run(root, in_report=False, in_ledger=True)
            with self.assertRaises(SystemExit) as raised:
                train.main(["--data-root", tmp, "--run-name", "r1", "--force", "--cpu", "--max-windows", "2",
                            "--report", str(root / "smoke.json"), "--ledger", str(ledger)])
            self.assertIn("já teve o teste avaliado", str(raised.exception))

    def test_timing_runs_are_not_eligible_for_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "timing.json"
            report.write_text(json.dumps({"kind": "timing", "eligible_for_test": False}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                train.main(["--evaluate-test", "--run-name", "t", "--report", str(report),
                            "--ledger", str(Path(tmp) / "l.jsonl")])

    def test_bootstrap_ap_with_several_events(self) -> None:
        accumulator = train.EvaluationAccumulator()
        rng = np.random.default_rng(0)
        for event in range(5):
            labels = (rng.random((1, 1, 1, 8, 8)) < 0.2).astype(np.float32)
            logits = rng.normal(labels * 2.0, 1.0).astype(np.float32)
            zeros = np.zeros_like(labels)
            batch = {"y": labels, "fire_t": zeros, "fire_last_t": zeros, "fire_window": zeros,
                     "crop32": np.ones_like(labels), "mask": np.ones_like(labels), "event": np.array([event])}
            accumulator.update(logits, batch)
        interval = accumulator.bootstrap_ap("all", n_boot=200)
        point = accumulator.model["all"].summary()["average_precision"]
        self.assertEqual(interval["n_events"], 5)
        self.assertLessEqual(interval["ci95"][0], point + 0.05)
        self.assertGreaterEqual(interval["ci95"][1], point - 0.05)
        for view in train.VIEWS:
            self.assertIsNotNone(accumulator.bootstrap_ap(view, n_boot=20))
        best_bin = accumulator.model["all"].summary()["best_threshold"]["bin"]
        paired = accumulator.bootstrap_vs_persistence("all", best_bin, n_boot=100)
        self.assertGreater(paired["point"], 0.0)  # persistência vazia (fire_t = 0) tem F2 = 0
        self.assertTrue(paired["model_beats_persistence"])
        with tempfile.TemporaryDirectory() as tmp:
            with np.load(accumulator.save_per_event(Path(tmp) / "counts.npz")) as saved:
                self.assertEqual(saved["all_logit_histograms"].shape, (5, 2, train.EVENT_BINS))
                self.assertEqual(saved["crop32_baseline_confusion"].shape, (5, len(train.BASELINES), 4))

    def test_existing_report_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "r.json"
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                train.main(["--data-root", tmp, "--cpu", "--report", str(report), "--ledger", str(Path(tmp) / "l.jsonl")])


if __name__ == "__main__":
    unittest.main()
