from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import core as core
from scripts import pde_diagnostics as pde


class SpreadGeometryTests(unittest.TestCase):
    def test_eastward_growth_points_to_increasing_columns(self) -> None:
        fire_t = np.zeros((20, 20), dtype=np.uint8)
        fire_t[8:12, 5:8] = 1
        fire_next = fire_t.copy()
        fire_next[8:12, 8:11] = 1
        spread = pde.spread_vectors(fire_t, fire_next)
        d_col, d_row = spread["direction"]
        self.assertGreater(d_col, 0.9)
        self.assertAlmostEqual(d_row, 0.0, places=6)
        self.assertEqual(spread["n_local"], 12)
        self.assertEqual(float(spread["distances_px"].max()), 3.0)

    def test_no_fire_or_no_new_pixels_returns_none(self) -> None:
        empty = np.zeros((5, 5), dtype=np.uint8)
        burning = empty.copy()
        burning[2, 2] = 1
        self.assertIsNone(pde.spread_vectors(empty, burning))
        self.assertIsNone(pde.spread_vectors(burning, burning))

    def test_precomputed_edt_matches_internal_one(self) -> None:
        fire_t = np.zeros((16, 16), dtype=np.uint8)
        fire_t[6:9, 6:9] = 1
        fire_next = fire_t.copy()
        fire_next[6:9, 9:11] = 1
        direct = pde.spread_vectors(fire_t, fire_next)
        shared = pde.spread_vectors(fire_t, fire_next, precomputed=pde.distance_to_fire(fire_t))
        self.assertEqual(direct["direction"], shared["direction"])
        np.testing.assert_array_equal(direct["distances_px"], shared["distances_px"])

    def test_front_speed_is_not_conditioned_on_movement(self) -> None:
        fire_t = np.zeros((10, 10), dtype=np.uint8)
        fire_t[4:6, 4:6] = 1
        stalled = pde.front_statistics(fire_t, fire_t)
        self.assertEqual(stalled["speed_px_per_day"], 0.0)
        self.assertEqual(stalled["share_front_advancing"], 0.0)
        grown = fire_t.copy()
        grown[4:6, 6] = 1
        moving = pde.front_statistics(fire_t, grown)
        self.assertAlmostEqual(moving["speed_px_per_day"], 2 / 4)
        self.assertAlmostEqual(moving["share_front_advancing"], 0.5)


class NullCalibrationTests(unittest.TestCase):
    """O nulo precisa randomizar na unidade certa, senão os p-valores ficam anticonservadores."""

    @staticmethod
    def _h0_sample(rng: np.random.Generator, n_events: int = 12, n_days: int = 10):
        """Sob H0: a direção do driver é do **evento**, e o espalhamento é independente dela."""
        groups = np.repeat(np.arange(n_events), n_days)
        angles = np.repeat(rng.uniform(0, 2 * np.pi, n_events), n_days)
        driver = np.column_stack([np.cos(angles), np.sin(angles)])
        free = rng.uniform(0, 2 * np.pi, groups.size)
        spread = np.column_stack([np.cos(free), np.sin(free)])
        return spread, driver, groups

    def test_rotation_null_is_calibrated_at_event_level(self) -> None:
        rng = np.random.default_rng(7)
        trials = 120
        rejections = 0
        for _ in range(trials):
            spread, driver, groups = self._h0_sample(rng)
            rejections += pde.rotation_pvalue(spread, driver, groups, 60, rng)["p_value_two_sided"] < 0.05
        rate = rejections / trials
        # A versão anterior (um ângulo por dia) rejeitava ~25% a 5% nominal.
        self.assertLess(rate, 0.15, f"erro tipo I inflado: {rate:.3f}")

    def test_rotation_null_still_detects_real_alignment(self) -> None:
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(10), 8)
        angles = rng.uniform(0, 2 * np.pi, groups.size)
        driver = np.column_stack([np.cos(angles), np.sin(angles)])
        spread = driver + rng.normal(0, 0.15, driver.shape)
        result = pde.rotation_pvalue(spread, driver, groups, 300, rng)
        self.assertGreater(result["mean_cosine_weighted"], 0.8)
        self.assertLess(result["p_value_two_sided"], 0.05)
        self.assertEqual(result["n_events"], 10)

    def test_rotation_null_is_two_sided(self) -> None:
        rng = np.random.default_rng(11)
        groups = np.repeat(np.arange(10), 8)
        angles = rng.uniform(0, 2 * np.pi, groups.size)
        driver = np.column_stack([np.cos(angles), np.sin(angles)])
        anti = -driver + rng.normal(0, 0.15, driver.shape)  # convenção invertida
        result = pde.rotation_pvalue(anti, driver, groups, 300, rng)
        self.assertLess(result["mean_cosine_weighted"], -0.8)
        self.assertLess(result["p_value_two_sided"], 0.05)
        self.assertGreater(result["p_value_one_sided"], 0.9)

    def test_circular_shift_preserves_each_event_multiset(self) -> None:
        rng = np.random.default_rng(5)
        values = np.arange(12.0).reshape(-1, 1)
        index = np.arange(4, 8)
        shifted = pde.circular_shift(values, index, rng)
        self.assertEqual(sorted(shifted.ravel().tolist()), sorted(values[index].ravel().tolist()))

    def test_shift_null_detects_daily_variation(self) -> None:
        rng = np.random.default_rng(9)
        groups = np.repeat(np.arange(8), 10)
        angles = rng.uniform(0, 2 * np.pi, groups.size)
        driver = np.column_stack([np.cos(angles), np.sin(angles)])
        spread = driver + rng.normal(0, 0.15, driver.shape)
        self.assertLess(pde.shift_pvalue(spread, driver, groups, 300, rng)["p_value"], 0.05)


class StatisticsTests(unittest.TestCase):
    def test_zero_norm_rows_are_dropped_per_test(self) -> None:
        rng = np.random.default_rng(1)
        rows = []
        for i in range(18):
            rows.append({
                "event": i % 3,
                "spread": tuple(rng.normal(size=2)),
                "wind_t": tuple(rng.normal(size=2)),
                "wind_t1": tuple(rng.normal(size=2)),
                # Metade dos evento-dias em terreno plano: só `upslope` deve perdê-los.
                "upslope": (0.0, 0.0) if i % 2 else tuple(rng.normal(size=2)),
                "n_local_new_pixels": 5,
            })
        result = pde.analyze_advection(rows, 100, 30, rng)
        self.assertEqual(result["tests"]["wind_t"]["dropped_zero_norm"], 0)
        self.assertEqual(result["tests"]["wind_t"]["n_event_days"], 18)
        self.assertEqual(result["tests"]["upslope"]["dropped_zero_norm"], 9)
        self.assertEqual(result["tests"]["upslope"]["n_event_days"], 9)
        self.assertTrue(np.isfinite(result["tests"]["wind_t"]["rotation_null"]["mean_cosine_weighted"]))
        self.assertEqual(result["partial_fit_wind_vs_upslope"]["n_event_days"], 9)

    def test_partial_fit_separates_wind_from_slope(self) -> None:
        rng = np.random.default_rng(1)
        wind = rng.normal(size=(200, 2))
        slope = rng.normal(size=(200, 2))
        np.testing.assert_allclose(pde.partial_fit(3.0 * wind, wind, slope), [1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(pde.partial_fit(0.5 * slope, wind, slope), [0.0, 1.0], atol=1e-9)

    def test_within_event_estimator_ignores_between_event_confounding(self) -> None:
        rng = np.random.default_rng(2)
        groups = np.repeat(np.arange(8), 20)
        event_level = np.repeat(rng.normal(0, 10, 8), 20)
        x = event_level + rng.normal(0, 1, groups.size)
        y = event_level + rng.normal(0, 1, groups.size)
        self.assertGreater(pde.spearman(x, y), 0.8)
        self.assertLess(abs(pde.within_event_partial(x, y, groups)), 0.25)
        w = rng.normal(0, 1, groups.size)
        confounded = w + rng.normal(0, 1.0, groups.size)
        outcome = w + rng.normal(0, 1.0, groups.size)
        self.assertGreater(pde.within_event_partial(confounded, outcome, groups), 0.3)
        self.assertLess(abs(pde.within_event_partial(confounded, outcome, groups, [w])), 0.25)

    def test_control_constant_within_event_is_flagged_as_inert(self) -> None:
        groups = np.repeat(np.arange(6), 5)
        constant = np.repeat(np.arange(6, dtype=float), 5)  # varia entre eventos, não dentro
        varying = np.arange(float(groups.size))
        inert = pde.control_diagnostics(constant, groups)
        self.assertFalse(inert["effective"])
        self.assertEqual(inert["share_events_with_variation"], 0.0)
        self.assertTrue(pde.control_diagnostics(varying, groups)["effective"])

    def test_bootstrap_labels_replicas_separately(self) -> None:
        groups = np.repeat(np.arange(5), 4)
        seen: list[int] = []

        def statistic(index: np.ndarray, sample_groups: np.ndarray) -> float:
            seen.append(len(np.unique(sample_groups)))
            return 1.0

        result = pde.bootstrap_by_group(statistic, groups, 30, np.random.default_rng(0))
        # Cada sorteio tem 5 réplicas rotuladas, mesmo quando um evento sai repetido.
        self.assertEqual(set(seen), {5})
        self.assertEqual(result["n_valid"], 30)
        self.assertEqual(result["ci95"], [1.0, 1.0])
        self.assertIsNone(pde.bootstrap_by_group(statistic, np.array([0, 0, 1, 1]), 5, np.random.default_rng(0)))

    def test_bootstrap_reports_failures(self) -> None:
        groups = np.repeat(np.arange(4), 3)
        result = pde.bootstrap_by_group(lambda i, g: None, groups, 7, np.random.default_rng(0))
        self.assertEqual((result["n_valid"], result["failures"]), (0, 7))

    def test_holm_is_monotone_and_bounded(self) -> None:
        adjusted = pde.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertLessEqual(max(adjusted.values()), 1.0)

    def test_spearman_handles_constant_input(self) -> None:
        self.assertIsNone(pde.spearman(np.ones(10), np.arange(10.0)))
        self.assertAlmostEqual(pde.spearman(np.arange(10.0), np.arange(10.0) ** 2), 1.0)

    def test_cosine_and_mean_vector(self) -> None:
        self.assertAlmostEqual(pde.cosine((1.0, 0.0), (0.0, 2.0)), 0.0)
        self.assertIsNone(pde.cosine((0.0, 0.0), (1.0, 0.0)))
        field = np.array([[1.0, 3.0], [np.nan, 5.0]])
        where = np.array([[True, True], [True, False]])
        self.assertEqual(pde.mean_vector(field, field, where), (2.0, 2.0))
        self.assertIsNone(pde.mean_vector(field, field, np.zeros_like(where)))


class FreedmanLaneTests(unittest.TestCase):
    """Permutar `x` bruto quebra a relação x-controles; o nulo deixa de ser trocável."""

    @staticmethod
    def _h0_sample(rng, n_events=10, n_days=12):
        """Sob H0: x e y dependem do mesmo controle, mas são independentes dado ele."""
        groups = np.repeat(np.arange(n_events), n_days)
        control = rng.normal(0, 1, groups.size)
        x = control + rng.normal(0, 0.5, groups.size)
        y = control + rng.normal(0, 0.5, groups.size)
        return x, y, groups, [control]

    def test_freedman_lane_is_calibrated_when_x_tracks_the_control(self) -> None:
        rng = np.random.default_rng(17)
        trials, rejections = 100, 0
        for _ in range(trials):
            x, y, groups, controls = self._h0_sample(rng)
            result = pde.freedman_lane_pvalue(x, y, groups, controls, 60, rng)
            rejections += result is not None and result["p_value"] < 0.05
        self.assertLess(rejections / trials, 0.15, f"erro tipo I inflado: {rejections / trials:.3f}")

    def test_freedman_lane_still_detects_a_real_partial_effect(self) -> None:
        rng = np.random.default_rng(19)
        groups = np.repeat(np.arange(10), 12)
        control = rng.normal(0, 1, groups.size)
        x = control + rng.normal(0, 0.5, groups.size)
        y = control + 1.5 * x + rng.normal(0, 0.3, groups.size)
        result = pde.freedman_lane_pvalue(x, y, groups, [control], 300, rng)
        self.assertGreater(result["statistic"], 0.5)
        self.assertLess(result["p_value"], 0.05)
        self.assertIn("Freedman-Lane", result["null"])

    def test_rank_design_and_partial_agree(self) -> None:
        rng = np.random.default_rng(21)
        groups = np.repeat(np.arange(6), 10)
        x, y, c = (rng.normal(size=groups.size) for _ in range(3))
        a, b, design, g = pde.rank_design(x, y, groups, [c])
        self.assertEqual(len(g), len(a))
        self.assertEqual(design.shape, (len(a), 2))
        direct = pde.correlate(pde.residualize(a, design), pde.residualize(b, design))
        self.assertAlmostEqual(direct, pde.within_event_partial(x, y, groups, [c]))

    def test_global_rank_keeps_bias_small_at_real_event_size(self) -> None:
        """Ranquear dentro do evento enviesava o estimador em +0,074 com 19 dias por evento."""
        rng = np.random.default_rng(5)
        values = []
        for _ in range(200):
            groups = np.repeat(np.arange(10), 19)
            control = rng.normal(0, 1, groups.size)
            x = control + rng.normal(0, 0.5, groups.size)
            y = control + rng.normal(0, 0.5, groups.size)
            values.append(pde.within_event_partial(x, y, groups, [control]))
        self.assertLess(abs(np.mean(values)), 0.05, f"viés do estimador: {np.mean(values):+.4f}")


class StructuringElementTests(unittest.TestCase):
    def test_disk_is_round_not_diamond(self) -> None:
        element = pde.disk(10)
        self.assertEqual(element.shape, (21, 21))
        # O canto a (7,7) do centro dista ~9,9: cabe no disco de raio 10, mas não no diamante (|7|+|7|=14).
        self.assertTrue(element[10 + 7, 10 + 7])
        self.assertFalse(element[0, 0])
        self.assertAlmostEqual(element.sum() / (np.pi * 100), 1.0, delta=0.05)


class CompositeTrackerTests(unittest.TestCase):
    def _field(self, value: float) -> np.ndarray:
        return np.full((4, 4), value, dtype=np.float32)

    def test_switch_requires_majority_of_pixels_to_change(self) -> None:
        tracker = pde.CompositeTracker()
        tracker.update(0, self._field(100.0))
        noisy = self._field(100.0)
        noisy[0, 0] = 999.0  # 1 de 16 pixels: ruído, não troca
        tracker.update(1, noisy)
        self.assertEqual(tracker.switches, 0)
        tracker.update(2, self._field(200.0))
        self.assertEqual(tracker.switches, 1)

    def test_pre_never_returns_a_composite_whose_window_reaches_t(self) -> None:
        tracker = pde.CompositeTracker()
        tracker.update(0, self._field(100.0))
        tracker.update(20, self._field(200.0))
        # O composto iniciado no dia 0 cobre [0, 15]: a janela ainda alcança o dia 15, então só vale
        # como "pré" a partir do dia 16.
        self.assertIsNone(tracker.pre(pde.COMPOSITE_SPAN_DAYS))
        self.assertEqual(tracker.pre(pde.COMPOSITE_SPAN_DAYS + 1)[0, 0], 100.0)
        self.assertEqual(tracker.pre(40)[0, 0], 200.0)

    def test_history_is_bounded(self) -> None:
        tracker = pde.CompositeTracker()
        for day in range(0, 200, 16):
            tracker.update(day, self._field(float(day)))
        self.assertLessEqual(len(tracker.history), pde.CompositeTracker.MAX_HISTORY)


def _event(root: Path, name: str, *, n_days: int, uphill_east: bool, nan_viirs: bool = False) -> None:
    frames = np.full((n_days, core.N_RAW_CHANNELS, 30, 30), 10.0, dtype=np.float32)
    frames[:, core.LANDCOVER_CHANNEL] = 9.0
    frames[:, core.ACTIVE_FIRE_CHANNEL] = 0.0
    frames[:, pde.WIND_SPEED_CHANNEL] = 5.0
    frames[:, pde.WIND_DIRECTION_CHANNEL] = 270.0  # vento de oeste: transporte para leste
    ramp = 100.0 * np.arange(30)
    frames[:, pde.ELEVATION_CHANNEL] = ramp[None, :] if uphill_east else ramp[:, None]
    if nan_viirs:
        frames[:, pde.VIIRS_CHANNELS] = np.nan
    for day in range(n_days):
        frames[day, core.ACTIVE_FIRE_CHANNEL, 12:18, 3 : 6 + 2 * day] = 20.0  # cresce para leste
    path = root / "hdf5" / "2018" / f"{name}.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with core.open_h5(path, "w") as handle:
        data = handle.create_dataset("data", data=frames)
        data.attrs["year"], data.attrs["fire_name"] = 2018, name
        data.attrs["img_dates"] = [f"2018-07-{d + 1:02d}" for d in range(n_days)]
        data.attrs["lnglat"] = [-123.0, 40.0]


class IntegrationTests(unittest.TestCase):
    def test_westerly_wind_and_eastward_growth_align_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for event in range(6):
                _event(root, f"fire_{event}", n_days=6, uphill_east=event % 2 == 0)
            # Manifesto inexistente: o relatório tem de se declarar não publicável por esse motivo.
            with mock.patch.object(pde, "HDF5_MANIFEST", root / "sem_manifesto.json"):
                report = pde.main(["--data-root", tmp, "--permutations", "200", "--bootstrap", "50",
                                   "--report", str(root / "pde.json")])
            self.assertTrue(Path(report["collected_file"]).exists())
            reanalysed = pde.main(["--data-root", tmp, "--from-collected", "--permutations", "200",
                                   "--bootstrap", "50", "--report", str(root / "pde2.json")])
        wind = report["advection"]["tests"]["wind_t"]["rotation_null"]
        self.assertGreater(wind["mean_cosine_weighted"], 0.9)
        self.assertEqual(wind["unit_of_randomization"], "evento")
        self.assertEqual(report["events"][0], "2018/fire_0")
        self.assertIn("repo", report["provenance"])
        self.assertAlmostEqual(report["scales"]["share_event_days_without_advance"], 0.0)
        self.assertIsNotNone(report["scales"]["front_pixels"])
        # Sem o manifesto de conversão, o relatório se declara não publicável.
        self.assertFalse(report["publishable"])
        self.assertTrue(any("manifesto" in p for p in report["publishability_problems"]))
        # Reanalisar a coleta salva reproduz os mesmos números, sem reler o HDF5.
        self.assertEqual(
            reanalysed["advection"]["tests"]["wind_t"]["rotation_null"]["mean_cosine_weighted"],
            wind["mean_cosine_weighted"],
        )
        self.assertEqual(reanalysed["scales"], report["scales"])

    def test_observability_is_one_when_viirs_is_all_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _event(root, "fire_nan", n_days=5, uphill_east=True, nan_viirs=True)
            collected = pde.collect(core.list_events(root / "hdf5", [2018]))
        shares = [r["viirs_nan_share_t1_near_fire"] for r in collected["growth"]]
        self.assertTrue(shares)
        self.assertTrue(all(s == 1.0 for s in shares))

    def test_empty_dataset_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "hdf5" / "2018").mkdir(parents=True)
            with self.assertRaises(SystemExit):
                pde.main(["--data-root", tmp, "--report", str(Path(tmp) / "pde.json")])


if __name__ == "__main__":
    unittest.main()
