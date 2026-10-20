from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import build_eda_notebook as builder
from scripts import core as core
from scripts import eda as eda


def _frames(n_days: int, height: int = 12, width: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = rng.normal(10.0, 1.0, size=(n_days, core.N_RAW_CHANNELS, height, width)).astype(np.float32)
    frames[:, core.LANDCOVER_CHANNEL] = 9.0
    frames[:, core.ACTIVE_FIRE_CHANNEL] = 0.0
    frames[:, 7] = 200.0
    frames[:, 13] = 90.0
    frames[:, 19] = 10.0
    return frames


def _write(root: Path, year: int, name: str, frames: np.ndarray, lnglat=(-120.0, 38.0)) -> core.EventRef:
    path = root / str(year) / f"{name}.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with core.open_h5(path, "w") as handle:
        data = handle.create_dataset("data", data=frames)
        data.attrs["year"] = year
        data.attrs["fire_name"] = name
        data.attrs["img_dates"] = [str(np.datetime64(f"{year}-07-01") + np.timedelta64(d, "D")) for d in range(len(frames))]
        data.attrs["lnglat"] = list(lnglat)
    return core.read_event_ref(path)


class ProtocolTests(unittest.TestCase):
    def test_fire_only_in_test_year_never_reaches_target_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for year in (2018, 2019, 2020):
                _write(root, year, f"fire_{year}", _frames(8, seed=year))
            test = _frames(8, seed=2021)
            test[4:, core.ACTIVE_FIRE_CHANNEL, 3:6, 3:6] = 20.0
            _write(root, 2021, "fire_2021", test)
            summary, normalization, _ = eda.run_eda(root, 2)
        self.assertEqual(summary["train_target"]["positives_next"], 0)
        self.assertEqual(normalization["fit_years"], [2018, 2020])
        self.assertNotIn("active_fire", summary["channels_by_year"]["2021"])
        self.assertNotIn("histogram_counts", summary["channels_by_year"]["2021"]["ndvi"])
        self.assertIn("histogram_counts", summary["channels_by_year"]["2019"]["ndvi"])
        self.assertEqual(summary["inventory"]["per_year"]["2021"]["images"], 8)
        self.assertEqual(sum(summary["active_fire_hour_histogram_utc_train"]), 0)


class FrontAccumulatorTests(unittest.TestCase):
    def test_persistence_distances_and_window_union(self) -> None:
        accumulator = eda.FrontAccumulator()
        frame_t, frame_next = _frames(2, seed=1)
        frame_t[core.ACTIVE_FIRE_CHANNEL, 5, 5] = 20.0
        frame_next[core.ACTIVE_FIRE_CHANNEL, 5, 5] = 20.0
        frame_next[core.ACTIVE_FIRE_CHANNEL, 5, 6] = 20.0
        frame_next[core.ACTIVE_FIRE_CHANNEL, 0, 11] = 20.0
        union = np.zeros((12, 12), dtype=bool)
        union[1, 10] = True
        accumulator.update(0, frame_t, frame_next, union, gamma_deg=0.0, ground_pixel_m=375.0)
        result = accumulator.to_dict()
        persistence = result["persistence_baseline_all_pairs"]
        self.assertEqual((persistence["tp"], persistence["fp"], persistence["fn"]), (1, 0, 2))
        to_t = result["new_pixel_distance"]["to_fire_t"]
        self.assertEqual(to_t["(0, 1] px"], 1)
        self.assertEqual(to_t["(5, 10] px"], 1)
        to_union = result["new_pixel_distance"]["to_window_union"]
        self.assertEqual(to_union["(0, 1] px"], 1)
        self.assertEqual(to_union["(1, 2] px"], 1)
        self.assertAlmostEqual(result["front_ring"]["share_of_new_pixels_in_ring"], 0.5)


class FrontDirectionIntegrationTests(unittest.TestCase):
    def test_westerly_wind_with_eastward_growth_gives_positive_cosine(self) -> None:
        accumulator = eda.FrontAccumulator()
        for event in range(3):
            frame_t, frame_next = _frames(2, 20, 20, seed=event)
            for frame in (frame_t, frame_next):
                frame[6] = 5.0
                frame[7] = 270.0  # vento de oeste: transporte para leste
                frame[14] = 100.0
            frame_t[core.ACTIVE_FIRE_CHANNEL, 8:12, 8:10] = 20.0
            frame_next[core.ACTIVE_FIRE_CHANNEL, 8:12, 8:12] = 20.0  # avança para leste da grade
            union = np.zeros((20, 20), dtype=bool)
            accumulator.update(event, frame_t, frame_next, union, gamma_deg=0.0, ground_pixel_m=375.0)
        cosine = accumulator.to_dict()["front_ring"]["cosine_front_direction_ignite_minus_not"]["wind_transport"]
        self.assertEqual(cosine["n_events"], 3)
        self.assertGreater(cosine["mean"], 0.5)


class ConventionCheckTests(unittest.TestCase):
    def test_aspect_offset_tracks_minus_gamma(self) -> None:
        check = eda.AspectGridCheck(min_slope_deg=0.5)
        for i, lng in enumerate((-123.0, -117.0, -111.0, -106.0)):
            event = core.EventRef(2018, f"fire_{i}", Path("x"), 1, 20, 20, ("2018-07-01",), (lng, 40.0))
            gamma = core.grid_geometry(event.lnglat)["gamma_deg"]
            frame = _frames(1, 20, 20)[0]
            frame[14] = 20.0 * np.arange(20)[None, :]  # sobe para leste da grade
            frame[12] = 3.0
            frame[13] = (270.0 - gamma) % 360.0  # encosta abaixo = oeste da grade, em azimute verdadeiro
            check.update(event, frame, gamma)
        result = check.to_dict()["regression_offset_on_minus_gamma"]
        self.assertAlmostEqual(result["slope"], 1.0, places=4)
        self.assertAlmostEqual(result["intercept_deg"], 0.0, places=4)

    def test_wind_axis_alignment_and_perpendicular(self) -> None:
        aligned, perpendicular = eda.WindAxisCheck(), eda.WindAxisCheck()
        frame = _frames(1)[0]
        frame[7], frame[19] = 270.0, 0.0  # vento de oeste sopra para leste: ângulo matemático 0°
        aligned.update(frame)
        frame[19] = 90.0
        perpendicular.update(frame)
        self.assertAlmostEqual(aligned.to_dict()["mean_cos_2delta"], 1.0, places=6)
        self.assertAlmostEqual(perpendicular.to_dict()["mean_cos_2delta"], -1.0, places=6)

    def test_forecast_equal_to_next_day_peaks_at_lag_one(self) -> None:
        check = eda.ForecastHorizonCheck(max_lag=3)
        rng = np.random.default_rng(0)
        for event_id in range(4):
            observed = np.cumsum(rng.normal(0, 1, 30)) + 290.0
            for day in range(29):
                frame = _frames(1, 2, 2)[0]
                frame[8] = frame[9] = observed[day]
                frame[20] = observed[day + 1] - 273.15
                frame[11] = frame[21] = 0.005
                check.update_day(np.datetime64("2018-07-01") + np.timedelta64(day, "D"), frame)
            check.finish_event(event_id)
        profile = check.to_dict()["temperature"]["corr_observed_by_lag"]
        self.assertEqual(int(np.argmax(profile)), 1)
        self.assertAlmostEqual(profile[1], 1.0, places=6)

    def test_composite_did_detects_early_drop(self) -> None:
        check = eda.CompositeLookAheadCheck()
        for _ in range(3):
            frames = _frames(24, 20, 20)
            frames[:, 3] = 5000.0
            frames[8:, 3] = 5100.0  # troca de composto no dia 8 (todos os pixels mudam)
            frames[8:, 3, 2:4, 2:4] = 3000.0  # pixels que vão queimar já caem no composto pós
            frames[20:, core.ACTIVE_FIRE_CHANNEL, 2:4, 2:4] = 20.0
            for day, frame in enumerate(frames):
                check.update_day(day, frame)
            check.finish_event()
        result = check.to_dict()
        did = result["ndvi"]["did"]
        self.assertEqual(did["n_events"], 3)
        self.assertLess(did["mean"], -1000)
        self.assertEqual(did["conclusion"], "inconclusivo (poucos eventos)")
        self.assertEqual(result["composite_switch_interval_days"]["median"], 8.0)
        self.assertEqual(result["ndvi"]["placebo_did"]["n_events"], 0)

    def test_composite_days_follow_dates_not_indices(self) -> None:
        check = eda.CompositeLookAheadCheck()
        frames = _frames(3, 6, 6)
        frames[:, 3] = 5000.0
        frames[2, core.ACTIVE_FIRE_CHANNEL, 1, 1] = 20.0
        for day, frame in zip((0, 1, 40), frames):
            check.update_day(day, frame)
        self.assertEqual(int(check.first_fire[1, 1]), 40)


class InventoryAndFigureTests(unittest.TestCase):
    def test_channel_inventory_counts_nan_and_out_of_range(self) -> None:
        tracker = eda.ChannelInventory([3, 12], histograms=True)
        frame = _frames(1, 2, 2)[0]
        frame[3] = [[np.nan, 2e4], [1.0, 2.0]]
        frame[12] = 5.0
        tracker.update(frame)
        result = tracker.to_dict()
        self.assertAlmostEqual(result["ndvi"]["missing_rate"], 0.25)
        self.assertEqual(result["ndvi"]["out_of_range"], 1)
        self.assertEqual(sum(result["slope"]["histogram_counts"]), 4)

    def test_external_comparison_is_optional(self) -> None:
        self.assertIsNone(eda.compare_with_official({"channels": {}}, None))
        self.assertIsNone(eda.official_statistics([2018, 2020], repo=Path("nao-existe")))

    def test_figures_from_minimal_run_do_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for year in (2018, 2019, 2020):
                frames = _frames(8, seed=year)
                frames[2:, core.ACTIVE_FIRE_CHANNEL, 5, 5:7] = 20.0
                frames[4:, core.ACTIVE_FIRE_CHANNEL, 6, 5:8] = 21.0
                _write(root, year, f"fire_{year}", frames)
            summary, _, _ = eda.run_eda(root, 2)
            summary["official_statistics_comparison"] = None
            result = eda.make_figures(summary, root / "figs")
        self.assertEqual(result["errors"], {})
        self.assertIn("wsts_inventario_por_ano", result["written"])


class NotebookBuilderTests(unittest.TestCase):
    def test_notebook_cells_are_valid_and_compile(self) -> None:
        notebook = builder.build()
        ids = [cell["id"] for cell in notebook.cells]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, cell["id"], "exec")

    def test_notebook_executes_on_minimal_summary(self) -> None:
        import json

        try:
            from nbclient import NotebookClient
        except ImportError:  # pragma: no cover
            self.skipTest("nbclient indisponível")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            for year in (2018, 2019, 2020, 2021):
                frames = _frames(8, seed=year)
                frames[2:, core.ACTIVE_FIRE_CHANNEL, 5, 5:7] = 20.0
                frames[4:, core.ACTIVE_FIRE_CHANNEL, 6, 5:8] = 21.0
                _write(data, year, f"fire_{year}", frames)
            summary, _, _ = eda.run_eda(data, 2)
            summary["provenance"] = {"repo": {"commit": None}}
            summary["official_statistics_comparison"] = None
            summary["figures"] = {}
            reports = root / "reports"
            reports.mkdir(parents=True)
            (reports / "eda_summary.json").write_text(json.dumps(summary, default=str), encoding="utf-8")
            client = NotebookClient(builder.build(), timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}})
            try:
                client.execute()
            except Exception as error:  # noqa: BLE001
                if "kernel" in str(error).lower() and "no such" in str(error).lower():
                    self.skipTest(f"kernel indisponível: {error}")
                raise


if __name__ == "__main__":
    unittest.main()
