from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import core as core


def _write_event(path: Path, frames: np.ndarray, dates: list[str], year: int, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with core.open_h5(path, "w") as handle:
        data = handle.create_dataset("data", data=frames)
        data.attrs["year"] = year
        data.attrs["fire_name"] = name
        data.attrs["img_dates"] = dates
        data.attrs["lnglat"] = [-120.0, 38.0]


def _frames(n_days: int, seed: int, height: int = 4, width: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = rng.normal(10.0, 2.0, size=(n_days, core.N_RAW_CHANNELS, height, width)).astype(np.float32)
    frames[:, core.LANDCOVER_CHANNEL] = rng.integers(1, core.N_LANDCOVER_CLASSES + 1, size=(n_days, height, width))
    frames[:, core.ACTIVE_FIRE_CHANNEL] = 0.0
    return frames


class FoldAndWindowTests(unittest.TestCase):
    def test_fold_matches_official_ordering(self) -> None:
        self.assertEqual(core.split_years(0), {"train": [2018, 2019], "validation": [2020], "test": [2021]})
        self.assertEqual(core.split_years(2), {"train": [2018, 2020], "validation": [2019], "test": [2021]})
        for fold in range(len(core.FOLDS)):
            split = core.split_years(fold)
            years = split["train"] + split["validation"] + split["test"]
            self.assertEqual(sorted(years), list(core.YEARS))
        with self.assertRaises(ValueError):
            core.split_years(12)

    def test_windows_require_consecutive_days_inside_event(self) -> None:
        dates = core.parse_dates([f"2018-07-{d:02d}" for d in range(1, 11)])
        self.assertEqual(core.valid_window_starts(dates, t_in=5, t_out=1), [0, 1, 2, 3, 4])
        gapped = dates.copy()
        gapped[6:] += np.timedelta64(1, "D")
        self.assertEqual(core.valid_window_starts(gapped, t_in=5, t_out=1), [0])
        self.assertEqual(core.date_gaps(gapped)[0]["delta_days"], 2)

    def test_skip_initial_aligns_targets_with_longer_history(self) -> None:
        dates = core.parse_dates([f"2018-07-{d:02d}" for d in range(1, 11)])
        short = core.valid_window_starts(dates, t_in=1, t_out=1, skip_initial=4)
        long = core.valid_window_starts(dates, t_in=5, t_out=1)
        self.assertEqual([s + 1 for s in short], [s + 5 for s in long])

    def test_parse_dates_rejects_corrupted_names(self) -> None:
        with self.assertRaises(ValueError):
            core.parse_dates(["2018\\fire_1\\2018-07-01"])
        self.assertEqual(str(core.parse_dates([b"2018-07-01.tif"])[0]), "2018-07-01")


class ActiveFireTests(unittest.TestCase):
    def test_preprocessing_matches_official_and_counts_lost_detections(self) -> None:
        raw = np.array([[np.nan, 45.0], [2130.0, 100.0]], dtype=np.float32)
        hours, lost = core.preprocess_active_fire(raw)
        np.testing.assert_array_equal(hours, [[0.0, 0.0], [21.0, 1.0]])
        self.assertEqual(lost, 1)
        np.testing.assert_array_equal(core.fire_mask(hours), [[0, 0], [1, 1]])


class NormalizationTests(unittest.TestCase):
    def test_statistics_use_only_train_years_of_fold_and_ignore_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_a = _frames(3, seed=1)
            train_a[0, 0, 0, 0] = np.nan
            train_a[1, core.LANDCOVER_CHANNEL, 0, 0] = np.nan
            train_a[2, core.LANDCOVER_CHANNEL, 0, 1] = 3.5
            train_b = _frames(2, seed=7)
            leaked = _frames(3, seed=2) + 1000.0
            _write_event(root / "2018" / "fire_a.hdf5", train_a, ["2018-07-01", "2018-07-02", "2018-07-03"], 2018, "fire_a")
            _write_event(root / "2020" / "fire_c.hdf5", train_b, ["2020-07-01", "2020-07-02"], 2020, "fire_c")
            _write_event(root / "2019" / "fire_v.hdf5", leaked, ["2019-07-01", "2019-07-02", "2019-07-03"], 2019, "fire_v")
            _write_event(root / "2021" / "fire_b.hdf5", leaked, ["2021-07-01", "2021-07-02", "2021-07-03"], 2021, "fire_b")
            stats = core.fit_normalization_for_fold(root, 2)
        self.assertEqual((stats["fold"], stats["fit_years"]), (2, [2018, 2020]))
        self.assertEqual([e["key"] for e in stats["events"]], ["2018/fire_a", "2020/fire_c"])
        channel = stats["channels"]["viirs_m11"]
        pooled = np.concatenate([train_a[:, 0].ravel(), train_b[:, 0].ravel()]).astype(np.float64)
        valid = pooled[np.isfinite(pooled)]
        self.assertAlmostEqual(channel["mean"], valid.mean(), places=5)
        self.assertAlmostEqual(channel["std"], valid.std(), places=4)
        self.assertAlmostEqual(channel["missing_rate"], 1 / pooled.size)
        self.assertEqual(stats["landcover_counts"]["invalid_or_nan"], 2)
        core.check_normalization_fold(stats, 2)
        with self.assertRaises(ValueError):
            core.check_normalization_fold(stats, 0)

    def test_constant_channel_std_is_flagged(self) -> None:
        tracker = core.StreamingChannelStats()
        frames = _frames(2, seed=8)
        frames[:, 12] = 5.0
        tracker.update(frames)
        slope = tracker.to_dict()["channels"]["slope"]
        self.assertTrue(slope["std_replaced_by_one"])
        self.assertEqual(slope["std"], 1.0)

    def test_event_year_attribute_must_match_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_event(root / "2019" / "fire_a.hdf5", _frames(2, seed=3), ["2018-07-01", "2018-07-02"], 2018, "fire_a")
            with self.assertRaises(ValueError):
                core.list_events(root, [2019])


class TransformTests(unittest.TestCase):
    def _normalization(self) -> dict:
        return {
            "channels": {
                core.CHANNEL_NAMES[c]: {"mean": 10.0, "std": 2.0} for c in core.CONTINUOUS_CHANNELS
            }
        }

    def test_output_contract_and_imputation(self) -> None:
        frames = _frames(2, seed=4)
        frames[0, 3, 1, 1] = np.nan
        frames[1, core.ACTIVE_FIRE_CHANNEL, 2, 2] = 21.0
        x = core.transform_frames(frames, self._normalization())
        names = core.input_channel_names()
        self.assertEqual(x.shape, (len(names), 2, 4, 5))
        self.assertFalse(np.isnan(x).any())
        self.assertNotIn("forecast_precipitation", names)
        self.assertEqual(x[names.index("ndvi"), 0, 1, 1], 0.0)
        self.assertEqual(x[names.index("missing_vegetation_index"), 0, 1, 1], 1.0)
        self.assertEqual(x[names.index("missing_vegetation_index"), 0, 0, 0], 0.0)
        self.assertEqual(x[names.index("active_fire_binary"), 1, 2, 2], 1.0)
        onehot = x[[names.index(f"landcover_{c}") for c in core.LANDCOVER_CLASSES]]
        np.testing.assert_array_equal(onehot.sum(axis=0), np.ones((2, 4, 5)))
        with_precip = core.transform_frames(frames, self._normalization(), include_forecast_precip=True)
        self.assertEqual(with_precip.shape[0], len(core.input_channel_names(include_forecast_precip=True)))

    def test_angles_are_continuous_and_axial_channel_is_mod_180(self) -> None:
        frames = _frames(1, seed=5, height=1, width=2)
        frames[0, 7] = [[359.0, 1.0]]
        frames[0, 19] = [[-89.0, 91.0]]
        x = core.transform_frames(frames, self._normalization())
        names = core.input_channel_names()
        sin7, cos7 = x[names.index("wind_direction_sin"), 0, 0], x[names.index("wind_direction_cos"), 0, 0]
        self.assertAlmostEqual(float(cos7[0]), float(cos7[1]), places=5)
        self.assertAlmostEqual(float(sin7[0]), -float(sin7[1]), places=5)
        sin2 = x[names.index("forecast_wind_direction_sin2"), 0, 0]
        cos2 = x[names.index("forecast_wind_direction_cos2"), 0, 0]
        self.assertAlmostEqual(float(sin2[0]), float(sin2[1]), places=5)
        self.assertAlmostEqual(float(cos2[0]), float(cos2[1]), places=5)

    def test_invalid_landcover_raises_instead_of_silent_mapping(self) -> None:
        frames = _frames(1, seed=6)
        frames[0, core.LANDCOVER_CHANNEL, 0, 0] = 0.0
        with self.assertRaises(ValueError):
            core.transform_frames(frames, self._normalization())
        frames[0, core.LANDCOVER_CHANNEL, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            core.transform_frames(frames, self._normalization())

    def test_infinite_values_are_imputed_and_flagged(self) -> None:
        frames = _frames(1, seed=9)
        frames[0, 5, 0, 0] = np.inf
        frames[0, 20, 1, 1] = np.nan
        x = core.transform_frames(frames, self._normalization())
        names = core.input_channel_names()
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[names.index("precipitation"), 0, 0, 0], 0.0)
        self.assertEqual(x[names.index("missing_gridmet"), 0, 0, 0], 1.0)
        self.assertEqual(x[names.index("missing_gfs"), 0, 1, 1], 1.0)


class VectorConventionTests(unittest.TestCase):
    def test_wind_from_north_blows_south(self) -> None:
        u, v = core.transport_vector(np.array(2.0), np.array(0.0))
        self.assertAlmostEqual(float(u), 0.0, places=6)
        self.assertAlmostEqual(float(v), -2.0, places=6)
        u, v = core.transport_vector(np.array(3.0), np.array(270.0))
        self.assertAlmostEqual(float(u), 3.0, places=6)
        self.assertAlmostEqual(float(v), 0.0, places=6)

    def test_upslope_points_against_aspect(self) -> None:
        east, north = core.upslope_vector(np.array(45.0), np.array(90.0))
        self.assertAlmostEqual(float(east), -1.0, places=6)
        self.assertAlmostEqual(float(north), 0.0, places=6)

    def test_grid_rows_grow_southward(self) -> None:
        d_col, d_row = core.to_grid_components(np.array(0.0), np.array(1.0))
        self.assertEqual(float(d_col), 0.0)
        self.assertEqual(float(d_row), -1.0)

    def test_grid_convergence_in_utm_10n(self) -> None:
        central = core.grid_geometry((-123.0, 37.0))
        self.assertAlmostEqual(central["gamma_deg"], 0.0, places=6)
        east = core.grid_geometry((-112.0, 37.0))
        self.assertLess(east["gamma_deg"], -6.0)
        self.assertGreater(east["gamma_deg"], -7.5)
        self.assertLess(east["ground_pixel_m"], core.PIXEL_SIZE_M)
        self.assertGreater(core.grid_geometry((-124.0, 40.0))["gamma_deg"], 0.0)

    def test_true_north_maps_to_gamma_azimuth_in_grid(self) -> None:
        gamma = -6.5
        grid_east, grid_north = core.true_to_grid(np.array(0.0), np.array(1.0), gamma)
        self.assertAlmostEqual(float(np.degrees(np.arctan2(grid_east, grid_north))), gamma, places=6)
        d_col, d_row = core.to_grid_components(np.array(0.0), np.array(1.0), gamma)
        self.assertLess(float(d_col), 0.0)
        self.assertLess(float(d_row), 0.0)

    def test_upslope_from_elevation_points_uphill_in_grid(self) -> None:
        rows, cols = np.mgrid[0:5, 0:5].astype(float)
        east_ramp = 10.0 * cols
        d_col, d_row = core.upslope_grid_from_elevation(east_ramp, ground_pixel_m=10.0)
        np.testing.assert_allclose(d_col, 1.0)
        np.testing.assert_allclose(d_row, 0.0)
        north_ramp = -10.0 * rows
        d_col, d_row = core.upslope_grid_from_elevation(north_ramp, ground_pixel_m=10.0)
        np.testing.assert_allclose(d_row, -1.0)

    def test_fire_mask_accepts_frame_and_dates_detect_duplicates(self) -> None:
        frame = np.zeros((core.N_RAW_CHANNELS, 2, 2), dtype=np.float32)
        frame[core.ACTIVE_FIRE_CHANNEL, 1, 0] = 20.0
        np.testing.assert_array_equal(core.fire_mask(frame), [[0, 0], [1, 0]])
        dates = core.parse_dates(["2018-07-01", "2018-07-01", "2018-06-30"])
        self.assertEqual([gap["delta_days"] for gap in core.date_gaps(dates)], [0, -1])


if __name__ == "__main__":
    unittest.main()
