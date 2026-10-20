from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts import convert_hdf5 as convert
from scripts import core as core


def _write_tif(path: Path, image: np.ndarray) -> None:
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=image.shape[1],
        width=image.shape[2],
        count=image.shape[0],
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(500000, 4200000, 375, 375),
        compress="deflate",
    ) as destination:
        destination.write(image)


def _image(rng: np.random.Generator) -> np.ndarray:
    image = rng.normal(0, 1, size=(core.N_RAW_CHANNELS, 3, 4)).astype(np.float32)
    image[0, 0, 0] = np.nan
    af = np.full((3, 4), np.nan, dtype=np.float32)
    af[1, 1], af[2, 3] = 2135.0, 30.0
    image[core.ACTIVE_FIRE_CHANNEL] = af
    return image


def _build_zip(root: Path, events: dict[str, list[str]], seed: int = 0) -> tuple[Path, dict[str, list[np.ndarray]]]:
    rng = np.random.default_rng(seed)
    archive = root / "wsts.zip"
    images: dict[str, list[np.ndarray]] = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for key, dates in events.items():
            images[key] = []
            for date in dates:
                image = _image(rng)
                images[key].append(image)
                tif = root / "tmp.tif"
                _write_tif(tif, image)
                handle.write(tif, f"{key}/{date}.tif")
            handle.writestr(f"{key}/", "")
        handle.writestr("__MACOSX/2018/._x.tif", "")
    return archive, images


class ConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_zip_to_hdf5_matches_official_preprocessing(self) -> None:
        dates = ["2018-07-01", "2018-07-02", "2018-07-04"]
        archive, images = _build_zip(self.root, {"2018/fire_7": dates})
        with zipfile.ZipFile(archive) as handle:
            events = convert.list_zip_events(handle.infolist())
        self.assertEqual(list(events), [(2018, "fire_7")])
        target = self.root / "hdf5" / "2018" / "fire_7.hdf5"
        record = convert.convert_event(archive.as_posix(), 2018, "fire_7", events[(2018, "fire_7")], target)
        self.assertFalse(target.with_suffix(".hdf5.tmp").exists())
        self.assertEqual(record["lost_detections_hhmm_lt_100"], 3)
        self.assertEqual(record["fire_pixels"], 3)
        self.assertEqual(len(record["date_gaps"]), 1)
        ref = core.read_event_ref(target)
        self.assertEqual(ref.dates, tuple(dates))
        stored = core.read_days(target, 0, 3)
        for day, image in enumerate(images["2018/fire_7"]):
            expected = image.copy()
            expected[-1] = np.floor_divide(np.nan_to_num(expected[-1], nan=0), 100)
            np.testing.assert_array_equal(np.isnan(stored[day]), np.isnan(expected))
            np.testing.assert_array_equal(np.nan_to_num(stored[day]), np.nan_to_num(expected))

    def test_failed_conversion_leaves_no_partial_file(self) -> None:
        archive, _ = _build_zip(self.root, {"2018/fire_1": ["2018-07-01"]})
        members = [convert.Member("2018/fire_1/2018-07-01.tif", 0, 0), convert.Member("2018/fire_1/2018-07-09.tif", 0, 0)]
        target = self.root / "hdf5" / "2018" / "fire_1.hdf5"
        with self.assertRaises(Exception):
            convert.convert_event(archive.as_posix(), 2018, "fire_1", members, target)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_suffix(".hdf5.tmp").exists())

    def test_main_end_to_end_resume_and_force(self) -> None:
        archive, _ = _build_zip(
            self.root,
            {"2018/fire_1": ["2018-07-01", "2018-07-02"], "2018/fire_2": ["2018-08-01"], "2019/fire_3": ["2019-06-01"]},
        )
        report = self.root / "manifest.json"
        base = ["--data-root", str(self.root), "--zip", str(archive), "--years", "2018", "--report", str(report)]
        first = convert.main(base)
        self.assertEqual((first["n_events"], first["n_images"], first["resumed_events"]), (2, 3, 0))
        self.assertFalse(first["full_run"])
        self.assertIsNone(first["counts_match_zenodo"])
        self.assertEqual(first["per_year"]["2018"]["fire_pixels"], 3)
        sidecar = json.loads((self.root / "hdf5" / "2018" / "fire_1.json").read_text(encoding="utf-8"))
        self.assertIn("sha256", sidecar)
        second = convert.main(base)
        self.assertEqual(second["resumed_events"], 2)
        self.assertEqual(second["lost_detections_hhmm_lt_100"], first["lost_detections_hhmm_lt_100"])
        third = convert.main(base + ["--force"])
        self.assertEqual(third["resumed_events"], 0)

    def test_partial_run_requires_explicit_report(self) -> None:
        with self.assertRaises(SystemExit):
            convert.main(["--data-root", str(self.root), "--max-events", "1"])

    def test_date_from_filename_follows_official_rule(self) -> None:
        self.assertEqual(convert.date_from_filename("2019/fire_1/2019-08-02.tif"), "2019-08-02")
        self.assertEqual(convert.date_from_filename("2019/fire_1/2019-08-02_extra.tif"), "2019-08-02")


if __name__ == "__main__":
    unittest.main()
