"""Converte o WildfireSpreadTS (GeoTIFF dentro do zip do Zenodo) para HDF5, um arquivo por evento.

Reimplementa `get_generator_for_hdf5` + `CreateHDF5Dataset.py` do repositório oficial
(que não roda nesta venv e corrompe nomes/datas no Windows), preservando:

- layout: dataset `data` float32 `[T, 23, H, W]` com attrs `year`, `fire_name`,
  `img_dates` (`YYYY-MM-DD`) e `lnglat` (centro da imagem);
- pré-processamento do canal de fogo ativo: NaN -> 0 e `hhmm // 100`.

Diferenças deliberadas: leitura direta do zip via `/vsizip/`
(sem extrair), escrita dia a dia (RAM ~ 1 dia), chunk `(1, 23, H, W)` e compressão `lzf`
sem perda, verificação do `.tmp` antes do rename atômico, e um sidecar JSON por evento
(`<fire>.json`) com a impressão digital da origem (CRC32/tamanho de cada membro do zip).
O sidecar permite retomar uma execução interrompida sem reconverter e preserva as contagens
de detecções `hhmm < 100`, que não podem ser recalculadas a partir do HDF5.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import check_resources, resource_snapshot, sha256_file, write_json
from scripts.core import (
    ACTIVE_FIRE_CHANNEL,
    DEFAULT_DATA_ROOT,
    N_RAW_CHANNELS,
    YEARS,
    date_gaps,
    open_h5,
    parse_dates,
    preprocess_active_fire,
)


DEFAULT_REPORT = Path("reports/hdf5_manifest.json")
EXPECTED_EVENTS = 607
EXPECTED_IMAGES = 13607
ZENODO_URL = "https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content"
SIDECAR_VERSION = 1


@dataclass(frozen=True)
class Member:
    name: str
    crc: int
    size: int


def list_zip_events(infos: Iterable[zipfile.ZipInfo | str]) -> dict[tuple[int, str], list[Member]]:
    """Agrupa entradas `ano/fire_x/AAAA-MM-DD.tif` por (ano, evento), com membros ordenados por nome."""
    events: dict[tuple[int, str], list[Member]] = defaultdict(list)
    for info in infos:
        name = info if isinstance(info, str) else info.filename
        parts = PurePosixPath(name).parts
        if len(parts) != 3 or not parts[2].endswith(".tif") or not parts[0].isdigit():
            continue
        member = Member(name, 0, 0) if isinstance(info, str) else Member(name, info.CRC, info.file_size)
        events[(int(parts[0]), parts[1])].append(member)
    return {key: sorted(members, key=lambda m: m.name) for key, members in sorted(events.items())}


def source_fingerprint(members: list[Member]) -> str:
    digest = hashlib.sha256()
    for member in members:
        digest.update(f"{member.name}|{member.crc}|{member.size}\n".encode("utf-8"))
    return digest.hexdigest()


def date_from_filename(name: str) -> str:
    """Mesma regra do oficial: `split("_")[0]` do nome do arquivo, sem `.tif`."""
    return PurePosixPath(name).name.split("_")[0].replace(".tif", "")


def gdal_path(archive: str, inner: str) -> str:
    """Caminho GDAL com o arquivo delimitado por chaves (necessário para URLs sem `.zip`)."""
    return f"/vsizip/{{{archive}}}/{inner}"


def sidecar_path(target: Path) -> Path:
    return target.with_suffix(".json")


def convert_event(
    archive: str,
    year: int,
    fire_name: str,
    members: list[Member],
    target: Path,
) -> dict[str, Any]:
    """Converte um evento lendo um TIF por vez; o alvo só aparece depois de verificado."""
    import rasterio

    dates = [date_from_filename(member.name) for member in members]
    parsed = parse_dates(dates)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".hdf5.tmp")
    lost_sub_hour = fire_pixels = 0
    lnglat = None
    height = width = None
    crs = transform = None
    try:
        with open_h5(temporary, "w") as handle:
            dataset = None
            for day, member in enumerate(members):
                with rasterio.open(gdal_path(archive, member.name)) as source:
                    image = source.read().astype(np.float32, copy=False)
                    if lnglat is None:
                        lnglat = [float(v) for v in source.lnglat()]
                        crs, transform = source.crs.to_string(), list(source.transform)[:6]
                    elif source.crs.to_string() != crs or not np.allclose(list(source.transform)[:6], transform):
                        raise ValueError(f"{member.name}: CRS/transform diverge do primeiro dia do evento.")
                if image.shape[0] != N_RAW_CHANNELS:
                    raise ValueError(f"{member.name}: {image.shape[0]} bandas; esperado {N_RAW_CHANNELS}.")
                if dataset is None:
                    height, width = image.shape[1:]
                    dataset = handle.create_dataset(
                        "data",
                        shape=(len(members), N_RAW_CHANNELS, height, width),
                        dtype=np.float32,
                        chunks=(1, N_RAW_CHANNELS, height, width),
                        compression="lzf",
                    )
                elif image.shape[1:] != (height, width):
                    raise ValueError(f"{member.name}: shape {image.shape[1:]} diverge de {(height, width)}.")
                hours, lost = preprocess_active_fire(image[ACTIVE_FIRE_CHANNEL])
                image[ACTIVE_FIRE_CHANNEL] = hours
                lost_sub_hour += lost
                fire_pixels += int(np.count_nonzero(hours > 0))
                dataset[day] = image
            dataset.attrs["year"] = year
            dataset.attrs["fire_name"] = fire_name
            dataset.attrs["img_dates"] = dates
            dataset.attrs["lnglat"] = lnglat
        record = {
            "sidecar_version": SIDECAR_VERSION,
            "year": year,
            "fire_name": fire_name,
            "n_days": len(members),
            "height": int(height),
            "width": int(width),
            "dates": dates,
            "first_date": dates[0],
            "last_date": dates[-1],
            "date_gaps": date_gaps(parsed),
            "dates_outside_folder_year": sum(1 for d in dates if not d.startswith(str(year))),
            "fire_pixels": fire_pixels,
            "lost_detections_hhmm_lt_100": lost_sub_hour,
            "lnglat": lnglat,
            "crs": crs,
            "transform": transform,
            "source_fingerprint": source_fingerprint(members),
        }
        verify_event(temporary, record)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    record["size_bytes"] = target.stat().st_size
    return record


def verify_event(path: Path, expected: dict[str, Any]) -> None:
    """Relê shape e atributos do HDF5 e confere com o evento de origem (datas completas e centro)."""
    with open_h5(path) as handle:
        data = handle["data"]
        checks = {
            "shape": tuple(data.shape) == (expected["n_days"], N_RAW_CHANNELS, expected["height"], expected["width"]),
            "year": int(data.attrs["year"]) == expected["year"],
            "fire_name": str(data.attrs["fire_name"]) == expected["fire_name"],
            "img_dates": [str(d) for d in parse_dates(data.attrs["img_dates"])] == list(expected["dates"]),
            "lnglat": bool(np.allclose(np.asarray(data.attrs["lnglat"], dtype=float), expected["lnglat"])),
        }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"{path}: verificação falhou em {failed}.")


def load_completed(target: Path, members: list[Member]) -> dict[str, Any] | None:
    """Registro de um evento já convertido, se alvo e sidecar existem e a origem confere."""
    sidecar = sidecar_path(target)
    if not (target.exists() and sidecar.exists()):
        return None
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        record.get("sidecar_version") != SIDECAR_VERSION
        or record.get("source_fingerprint") != source_fingerprint(members)
        or record.get("size_bytes") != target.stat().st_size
    ):
        return None
    return record


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _library_versions() -> dict[str, str]:
    import h5py
    import rasterio

    return {
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
    }


class HttpRangeFile(io.RawIOBase):
    """Arquivo remoto somente leitura via HTTP Range (usado só para listar o zip no modo --remote)."""

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        import requests

        self.url, self.pos, self.timeout = url, 0, timeout
        self.session = requests.Session()
        self.size = int(self.session.head(url, allow_redirects=True, timeout=timeout).headers["content-length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer: Any) -> int:
        if not len(buffer) or self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        response = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=self.timeout)
        if response.status_code != 206:
            raise RuntimeError(f"Servidor não respondeu 206 a um Range (HTTP {response.status_code}).")
        chunk = response.content
        buffer[: len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


def build_manifest(records: list[dict[str, Any]], *, full_run: bool, resumed: int) -> dict[str, Any]:
    per_year: dict[str, dict[str, int]] = {}
    for record in records:
        entry = per_year.setdefault(str(record["year"]), {"events": 0, "images": 0, "fire_pixels": 0})
        entry["events"] += 1
        entry["images"] += record["n_days"]
        entry["fire_pixels"] += record["fire_pixels"]
    lowered: dict[tuple[int, str], int] = defaultdict(int)
    for record in records:
        lowered[(record["year"], record["fire_name"].lower())] += 1
    n_images = sum(r["n_days"] for r in records)
    return {
        "full_run": full_run,
        "n_events": len(records),
        "n_images": n_images,
        "resumed_events": resumed,
        "per_year": per_year,
        "counts_match_zenodo": (len(records) == EXPECTED_EVENTS and n_images == EXPECTED_IMAGES) if full_run else None,
        "events_with_date_gaps": sum(1 for r in records if r["date_gaps"]),
        "events_with_dates_outside_folder_year": sum(1 for r in records if r["dates_outside_folder_year"]),
        "lost_detections_hhmm_lt_100": sum(r["lost_detections_hhmm_lt_100"] for r in records),
        "fire_names_unique_case_insensitive": all(count == 1 for count in lowered.values()),
        "min_height": min(r["height"] for r in records),
        "min_width": min(r["width"] for r in records),
        "max_pixels_per_image": max(r["height"] * r["width"] for r in records),
        "crs": sorted({r["crs"] for r in records}),
        "pixel_sizes": sorted({(abs(r["transform"][0]), abs(r["transform"][4])) for r in records}),
        "total_size_gib": round(sum(r["size_bytes"] for r in records) / 2**30, 3),
        "events": records,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converte o WildfireSpreadTS (zip) em HDF5 por evento, com retomada.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--zip", type=str, default=None, help="Padrão: <data-root>/raw/WildfireSpreadTS.zip")
    parser.add_argument("--remote", action="store_true", help="Lê o zip direto do Zenodo via /vsicurl/ (smoke).")
    parser.add_argument("--years", type=int, nargs="*", default=list(YEARS))
    parser.add_argument("--max-events", type=int, default=None, help="Limita eventos por execução (smoke).")
    parser.add_argument("--force", action="store_true", help="Reconverte mesmo eventos já concluídos.")
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--min-disk-gib", type=float, default=5.0)
    parser.add_argument("--report", type=Path, default=None, help=f"Obrigatório fora da execução completa ({DEFAULT_REPORT}).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    full_run = args.max_events is None and not args.remote and sorted(args.years) == list(YEARS)
    if args.report is None:
        if not full_run:
            raise SystemExit("Execução parcial/remota: informe --report explícito para não sobrescrever o manifesto oficial.")
        args.report = DEFAULT_REPORT
    if args.remote:
        archive = f"/vsicurl/{ZENODO_URL}"
        with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(ZENODO_URL), buffer_size=1 << 20)) as handle:
            infos = handle.infolist()
        input_identity: dict[str, Any] = {"source": "remote", "url": ZENODO_URL}
    else:
        zip_path = Path(args.zip) if args.zip else args.data_root / "raw" / "WildfireSpreadTS.zip"
        archive = zip_path.resolve().as_posix()
        with zipfile.ZipFile(zip_path) as handle:
            infos = handle.infolist()
        input_identity = {
            "source": "local",
            "zip_name": zip_path.name,
            "zip_size_bytes": zip_path.stat().st_size,
            "download_manifest": "reports/download_manifest.json",
        }
    events = list_zip_events(infos)
    selected = [(key, members) for key, members in events.items() if key[0] in args.years]
    if args.max_events is not None:
        selected = selected[: args.max_events]
    if not selected:
        raise SystemExit(f"Nenhum evento encontrado para os anos {args.years}.")

    hdf5_root = args.data_root / "hdf5"
    for stale in hdf5_root.glob("*/*.hdf5.tmp"):
        stale.unlink()
    started = time.perf_counter()
    records, resumed = [], 0
    for position, ((year, fire_name), members) in enumerate(selected, start=1):
        check_resources(min_available_ram_gib=0.3, min_disk_free_gib=args.min_disk_gib)
        target = hdf5_root / str(year) / f"{fire_name}.hdf5"
        record = None if args.force else load_completed(target, members)
        status = "retomado"
        if record is None:
            record = convert_event(archive, year, fire_name, members, target)
            if not args.skip_hashes:
                record["sha256"] = sha256_file(target)
            write_json(record, sidecar_path(target))
            status = "convertido"
        else:
            resumed += 1
        records.append(record)
        print(
            f"[{position}/{len(selected)}] {year}/{fire_name}: {record['n_days']} dias "
            f"{record['height']}x{record['width']} gaps={len(record['date_gaps'])} "
            f"lost<100={record['lost_detections_hhmm_lt_100']} ({status})",
            flush=True,
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _git_commit(),
        "libraries": _library_versions(),
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "input": input_identity,
        "layout": "data float32 [T,23,H,W], chunks (1,23,H,W), lzf; attrs year, fire_name, img_dates, lnglat",
        "active_fire_preprocessing": "NaN -> 0; hhmm // 100 (idêntico ao oficial)",
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "resources": resource_snapshot(),
    }
    report.update(build_manifest(records, full_run=full_run, resumed=resumed))
    write_json(report, args.report)
    print(f"Eventos: {report['n_events']} ({resumed} retomados) | imagens: {report['n_images']} | {report['total_size_gib']} GiB")
    print(f"Manifesto: {args.report}")
    if full_run and not report["counts_match_zenodo"]:
        raise SystemExit("Contagens divergem do registro Zenodo (607 eventos / 13.607 imagens).")
    return report


if __name__ == "__main__":
    main()
