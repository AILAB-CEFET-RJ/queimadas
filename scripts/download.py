from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import check_resources, write_json
from scripts.core import DEFAULT_DATA_ROOT


ZENODO_RECORD = "8006177"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DEFAULT_REPORT = Path("reports/download_manifest.json")
CHUNK_SIZE = 8 * 1024 * 1024


def fetch_record(api_url: str = ZENODO_API) -> dict[str, Any]:
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    return response.json()


def md5_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url: str, target: Path, expected_size: int, *, max_retries: int = 20) -> None:
    """Baixa `url` para `target`, retomando a partir do tamanho parcial já gravado."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size == expected_size:
        return
    for attempt in range(1, max_retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise RuntimeError(f"Arquivo parcial maior que o esperado: {partial}")
        if offset == expected_size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                if offset and response.status_code != 206:
                    raise RuntimeError(
                        f"Servidor não aceitou retomada (HTTP {response.status_code}); apague {partial}."
                    )
                response.raise_for_status()
                started, written = time.monotonic(), 0
                with partial.open("ab") as stream:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        stream.write(chunk)
                        written += len(chunk)
                        if written % (512 * CHUNK_SIZE) < CHUNK_SIZE:
                            done = offset + written
                            rate = written / max(time.monotonic() - started, 1e-6) / 2**20
                            print(
                                f"{target.name}: {done / 2**30:.2f}/{expected_size / 2**30:.2f} GiB "
                                f"({rate:.1f} MiB/s)",
                                flush=True,
                            )
        except (requests.RequestException, RuntimeError) as error:
            if isinstance(error, RuntimeError) and "retomada" in str(error):
                raise
            wait = min(60, 2**attempt)
            print(f"Falha na tentativa {attempt}: {error}. Nova tentativa em {wait}s.", flush=True)
            time.sleep(wait)
    if not partial.exists() or partial.stat().st_size != expected_size:
        raise RuntimeError(f"Download incompleto de {url}.")
    partial.replace(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa o WildfireSpreadTS do Zenodo (retomável) e confere MD5 contra o registro."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Subconjunto de arquivos do registro (padrão: todos).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.data_root / "raw"
    record = fetch_record()
    files = [item for item in record["files"] if args.files is None or item["key"] in args.files]
    total = sum(int(item["size"]) for item in files)
    check_resources(min_disk_free_gib=total / 2**30 + 5)

    entries = []
    for item in files:
        target = raw_dir / item["key"]
        algorithm, expected = item["checksum"].split(":", 1)
        if algorithm != "md5":
            raise ValueError(f"Checksum não suportado: {item['checksum']}")
        download_resumable(item["links"]["self"], target, int(item["size"]))
        observed = md5_file(target)
        if observed != expected:
            raise RuntimeError(f"MD5 divergente em {target}: {observed} != {expected}")
        print(f"OK {item['key']} md5={observed}", flush=True)
        entries.append(
            {
                "key": item["key"],
                "size_bytes": int(item["size"]),
                "md5": observed,
                "url": item["links"]["self"],
                "local_path": target.as_posix(),
            }
        )

    write_json(
        {
            "zenodo_record": record["id"],
            "doi": record.get("doi"),
            "title": record["metadata"].get("title"),
            "version": record["metadata"].get("version"),
            "license": (record["metadata"].get("license") or {}).get("id"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": entries,
            "md5_verified": True,
        },
        args.report,
    )
    print(f"Manifesto: {args.report}")


if __name__ == "__main__":
    main()
