"""Utilitários de E/S e de recursos usados por todos os scripts: JSON determinístico, SHA-256 em blocos e a
checagem de RAM e disco que interrompe uma execução antes de esgotar a máquina."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resource_snapshot() -> dict[str, float | int | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(Path.cwd().anchor or Path.cwd()))
        process = psutil.Process(os.getpid())
        return {
            "available_ram_bytes": int(memory.available),
            "available_ram_gib": round(memory.available / 2**30, 3),
            "process_rss_bytes": int(process.memory_info().rss),
            "process_rss_mib": round(process.memory_info().rss / 2**20, 3),
            "disk_free_bytes": int(disk.free),
            "disk_free_gib": round(disk.free / 2**30, 3),
        }
    except Exception:
        return {
            "available_ram_bytes": None,
            "available_ram_gib": None,
            "process_rss_bytes": None,
            "process_rss_mib": None,
            "disk_free_bytes": None,
            "disk_free_gib": None,
        }


def check_resources(
    *,
    min_available_ram_gib: float = 0.5,
    min_disk_free_gib: float = 0.5,
) -> dict[str, float | int | None]:
    snapshot = resource_snapshot()
    ram = snapshot["available_ram_gib"]
    disk = snapshot["disk_free_gib"]
    if ram is not None and float(ram) < min_available_ram_gib:
        raise RuntimeError(
            f"RAM disponível insuficiente: {ram} GiB; mínimo={min_available_ram_gib} GiB."
        )
    if disk is not None and float(disk) < min_disk_free_gib:
        raise RuntimeError(
            f"Disco livre insuficiente: {disk} GiB; mínimo={min_disk_free_gib} GiB."
        )
    return snapshot
