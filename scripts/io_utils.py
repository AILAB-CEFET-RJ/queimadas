"""Utilitários de E/S e de recursos usados por todos os scripts: JSON determinístico, SHA-256 em blocos e a
checagem de RAM e disco que interrompe uma execução antes de esgotar a máquina."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def atomic_write(path: str | Path, writer: Any) -> Path:
    """Grava num temporário do mesmo diretório, força para o disco e só então troca pelo destino.

    Numa queda de energia o destino fica com a versão antiga ou com a nova, nunca truncado: o `fsync` antes do
    `os.replace` garante que o rename (registrado no journal) não chegue ao disco antes dos dados.
    `writer(stream)` recebe o arquivo binário aberto.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    return atomic_write(path, lambda stream: stream.write(text.encode("utf-8")))


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


def power_status() -> dict[str, Any]:
    """Se o computador está na tomada (`on_ac`) e a carga da bateria; `None` onde não dá para saber."""
    if sys.platform != "win32":
        return {"on_ac": None, "battery_percent": None}
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return {"on_ac": None, "battery_percent": None}
    return {"on_ac": {0: False, 1: True}.get(status.ACLineStatus),
            "battery_percent": None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent)}


@contextmanager
def prevent_sleep() -> Iterator[dict[str, Any]]:
    """Impede a suspensão automática enquanto o bloco roda e libera ao sair, inclusive por erro.

    No Windows usa `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`: o sistema não suspende por
    inatividade, mas a tela pode apagar. Não impede a suspensão por tampa fechada nem por bateria crítica, que
    seguem as configurações de energia. Fora do Windows não faz nada e diz como fazer (`systemd-inhibit`).
    """
    status: dict[str, Any] = {"platform": sys.platform, "active": False}
    if sys.platform != "win32":
        status["note"] = "sem bloqueio automático; no Linux rode sob `systemd-inhibit --what=sleep ...`"
        yield status
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.windll.kernel32
    status["active"] = bool(kernel32.SetThreadExecutionState(es_continuous | es_system_required))
    try:
        yield status
    finally:
        kernel32.SetThreadExecutionState(es_continuous)
