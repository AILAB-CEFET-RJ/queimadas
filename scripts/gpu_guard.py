"""Proteções do executor contra GPU perdida e treino travado (2026-11-18).

Motivação: no cefet a GPU0 caiu do barramento PCIe (`lspci` mostra `rev ff`) em 16/11 e em 17/11. Na segunda vez,
o processo do fold 2 ficou 14 h travado sem gravar época, e a queda quebrou o NVML da máquina inteira: os folds
seguintes da GPU1 (7 e 8) começaram sem CUDA e o `stconv_train.py` caiu em silêncio para a CPU (~10x mais lento e
outro determinismo numérico). Este módulo dá ao executor duas checagens:

- `gpu_health`: antes de cada tentativa de treino ou teste, confere o `nvidia-smi` (NVML) e, num processo filho,
  se o torch enxerga e usa a GPU visível (`CUDA_VISIBLE_DEVICES` é herdado). Falhou → o executor para sem iniciar
  o fold.
- `execute_with_watchdog`: roda um passo como o `run_baseline.execute`, mas encerra o processo (e os workers do
  DataLoader) se o log do fold ficar `stall_minutes` sem uma linha `época` nova.

Não importa torch (o executor fica vivo por dias; o teste do torch roda num processo filho).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
# Código de saída devolvido quando o watchdog encerra o passo (o mesmo do `timeout` do coreutils).
WATCHDOG_EXIT_CODE = 124
EPOCH_PREFIX = "época "
TORCH_PROBE = (
    "import torch, sys\n"
    "if not torch.cuda.is_available():\n"
    "    sys.exit('torch.cuda.is_available() é False')\n"
    "x = torch.ones(1, device='cuda:0')\n"
    "print('ok', torch.cuda.get_device_name(0), float((x + x).item()))\n"
)


class GpuUnavailable(RuntimeError):
    """A GPU não passou na checagem: o executor deve parar sem iniciar o fold."""


class TrainingStalled(RuntimeError):
    """O watchdog encerrou um fold travado: o executor deve parar (a GPU provavelmente caiu)."""


def _run(command: list[str], timeout: float) -> tuple[int, str]:
    try:
        done = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as error:
        return 127, str(error)
    except subprocess.TimeoutExpired:
        return WATCHDOG_EXIT_CODE, f"sem resposta em {timeout:.0f} s"
    return done.returncode, (done.stdout + done.stderr).strip()


def gpu_health(timeout: float = 180.0, runner: Callable[[list[str], float], tuple[int, str]] = _run) -> tuple[bool, str]:
    """`(ok, mensagem)`: o NVML responde e o torch usa a GPU visível deste executor."""
    code, out = runner(["nvidia-smi", "--query-gpu=index,name,pci.bus_id", "--format=csv,noheader"], timeout)
    if code != 0 or not out:
        return False, f"nvidia-smi falhou (código {code}): {out.splitlines()[-1] if out else 'sem saída'}"
    code, probe = runner([sys.executable, "-c", TORCH_PROBE], timeout)
    if code != 0 or not probe.splitlines() or not probe.splitlines()[-1].startswith("ok "):
        return False, f"torch não usa a GPU (código {code}): {probe.splitlines()[-1] if probe else 'sem saída'}"
    gpus = "; ".join(line.strip() for line in out.splitlines())
    return True, f"{probe.splitlines()[-1]} | nvidia-smi: {gpus}"


def count_epoch_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        return sum(1 for line in stream if line.startswith(EPOCH_PREFIX))


def _terminate(process: subprocess.Popen, grace: float) -> None:
    """SIGTERM no grupo do processo (pega os workers do DataLoader); SIGKILL se não sair em `grace` s."""
    if os.name == "posix":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                continue
    else:
        process.kill()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass


def execute_with_watchdog(
    command: list[str], log_path: Path, *, stall_minutes: float, poll_seconds: float = 30.0,
    grace_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Roda `command` com a saída anexada a `log_path`; encerra se não surgir `época` nova em `stall_minutes`.

    O relógio do watchdog começa no lançamento (a primeira época inclui carregar os dados) e reinicia a cada linha
    `época` nova. `stall_minutes <= 0` desliga o watchdog. Devolve o código de saída, ou `WATCHDOG_EXIT_CODE` quando
    encerrou o passo (anotado no próprio log do fold).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] $ {' '.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        epochs, last_progress = count_epoch_lines(log_path), clock()
        while True:
            code = process.poll()
            if code is not None:
                return code
            now_epochs = count_epoch_lines(log_path)
            if now_epochs != epochs:
                epochs, last_progress = now_epochs, clock()
            elif stall_minutes > 0 and clock() - last_progress > stall_minutes * 60:
                _terminate(process, grace_seconds)
                stream.write(f"[{datetime.now().isoformat(timespec='seconds')}] [watchdog] {stall_minutes:g} min sem "
                             f"linha '{EPOCH_PREFIX.strip()}' nova; processo {process.pid} encerrado "
                             f"(código {process.poll()})\n")
                stream.flush()
                return WATCHDOG_EXIT_CODE
            sleep(poll_seconds)
