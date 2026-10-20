from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import resource_snapshot, sha256_file, write_json


DEFAULT_REPORT = Path("reports/ambiente.json")
REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-wsts-cuda.txt",
    "requirements-wsts-lock.txt",
)
KEY_PACKAGES = ("numpy", "scipy", "pandas", "torch", "h5py", "torchmetrics", "rasterio", "xarray")
CRLF, LF = b"\r\n", b"\n"


def _git(path: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(path: Path) -> dict[str, Any]:
    status = _git(path, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git(path, "rev-parse", "HEAD"),
        "remote": _git(path, "remote", "get-url", "origin"),
        "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def sha256_text_normalized(path: Path) -> str:
    """SHA-256 do texto com fins de linha normalizados para LF (estável entre checkouts)."""
    content = path.read_bytes().replace(CRLF, LF)
    return hashlib.sha256(content).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def torch_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["device_total_memory_gib"] = round(props.total_memory / 2**30, 3)
        info["nvidia_driver"] = _nvidia_driver()
    return info


def _nvidia_driver() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0] if output else None


def provenance(files: dict[str, Path | None] | None = None, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Bloco de proveniência para artefatos de resultado: código, clones, ambiente e insumos (SHA-256)."""
    return {
        "repo": git_state(repo_root),
        "stconvs2s": git_state(repo_root / ".stconvs2s") if (repo_root / ".stconvs2s").exists() else None,
        "python": sys.version.split()[0],
        "torch": torch_info(),
        "code_sha256_lf": {
            path.relative_to(repo_root).as_posix(): sha256_text_normalized(path)
            for path in sorted((repo_root / "scripts").glob("*.py"))
            if path.exists()
        },
        "input_sha256": {
            name: sha256_file(path) if path is not None and Path(path).exists() else None
            for name, path in (files or {}).items()
        },
    }


def build_report(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_executable": Path(sys.executable).relative_to(repo_root).as_posix()
        if Path(sys.executable).is_relative_to(repo_root)
        else "<fora do repositório>",
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in KEY_PACKAGES},
        "torch": torch_info(),
        "requirements_sha256_lf": {
            name: sha256_text_normalized(repo_root / name)
            for name in REQUIREMENT_FILES
            if (repo_root / name).exists()
        },
        "repo": git_state(repo_root),
        "third_party_repos": {
            name: git_state(repo_root / name)
            for name in (".stconvs2s", ".wildfirespreadts")
            if (repo_root / name).exists()
        },
        "resources": resource_snapshot(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registra versões, GPU e hashes de dependências da frente WildfireSpreadTS."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    write_json(report, args.report)
    print(f"Python {report['python']} | torch {report['torch']['version']} | CUDA {report['torch']['cuda_available']}")
    print(f"Relatório: {args.report}")


if __name__ == "__main__":
    main()
