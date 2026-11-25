"""Pipeline completo num comando: ambiente → clones fixados → download → conversão → treino → pacote de resultados.

Feito para quem vai rodar sem acompanhar: cada etapa é idempotente, então **a retomada é rodar o mesmo comando de
novo** depois de qualquer queda (energia, driver, reboot). O que já terminou é pulado:

1. **Ambiente**: Python, pacotes, `nvidia-smi` e um teste do torch na GPU num processo filho (inclusive se o build do
   torch tem kernels para a arquitetura da placa: uma RTX 50xx precisa de CUDA >= 12.8, e o `requirements-cuda.txt`
   é cu126) e espaço em disco. Grava `reports/ambiente.json` e `reports/pipeline_preflight.json`.
2. **Clones de terceiros** (`.stconvs2s` e `.wildfirespreadts`), clonados se faltarem e conferidos no commit fixado.
3. **Download** do Zenodo (`download.py`: retomável por HTTP Range, MD5 conferido). Pulado se a conversão já está
   completa.
4. **Conversão** para HDF5 (`convert_hdf5.py`: retomável por evento). Pulada se os 607 eventos e o manifesto já
   existem. Com `--delete-zip`, o zip (~45 GiB) é apagado depois de uma conversão completa e conferida.
5. **Treino** pelo executor que já existe, sem mudar nada do protocolo: `run_experiment.py` (experimentos do
   catálogo, por exemplo `tin1_es`) ou `run_baseline.py` (receitas `paper_es` e `paper`). Um fold com relatório é
   pulado, e um fold interrompido continua do `last.pt` da última época.
6. **Pacote** `reports/entrega_<treino>_<data>.zip` com resumos, relatórios por fold, logs, livro-razão do teste e
   manifestos: é o que volta para quem analisa.

Antes do treino completo roda um run curto de conferência (fold 2, 2 eventos por ano, 4 passos, ~1 min), que nunca
avalia o teste e vai para `reports/demo_runs/`: uma instalação quebrada para ali, em minutos, e não depois de horas de
tentativas do executor. `--smoke` para depois dele; `--skip-smoke` o pula. `--check-only` só faz as etapas 1 e 2.

Uso (da raiz do repositório):

    python scripts/run_pipeline.py --train tin1_es --folds 3 4 5 7 8 9 10 11
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Só módulos sem torch: o pipeline fica vivo durante todo o treino e não deve ocupar memória da GPU.
from scripts import run_baseline as baseline  # noqa: E402
from scripts import run_experiment as experiment  # noqa: E402
from scripts.convert_hdf5 import EXPECTED_EVENTS, EXPECTED_IMAGES  # noqa: E402
from scripts.core import DEFAULT_DATA_ROOT  # noqa: E402
from scripts.io_utils import write_json  # noqa: E402
from scripts.run_registry import DEMO_REPORT_DIR, REPORTS_DIR, RUNS_REPORT_DIR, TEST_LEDGER  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
SCRIPTS_REL = SCRIPTS.relative_to(REPO_ROOT).as_posix()
HDF5_MANIFEST = REPORTS_DIR / "hdf5_manifest.json"
DOWNLOAD_MANIFEST = REPORTS_DIR / "download_manifest.json"
PREFLIGHT_REPORT = REPORTS_DIR / "pipeline_preflight.json"
PIPELINE_LOG = "pipeline.log"
ZIP_NAME = "WildfireSpreadTS.zip"

# Clones de terceiros, no commit em que os runs foram feitos (o mesmo do README).
CLONES: dict[str, dict[str, Any]] = {
    ".stconvs2s": {
        "url": "https://github.com/noemicho/stconvs2s", "branch": "noemi",
        "commit": "c355e9e023a84625f8e31afd386e47fea98708d6", "required": True,
        "why": "o treino importa model.stconvs2s.STConvS2S_C",
    },
    ".wildfirespreadts": {
        "url": "https://github.com/SebastianGer/WildfireSpreadTS", "branch": None,
        "commit": "f62dc49f2c2a6d249c071533ae5340c0d01bfb95", "required": False,
        "why": "o relatório de cada fold registra o pos_weight da tabela oficial",
    },
}

# Tamanhos medidos na conversão de referência (zip do Zenodo v1.0 e HDF5 lzf), com folga para checkpoints e logs.
ZIP_GIB = 45.1
HDF5_GIB = 24.2
MARGIN_GIB = 10.0
MIN_RAM_GIB = 16.0

# O que cada `--train` roda. `experiment` usa o catálogo de run_experiment.py; `recipe` usa run_baseline.py.
TRAIN_CHOICES = ("none", *sorted(experiment.EXPERIMENTS), *sorted(baseline.RECIPE_RUNS))

TORCH_PROBE = r"""
import json, sys
import torch
out = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
if out["cuda_available"]:
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    archs = torch.cuda.get_arch_list()
    out.update(device=torch.cuda.get_device_name(0), capability=arch, arch_list=archs,
               vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    def cc(name):
        digits = name.split("_")[1].rstrip("abcdefghijklmnopqrstuvwxyz")
        return int(digits[:-1]), int(digits[-1])
    # Um cubin sm_XY roda em sm_XZ com Z >= Y (mesma família); PTX compute_XY compila para qualquer cc >= XY.
    out["arch_supported"] = any(
        (a.startswith("sm_") and cc(a)[0] == major and cc(a)[1] <= minor)
        or (a.startswith("compute_") and cc(a) <= (major, minor)) for a in archs)
    try:
        conv = torch.nn.Conv3d(4, 4, 3, padding=1).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            y = conv(torch.randn(2, 4, 3, 16, 16, device="cuda"))
        y.float().mean().backward()
        torch.cuda.synchronize()
        out["amp_conv3d"] = "ok"
    except Exception as error:  # noqa: BLE001
        out["amp_conv3d"] = f"{type(error).__name__}: {error}"
print("PROBE " + json.dumps(out))
"""


class PipelineError(RuntimeError):
    """Etapa que não pode seguir; a mensagem diz o que fazer."""


# --------------------------------------------------------------------------- estado das etapas


def conversion_status(data_root: Path, manifest_path: Path = HDF5_MANIFEST) -> dict[str, Any]:
    """Se a conversão está completa: 607 HDF5 com sidecar, nenhum `.tmp` e o manifesto da execução completa."""
    hdf5 = data_root / "hdf5"
    files = sorted(hdf5.glob("*/*.hdf5")) if hdf5.exists() else []
    with_sidecar = sum(1 for path in files if path.with_suffix(".json").exists())
    temporaries = len(list(hdf5.glob("*/*.hdf5.tmp"))) if hdf5.exists() else 0
    manifest_ok = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ok = bool(manifest.get("full_run") and manifest.get("counts_match_zenodo")
                           and manifest.get("n_events") == EXPECTED_EVENTS and manifest.get("n_images") == EXPECTED_IMAGES)
    complete = len(files) == EXPECTED_EVENTS and with_sidecar == EXPECTED_EVENTS and not temporaries and manifest_ok
    return {"hdf5_files": len(files), "with_sidecar": with_sidecar, "temporaries": temporaries,
            "manifest_ok": manifest_ok, "complete": complete}


def zip_status(data_root: Path, manifest_path: Path = DOWNLOAD_MANIFEST) -> dict[str, Any]:
    """Se o zip local tem o tamanho que o manifesto de download registrou com MD5 conferido."""
    target = data_root / "raw" / ZIP_NAME
    size = target.stat().st_size if target.exists() else None
    partial = target.with_suffix(target.suffix + ".part")
    expected = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("md5_verified"):
            expected = next((f["size_bytes"] for f in manifest.get("files", []) if f.get("key") == ZIP_NAME), None)
    return {"path": target.as_posix(), "size_bytes": size, "expected_bytes": expected,
            "partial_bytes": partial.stat().st_size if partial.exists() else 0,
            "verified": size is not None and expected is not None and size == expected}


def disk_needed_gib(conversion: dict[str, Any], zip_info: dict[str, Any]) -> float:
    """Espaço livre ainda necessário: o que falta baixar, o que falta converter e uma folga."""
    need = MARGIN_GIB
    if not conversion["complete"]:
        have = (zip_info["size_bytes"] or 0) + zip_info["partial_bytes"]
        need += max(ZIP_GIB - have / 2**30, 0.0)
        need += HDF5_GIB * (1 - conversion["with_sidecar"] / EXPECTED_EVENTS)
    return round(need, 1)


def free_gib(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return round(shutil.disk_usage(probe).free / 2**30, 1)


# --------------------------------------------------------------------------- comandos


def script(name: str) -> str:
    return f"{SCRIPTS_REL}/{name}"


def run_names(train: str, folds: list[int]) -> list[str]:
    if train in experiment.EXPERIMENTS:
        return [experiment.run_name(fold, train) for fold in folds]
    return [baseline.run_name(fold, train) for fold in folds]


def summary_path(train: str) -> Path:
    if train in experiment.EXPERIMENTS:
        return experiment.EXPERIMENTS[train]["summary"]
    return baseline.RECIPE_RUNS[train]["summary"]


def runner_log(train: str) -> str:
    if train in experiment.EXPERIMENTS:
        return experiment.EXPERIMENTS[train]["runner_log"]
    return baseline.RECIPE_RUNS[train]["runner_log"]


def train_command(args: argparse.Namespace) -> list[str]:
    """Linha do executor de treino, com os mesmos padrões que ele já tem (nada do protocolo muda aqui)."""
    common = ["--data-root", str(args.data_root), "--folds", *map(str, args.folds),
              "--accumulate", str(args.accumulate), "--num-workers", str(args.num_workers)]
    if args.train in experiment.EXPERIMENTS:
        command = [sys.executable, "-u", script("run_experiment.py"), "--experiment", args.train, *common,
                   "--stall-minutes", f"{args.stall_minutes:g}"]
        return command + (["--allow-cpu"] if args.allow_cpu else [])
    return [sys.executable, "-u", script("run_baseline.py"), "--recipe", args.train, *common]


def smoke_command(args: argparse.Namespace, run_name: str, report: Path) -> list[str]:
    """Run curto de demonstração com a receita e a configuração do `--train` escolhido (fold 2, nunca testa)."""
    if args.train in experiment.EXPERIMENTS:
        spec = experiment.EXPERIMENTS[args.train]
        train_script, recipe, extra = spec["script"], spec["recipe"], list(spec["args"])
    else:
        train_script, recipe, extra = baseline.TRAIN_SCRIPT, args.train, []
    command = [sys.executable, "-u", train_script, "--recipe", recipe, "--fold", "2", "--run-name", run_name,
               "--data-root", str(args.data_root), "--accumulate", str(args.accumulate), "--num-workers", "0",
               "--max-events", "2", "--max-windows", "16", "--max-steps", "4", "--bootstrap", "10",
               "--report", str(report), "--force", *extra]
    return command + ([] if args.allow_cpu else ["--require-cuda"])


# --------------------------------------------------------------------------- etapas


def run_step(command: list[str], log: baseline.Log, env: dict[str, str] | None = None) -> int:
    """Roda um passo com a saída no console (quem lançou acompanha) e devolve o código de saída."""
    log(f"$ {' '.join(command)}")
    return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode


def probe_torch(runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                env: dict[str, str] | None = None) -> dict[str, Any]:
    done = runner([sys.executable, "-c", TORCH_PROBE], cwd=REPO_ROOT, capture_output=True, text=True,
                  timeout=600, env=env)
    lines = [line for line in (done.stdout or "").splitlines() if line.startswith("PROBE ")]
    if done.returncode != 0 or not lines:
        tail = ((done.stderr or "") + (done.stdout or "")).strip().splitlines()[-3:]
        return {"error": f"o teste do torch falhou (código {done.returncode}): {' | '.join(tail) or 'sem saída'}"}
    return json.loads(lines[-1][len("PROBE "):])


def gpu_problems(probe: dict[str, Any]) -> list[str]:
    """O que impede treinar na GPU, com a correção; vazio quando está tudo certo."""
    if "error" in probe:
        return [probe["error"] + ". Instale o PyTorch de um dos requirements-cuda*.txt (veja o README)."]
    if not probe.get("cuda_available"):
        return [f"o torch {probe.get('torch')} não enxerga CUDA (build CUDA {probe.get('cuda_build')}). Confira o "
                "nvidia-smi e se o torch instalado é o de GPU (requirements-cuda*.txt), não o de CPU."]
    if probe.get("amp_conv3d") == "ok":
        return []
    if not probe.get("arch_supported"):
        return [f"o torch {probe['torch']} (CUDA {probe['cuda_build']}) não tem kernels para {probe['device']} "
                f"({probe['capability']}; o build tem {', '.join(probe['arch_list'])}). Placas RTX 50xx (sm_120) "
                "precisam de CUDA >= 12.8: pip install -r requirements-cuda130.txt (driver >= 580)."]
    return [f"a GPU não rodou uma Conv3d em fp16: {probe.get('amp_conv3d')}"]


def preflight(args: argparse.Namespace, log: baseline.Log, env: dict[str, str]) -> dict[str, Any]:
    """Confere o que dá para conferir antes de baixar 45 GiB ou treinar por dias; erra cedo, com a correção."""
    problems: list[str] = []
    warnings: list[str] = []
    version = sys.version_info
    if (version.major, version.minor) != (3, 12):
        warnings.append(f"Python {version.major}.{version.minor}: a suíte e o lock foram validados no 3.12.")
    missing = []
    for module in ("numpy", "scipy", "h5py", "rasterio", "requests", "psutil", "torch"):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(module)
    if missing:
        problems.append(f"pacotes ausentes: {', '.join(missing)}. Rode pip install -r requirements-lock.txt e o "
                        "requirements-cuda*.txt da sua GPU.")
    if shutil.which("git") is None:
        problems.append("git não encontrado no PATH (necessário para os clones fixados).")
    probe: dict[str, Any] = {}
    if args.train != "none" and not missing:
        if not args.allow_cpu and shutil.which("nvidia-smi") is None:
            problems.append("nvidia-smi não encontrado: instale o driver NVIDIA (o treino exige GPU).")
        probe = probe_torch(env=env)
        log(f"[ambiente] torch: {json.dumps(probe, ensure_ascii=False)}")
        if not args.allow_cpu:
            problems += gpu_problems(probe)
        elif probe.get("vram_gib") is None:
            warnings.append("treino na CPU (--allow-cpu): só para depuração, não é comparável aos runs da GPU.")
    conversion = conversion_status(args.data_root)
    zip_info = zip_status(args.data_root)
    need, free = disk_needed_gib(conversion, zip_info), free_gib(args.data_root)
    if free < need:
        problems.append(f"espaço livre insuficiente em {args.data_root}: {free} GiB livres, {need} GiB necessários "
                        f"(zip ~{ZIP_GIB:g} GiB + HDF5 ~{HDF5_GIB:g} GiB + folga {MARGIN_GIB:g} GiB, menos o que já "
                        "existe). Aponte --data-root para outro disco.")
    try:
        import psutil

        ram = round(psutil.virtual_memory().total / 2**30, 1)
        if ram < MIN_RAM_GIB:
            warnings.append(f"RAM total {ram} GiB (< {MIN_RAM_GIB:g}): use --num-workers 2 se faltar memória.")
    except ImportError:
        ram = None
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": sys.platform, "torch_probe": probe, "ram_gib": ram,
        "data_root": str(args.data_root), "disk_free_gib": free, "disk_needed_gib": need,
        "conversion": conversion, "zip": zip_info, "problems": problems, "warnings": warnings,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(report, PREFLIGHT_REPORT)
    for warning in warnings:
        log(f"[ambiente] AVISO: {warning}")
    if problems:
        raise PipelineError("checagem do ambiente falhou:\n  - " + "\n  - ".join(problems))
    log(f"[ambiente] ok | disco livre {free} GiB (necessário {need}) | RAM {ram} GiB")
    return report


def _git(path: Path, *command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(path), *command], capture_output=True, text=True)


def ensure_clones(log: baseline.Log, root: Path = REPO_ROOT) -> None:
    """Clona os repositórios de terceiros que faltam e confere o commit dos que existem (sem alterá-los)."""
    for folder, spec in CLONES.items():
        path = root / folder
        if not path.exists():
            log(f"[clones] clonando {spec['url']} em {folder} ({spec['why']})")
            command = ["git", "clone", "-c", "core.longpaths=true"]
            command += ["-b", spec["branch"]] if spec["branch"] else []
            done = subprocess.run([*command, spec["url"], str(path)], capture_output=True, text=True)
            if done.returncode == 0:
                done = _git(path, "-c", "advice.detachedHead=false", "checkout", spec["commit"])
            if done.returncode != 0:
                message = f"não consegui clonar {spec['url']} @ {spec['commit'][:7]}: {done.stderr.strip()}"
                if spec["required"]:
                    raise PipelineError(message)
                log(f"[clones] AVISO: {message} (opcional: {spec['why']})")
                continue
        head = _git(path, "rev-parse", "HEAD").stdout.strip()
        if head != spec["commit"]:
            message = (f"{folder} está em {head[:7] or '?'}, não no commit fixado {spec['commit'][:7]}. Rode: "
                       f"git -C {folder} fetch && git -C {folder} checkout {spec['commit']}")
            if spec["required"]:
                raise PipelineError(message)
            log(f"[clones] AVISO: {message}")
        else:
            log(f"[clones] {folder} @ {head[:7]} ok")


def ensure_data(args: argparse.Namespace, log: baseline.Log) -> dict[str, Any]:
    """Download e conversão, cada um pulado quando já concluído; os dois scripts retomam de onde pararam."""
    status = conversion_status(args.data_root)
    if status["complete"]:
        log(f"[dados] conversão completa ({status['hdf5_files']} eventos); download e conversão pulados")
    else:
        zip_info = zip_status(args.data_root)
        if zip_info["verified"]:
            log(f"[dados] zip já baixado e conferido ({zip_info['size_bytes'] / 2**30:.1f} GiB); download pulado")
        else:
            log("[dados] baixando o WildfireSpreadTS do Zenodo (~45 GiB; retomável, MD5 conferido)")
            if run_step([sys.executable, "-u", script("download.py"), "--data-root", str(args.data_root)], log):
                raise PipelineError("o download falhou; rode o mesmo comando de novo para retomar.")
        log(f"[dados] convertendo para HDF5 ({status['with_sidecar']}/{EXPECTED_EVENTS} eventos já prontos)")
        if run_step([sys.executable, "-u", script("convert_hdf5.py"), "--data-root", str(args.data_root)], log):
            raise PipelineError("a conversão falhou; rode o mesmo comando de novo para retomar.")
        status = conversion_status(args.data_root)
        if not status["complete"]:
            raise PipelineError(f"a conversão terminou incompleta: {status}")
        log(f"[dados] conversão completa: {status['hdf5_files']} eventos")
    zip_path = args.data_root / "raw" / ZIP_NAME
    if args.delete_zip and zip_path.exists():
        zip_path.unlink()
        log(f"[dados] --delete-zip: {zip_path} apagado (a conversão está completa e conferida)")
    return status


def run_smoke(args: argparse.Namespace, log: baseline.Log, env: dict[str, str]) -> dict[str, Any]:
    """Treino curto no fold 2 com a configuração do `--train`: confere GPU, AMP, dados e normalização."""
    baseline.ensure_normalization(args.data_root, 2, log)
    run_name = f"pipeline_smoke_{args.train}"
    report = DEMO_REPORT_DIR / f"{run_name}.json"
    log(f"[smoke] run curto ({run_name}): 2 eventos por ano, 4 passos, sem teste")
    if run_step(smoke_command(args, run_name, report), log, env):
        raise PipelineError(f"o run de smoke falhou; veja a saída acima ({run_name}).")
    payload = json.loads(report.read_text(encoding="utf-8"))
    device = payload.get("provenance", {}).get("device") or payload.get("device")
    log(f"[smoke] ok | relatório {report.relative_to(REPO_ROOT).as_posix()} | dispositivo {device}")
    return payload


def bundle(args: argparse.Namespace, log: baseline.Log, stamp: str) -> Path:
    """Zip com o que quem analisa precisa: resumo, relatórios e logs dos folds, livro-razão e manifestos."""
    names = run_names(args.train, args.folds)
    logs = REPORTS_DIR / "logs"
    candidates = [summary_path(args.train), TEST_LEDGER, logs / runner_log(args.train), logs / PIPELINE_LOG]
    candidates += [REPORTS_DIR / name for name in (HDF5_MANIFEST.name, DOWNLOAD_MANIFEST.name, PREFLIGHT_REPORT.name,
                                                    "ambiente.json")]
    for name in names:
        candidates += [RUNS_REPORT_DIR / f"{name}.json", logs / f"{name}.log"]
    target = REPORTS_DIR / f"entrega_{args.train}_{stamp}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in candidates:
            if path.exists() and path.is_relative_to(REPORTS_DIR):
                archive.write(path, path.relative_to(REPORTS_DIR).as_posix())
    log(f"[entrega] {target.relative_to(REPO_ROOT).as_posix()} ({target.stat().st_size / 2**20:.1f} MiB)")
    return target


class SingleInstance:
    """Trava em `<data-root>/.pipeline.lock`: duas cópias do pipeline nos mesmos dados disputariam os mesmos runs."""

    def __init__(self, data_root: Path) -> None:
        self.path = data_root / ".pipeline.lock"

    def __enter__(self) -> "SingleInstance":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text(encoding="utf-8").strip() or 0)
            except ValueError:
                pid = 0
            if pid and _pid_alive(pid):
                raise PipelineError(f"já há um pipeline rodando nestes dados (pid {pid}, {self.path}). Se ele não "
                                    "existe mais, apague o arquivo e rode de novo.")
        self.path.write_text(str(os.getpid()), encoding="utf-8")
        return self

    def __exit__(self, *_: Any) -> None:
        self.path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        return True


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline completo (download, conversão, treino), retomável: rode o mesmo comando de novo.")
    parser.add_argument("--train", choices=TRAIN_CHOICES, default="tin1_es",
                        help="tin1_es: experimento T_in = 1; paper_es / paper: baseline; none: só os dados.")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="Folds a treinar, na ordem dada (padrão: os 12, na ordem do executor).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Dados brutos, HDF5, normalizações e checkpoints (~80 GiB livres; ~35 com --delete-zip "
                             "depois da conversão).")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Índice ou UUID da GPU (vira CUDA_VISIBLE_DEVICES do treino).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2,
                        help="Micro-lotes por passo; 2 é o dos runs já feitos e deve ser mantido para parear.")
    parser.add_argument("--stall-minutes", type=float, default=45.0, help="Watchdog do executor de experimentos.")
    parser.add_argument("--delete-zip", action="store_true", help="Apaga o zip (~45 GiB) depois da conversão completa.")
    parser.add_argument("--check-only", action="store_true",
                        help="Só confere ambiente, GPU, disco e clones (~1 min), sem baixar nem treinar.")
    parser.add_argument("--smoke", action="store_true",
                        help="Para depois do run curto de conferência, sem o treino completo.")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Não roda o run curto de conferência antes do treino completo.")
    parser.add_argument("--allow-cpu", action="store_true", help="Só depuração: não exige GPU.")
    args = parser.parse_args(argv)
    if not args.data_root.is_absolute():
        args.data_root = (REPO_ROOT / args.data_root).resolve()
    args.folds = list(baseline.FOLD_ORDER) if args.folds is None else args.folds
    unknown = sorted(set(args.folds) - set(baseline.FOLD_ORDER))
    if unknown:
        parser.error(f"folds inexistentes: {unknown}")
    if args.smoke and args.train == "none":
        parser.error("--smoke precisa de um --train (a configuração que o run curto confere).")
    return args


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        # Console do Windows em cp1252 ou saída redirecionada: acentos e setas não podem derrubar um run de dias.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    log = baseline.Log(REPORTS_DIR / "logs" / PIPELINE_LOG)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    began = time.perf_counter()
    log(f"Início | treino {args.train}{' (smoke)' if args.smoke else ''} | folds {args.folds} | dados {args.data_root}")
    try:
        with SingleInstance(args.data_root):
            preflight(args, log, env)
            subprocess.run([sys.executable, script("env_report.py")], cwd=REPO_ROOT, capture_output=True)
            ensure_clones(log)
            if args.check_only:
                log("Fim | --check-only: ambiente, GPU, disco e clones ok")
                return 0
            ensure_data(args, log)
            if args.train == "none":
                log("Fim | --train none: dados prontos")
                return 0
            if args.smoke or not args.skip_smoke:
                run_smoke(args, log, env)
            if args.smoke:
                log("Fim | smoke ok: a instalação treina na GPU. Rode o mesmo comando sem --smoke.")
                return 0
            log(f"[treino] {args.train}, folds {args.folds} (retomável; um fold pronto é pulado)")
            code = run_step(train_command(args), log, env)
            summary = json.loads(summary_path(args.train).read_text(encoding="utf-8")) \
                if summary_path(args.train).exists() else {}
            target = bundle(args, log, stamp)
    except PipelineError as error:
        log(f"PARADA: {error}")
        return 2
    aggregate = summary.get("aggregate", {})
    done = [f["fold"] for f in summary.get("folds", []) if f.get("test_ap") is not None]
    pending = [fold for fold in args.folds if fold not in done]
    log(f"Fim | código {code} | folds testados {done} | pendentes {pending or 'nenhum'} | falhas "
        f"{sorted(summary.get('failures', {})) or 'nenhuma'} | média {aggregate.get('test_ap_mean')} | "
        f"{round((time.perf_counter() - began) / 3600, 2)} h | envie {target.name}")
    if pending:
        log("Há folds pendentes: rode o mesmo comando de novo (o que terminou é pulado, o que parou retoma).")
    return 0 if code == 0 and not pending else 1


if __name__ == "__main__":
    sys.exit(main())
