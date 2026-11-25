"""Auditoria física da linha de base: quanto a previsão viola cada propriedade de `physics.py`, comparada ao observado.

Lê o `best.pt` de cada run (SHA-256 conferido contra o relatório do treino) e avalia **só a validação** do fold,
com o mesmo dataset de avaliação do treino. O teste nunca é aberto aqui: a auditoria decide quais penalidades
valem um treino, e decisão de modelagem não usa o teste. O livro-razão do teste não é tocado.

Uso típico (12 folds da `paper_es`, a partir de `reports/baseline_es_summary.json`):

    python scripts/physics_audit.py
    python scripts/physics_audit.py --runs baseline_es_f02 --max-windows 64   # ensaio rápido

Saídas: `reports/physics_audit/<run>.json` por run e `reports/physics_audit/summary.json`. Um run já auditado é
pulado (retomável), salvo com `--force`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core import grid_geometry  # noqa: E402
from scripts.env_report import provenance  # noqa: E402
from scripts.io_utils import check_resources, sha256_file, write_json  # noqa: E402
from scripts.physics import (  # noqa: E402
    DEFAULT_DELTA_STD,
    DEFAULT_RADIUS_PX,
    MONOTONE_DRIVERS,
    EventSums,
    PhysicsLayout,
    monotone_sums,
    perturb,
    property_sums,
)
from scripts.run_registry import REPORTS_DIR, RUNS_REPORT_DIR  # noqa: E402

OUTPUT_DIR = REPORTS_DIR / "physics_audit"
DEFAULT_SUMMARY = REPORTS_DIR / "baseline_es_summary.json"
PROPERTIES = ("nonburnable", "far", "wind", "upslope")
READING = {
    "nonburnable": "fração da massa de fogo em classes não combustíveis (menor = mais físico)",
    "far": "fração da massa de fogo novo a mais de `radius` px do fogo em t",
    "wind": "cosseno médio entre a normal externa da frente e o transporte do vento, nos pixels novos perto da frente",
    "upslope": "cosseno médio entre a normal externa da frente e a subida do terreno, nos pixels novos perto da frente",
    "monotone": "fração de pixels perto do fogo em que a probabilidade cai sob a perturbação propícia ao fogo",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mede violações físicas da linha de base na validação.")
    parser.add_argument("--runs", nargs="*", help="Nomes dos runs. Padrão: os folds de --summary, na ordem do fold.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--region", choices=("full", "crop32", "config"), default="full",
                        help="Região avaliada. `full` (padrão) vê a frente inteira; `config` usa a do treino.")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS_PX)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA_STD, help="Perturbação, em desvios-padrão.")
    parser.add_argument("--monotone", choices=("per_driver", "combined", "off"), default="per_driver")
    parser.add_argument("--max-windows", type=int, default=None, help="Ensaio: limita as janelas (sai em `demo/`).")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def runs_from_summary(path: Path) -> list[str]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    folds = summary["folds"]
    entries = folds.values() if isinstance(folds, dict) else folds
    return [entry["run_name"] for entry in sorted(entries, key=lambda e: e["fold"])]


def audit_run(run_name: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from scripts.stconv_dataset import WSTSWindowDataset
    from scripts.stconv_train import (FINE_BINS, LOGIT_RANGE, SPLITS_DIR, _loaders, build_model, configure_determinism,
                                           resolve_protocol_split,)
    from scripts.core import check_normalization

    report = json.loads((RUNS_REPORT_DIR / f"{run_name}.json").read_text(encoding="utf-8"))
    if report.get("kind") != "full":
        raise SystemExit(f"{run_name!r} não é um run completo ({report.get('kind')!r}).")
    data_root = Path(report["data_root"])
    checkpoint = data_root / report["checkpoint"]
    if sha256_file(checkpoint) != report["checkpoint_sha256"]:
        raise SystemExit(f"SHA-256 de {checkpoint} diverge do registrado no treino.")
    normalization_path = data_root / report["normalization_file"]
    if sha256_file(normalization_path) != report["normalization_sha256"]:
        raise SystemExit(f"SHA-256 de {normalization_path} diverge do registrado no treino.")
    config = report["config"]
    configure_determinism(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    split_kind = config.get("split_kind", "temporal")
    definition_path = (Path(config["split_definition"]) if config.get("split_definition")
                       else (SPLITS_DIR / f"{split_kind}_split.json" if split_kind != "temporal" else None))
    spec = resolve_protocol_split(data_root / "hdf5", split_kind, config["fold"], definition_path)
    check_normalization(normalization, spec)
    events = spec.events["validation"]  # nunca "test"
    region = config.get("eval_region", "full") if args.region == "config" else args.region
    dataset = WSTSWindowDataset(
        events, normalization, mode="eval", t_in=config["t_in"], patch=config["patch"], halo=config["halo"],
        include_forecast_precip=config["include_forecast_precip"], drop_groups=config["drop_groups"],
        eval_region=region, max_windows=args.max_windows,
    )
    layout = PhysicsLayout.from_channels(dataset.channel_names, normalization)
    gammas = torch.tensor([grid_geometry(e.lnglat)["gamma_deg"] for e in events], dtype=torch.float32)
    model = build_model(len(dataset.channel_names), config["t_in"], config, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.eval()
    batch_size = args.eval_batch_size or config.get("eval_batch_size") or 8
    args.seed = config["seed"]
    loader = _loaders(args, dataset, shuffle=False, batch_size=batch_size, device=device)
    perturbations = {
        "per_driver": {name: (name,) for name, _ in MONOTONE_DRIVERS},
        "combined": {"combined": tuple(name for name, _ in MONOTONE_DRIVERS)},
        "off": {},
    }[args.monotone]

    def predict(x: Any) -> Any:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            return torch.sigmoid(model(x)[:, :, -1:].float())

    low, high = LOGIT_RANGE
    threshold_logit = low + (high - low) * report["validation_threshold_bin"] / FINE_BINS
    threshold = float(1.0 / (1.0 + np.exp(-threshold_logit)))
    sums = EventSums()
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            fields = {k: batch[k].to(device) for k in ("mask", "fire_t", "y")}
            event_ids = [int(e) for e in batch["event"]]
            gamma = gammas[event_ids].to(device)
            prob = predict(x)
            binary = (prob >= threshold).float()
            for source, field in (("model", prob), ("model_binary", binary), ("observed", fields["y"])):
                for name, (num, den) in property_sums(field, fields, x, layout, gamma, args.radius).items():
                    sums.add(f"{name}.{source}", event_ids, num.cpu().tolist(), den.cpu().tolist())
            for label, drivers in perturbations.items():
                shifted = predict(perturb(x, layout, drivers, args.delta))
                for name, (num, den) in monotone_sums(prob, shifted, fields, args.radius).items():
                    sums.add(f"monotone.{label}.{name}", event_ids, num.cpu().tolist(), den.cpu().tolist())
    elapsed = time.perf_counter() - started
    dataset.close()

    results: dict[str, Any] = {}
    for offset, name in enumerate(PROPERTIES):
        results[name] = {"reading": READING[name],
                         **sums.summary(f"{name}.model", n_boot=args.bootstrap, seed=offset,
                                        reference=f"{name}.observed")}
        results[f"{name}_binary"] = {"reading": READING[name] + "; previsão binarizada no limiar F2 da validação",
                                     **sums.summary(f"{name}.model_binary", n_boot=args.bootstrap, seed=offset + 20,
                                                    reference=f"{name}.observed")}
    for label in perturbations:
        results[f"monotone.{label}"] = {
            "reading": READING["monotone"],
            "violation": sums.summary(f"monotone.{label}.violation", n_boot=args.bootstrap, seed=10),
            "mean_change": sums.summary(f"monotone.{label}.mean_change", n_boot=args.bootstrap, seed=11),
        }
    return {
        "run_name": run_name,
        "fold": config["fold"],
        "recipe": report.get("recipe") or config.get("recipe"),
        "cell": "validation",
        "region": region,
        "radius_px": args.radius,
        "delta_std": args.delta,
        "threshold_probability": threshold,
        "monotone_mode": args.monotone,
        "kind": "demo" if args.max_windows is not None else "full",
        "n_windows": len(dataset.windows),
        "n_events": len(events),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "elapsed_seconds": round(elapsed, 1),
        "device": device.type,
        "results": results,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance(),
    }


def aggregate(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """Média ± desvio entre folds do valor do modelo, do observado e da diferença (descritivo, folds não independentes)."""
    table: dict[str, Any] = {}
    keys = sorted({k for a in audits for k in a["results"]})
    for key in keys:
        rows = [a["results"][key] for a in audits if key in a["results"]]
        entry: dict[str, Any] = {"folds": len(rows)}
        for field in ("value", "reference_value", "difference"):
            values = [r[field] for r in rows if r.get(field) is not None]
            if values:
                entry[field] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        if "violation" in rows[0]:
            values = [r["violation"]["value"] for r in rows if r["violation"].get("value") is not None]
            if values:
                entry["violation"] = {"mean": float(np.mean(values)),
                                      "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        excluded = [r for r in rows if "difference_ci95" in r and (r["difference_ci95"][0] > 0 or r["difference_ci95"][1] < 0)]
        if "difference" in entry:
            entry["folds_difference_ci_excludes_zero"] = len(excluded)
        table[key] = entry
    return table


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    runs = args.runs or runs_from_summary(args.summary)
    output_dir = args.output_dir / ("demo" if args.max_windows is not None else "")
    check_resources(min_available_ram_gib=0.5, min_disk_free_gib=1)
    audits = []
    for run_name in runs:
        path = output_dir / f"{run_name}.json"
        if path.exists() and not args.force:
            print(f"[{run_name}] já auditado, pulando ({path})", flush=True)
            audits.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        print(f"[{run_name}] auditando a validação...", flush=True)
        audit = audit_run(run_name, args)
        write_json(audit, path)
        audits.append(audit)
        r = audit["results"]
        line = ", ".join(f"{k}: {r[k].get('value', float('nan')):.3f} (obs {r[k].get('reference_value') or float('nan'):.3f})"
                         for k in PROPERTIES if r[k].get("value") is not None)
        print(f"[{run_name}] {audit['elapsed_seconds']:.0f} s — {line}", flush=True)
    summary = {
        "runs": [a["run_name"] for a in audits],
        "aggregate": aggregate(audits),
        "reading": "desvio entre folds não é erro-padrão; o IC por evento de cada fold está nos JSONs por run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(summary, output_dir / "summary.json")
    return summary


if __name__ == "__main__":
    main()
