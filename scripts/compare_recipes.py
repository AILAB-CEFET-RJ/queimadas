"""Comparação pareada `paper` × `paper_es` nos folds que rodaram nas duas receitas.

Pré-especificada antes do primeiro run da `paper_es`:

- Δ = AP(`paper_es`) − AP(`paper`) por fold, na métrica do baseline (vista `all` com `eval_region=crop32`);
- IC 95% do Δ por bootstrap pareado de eventos: cada reamostragem sorteia os eventos de teste com reposição e
  calcula o AP das duas receitas nos mesmos eventos, a partir das contagens por evento salvas no teste (bins
  grossos; o Δ pontual oficial vem do histograma fino dos relatórios);
- Δ médio por ano de teste e nos folds pareados, só descritivos: os folds não são independentes (folds com o mesmo
  ano de teste testam os mesmos eventos) e há uma semente só, então o IC é por fold e ausência de significância não é
  equivalência;
- igualdade da `val_loss` por época até o primeiro corte do lr como teste de determinismo na GPU.

Não avalia nada de novo no teste: só lê os relatórios e os `.npz` gravados no teste único de cada run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.env_report import provenance  # noqa: E402
from scripts.io_utils import sha256_file, write_json  # noqa: E402
from scripts.run_baseline import read_report  # noqa: E402
from scripts.run_registry import REPORTS_DIR  # noqa: E402
from scripts.stconv_train import average_precision_from_counts  # noqa: E402

VIEW = "all"
N_BOOT = 1000
SEED = 0
OUTPUT = REPORTS_DIR / "baseline_recipe_comparison.json"


def load_test_counts(report: dict[str, Any], data_root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """Arquivo, índices dos eventos e histogramas grossos de logits por evento (`[evento, classe, bin]`) do teste."""
    path = data_root / report["test"]["per_event_counts"]
    with np.load(path) as saved:
        return path, saved[f"{VIEW}_events"], saved[f"{VIEW}_logit_histograms"]


def same_weights(reference: dict[str, Any], candidate: dict[str, Any], data_root: Path) -> bool:
    """Pesos iguais nos checkpoints testados das duas receitas.

    Comparar o SHA dos dois arquivos não serve (o `config` gravado no checkpoint traz o nome da receita); o SHA de
    cada um é conferido contra o do relatório, para que os pesos comparados sejam os que foram testados.
    """
    import torch

    weights = []
    for report in (reference, candidate):
        path = data_root / report["checkpoint"]
        if sha256_file(path) != report["checkpoint_sha256"]:
            raise ValueError(f"{path}: SHA-256 diferente do registrado em {report['run_name']}")
        weights.append(torch.load(path, map_location="cpu", weights_only=True)["model"])
    a, b = weights
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def paired_delta_bootstrap(reference: np.ndarray, candidate: np.ndarray, n_boot: int = N_BOOT,
                           seed: int = SEED) -> dict[str, Any]:
    """IC 95% de AP(candidata) − AP(referência), reamostrando os mesmos eventos para as duas."""
    if reference.shape != candidate.shape:
        raise ValueError(f"contagens com formas diferentes: {reference.shape} e {candidate.shape}")

    def delta(index: np.ndarray) -> float | None:
        a, b = reference[index].sum(axis=0), candidate[index].sum(axis=0)
        ap_a, ap_b = average_precision_from_counts(a[0], a[1]), average_precision_from_counts(b[0], b[1])
        return None if ap_a is None or ap_b is None else ap_b - ap_a

    n = len(reference)
    rng = np.random.default_rng(seed)
    values = [v for v in (delta(rng.integers(0, n, n)) for _ in range(n_boot)) if v is not None]
    if not values:
        raise ValueError("nenhuma reamostragem com positivos nas duas receitas: o AP não é definido")
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "point_coarse_bins": delta(np.arange(n)),
        "ci95": [float(low), float(high)],
        "share_below_zero": float(np.mean(np.asarray(values) < 0)),
        "excludes_zero": bool(low > 0 or high < 0),
        "n_events": n,
        "n_boot": n_boot,
        "n_valid_resamples": len(values),
        "seed": seed,
    }


def prefix_check(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """`val_loss` por época das duas receitas até o primeiro corte do lr da candidata (devem ser iguais)."""
    cuts = candidate["protocol"]["optimization"].get("lr_reductions") or []
    last = cuts[0]["after_epoch"] if cuts else candidate["best_epoch"]
    losses = [{e["epoch"]: e["val_loss"] for e in report["history"] if e.get("validated") and e["epoch"] <= last}
              for report in (reference, candidate)]
    common = sorted(set(losses[0]) & set(losses[1]))
    max_diff = max((abs(losses[0][e] - losses[1][e]) for e in common), default=None)
    return {"epochs_compared": len(common), "up_to_epoch": last, "max_abs_val_loss_diff": max_diff,
            "identical": bool(common) and len(common) == last and max_diff == 0.0}


def compare_fold(fold: int, report_dir: Path, data_root: Path) -> dict[str, Any] | None:
    reference, candidate = read_report(fold, report_dir, "paper"), read_report(fold, report_dir, "paper_es")
    if reference is None or candidate is None or "test" not in reference or "test" not in candidate:
        return None
    if reference["test"]["events"] != candidate["test"]["events"]:
        raise ValueError(f"fold {fold}: as receitas foram testadas em eventos diferentes")
    ref_path, ref_events, ref_counts = load_test_counts(reference, data_root)
    cand_path, cand_events, cand_counts = load_test_counts(candidate, data_root)
    if not np.array_equal(ref_events, cand_events):
        raise ValueError(f"fold {fold}: índices de eventos diferentes nas contagens salvas")
    ap = {name: report["test"]["metrics"][VIEW]["model"]["average_precision"]
          for name, report in (("paper", reference), ("paper_es", candidate))}
    return {
        "fold": fold,
        "test_year": "/".join(map(str, candidate["protocol"]["split"]["test"])),
        "test_ap": ap,
        "delta": ap["paper_es"] - ap["paper"],
        "best_epoch": {"paper": reference["best_epoch"], "paper_es": candidate["best_epoch"]},
        "same_weights": same_weights(reference, candidate, data_root),
        "same_test_counts": bool(np.array_equal(ref_counts, cand_counts)),
        "bootstrap": paired_delta_bootstrap(ref_counts, cand_counts),
        "prefix": prefix_check(reference, candidate),
        "inputs": {"paper": ref_path, "paper_es": cand_path},
    }


def summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    by_year: dict[str, list[float]] = {}
    for entry in folds:
        by_year.setdefault(str(entry["test_year"]), []).append(entry["delta"])
    deltas = [entry["delta"] for entry in folds]
    return {
        "n_folds": len(folds),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "mean_test_ap": {name: float(np.mean([e["test_ap"][name] for e in folds])) if folds else None
                         for name in ("paper", "paper_es")},
        "mean_delta_by_test_year": {year: {"n": len(v), "mean": float(np.mean(v))} for year, v in sorted(by_year.items())},
        "folds_with_ci_excluding_zero": [e["fold"] for e in folds if e["bootstrap"]["excludes_zero"]],
        "folds_with_same_weights": [e["fold"] for e in folds if e["same_weights"]],
        "folds_with_identical_prefix": [e["fold"] for e in folds if e["prefix"]["identical"]],
        "reading": ("Δ = paper_es − paper. IC por fold (eventos de teste reamostrados, uma semente); as médias são "
                    "descritivas, com folds não independentes. Ausência de significância não é equivalência."),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Comparação pareada paper × paper_es por bootstrap de eventos.")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR / "stconv_runs")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="padrão: o `data_root` gravado nos relatórios")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    folds = []
    for fold in args.folds:
        report = read_report(fold, args.report_dir, "paper_es")
        data_root = args.data_root or (Path(report["data_root"]) if report else None)
        entry = compare_fold(fold, args.report_dir, data_root) if data_root else None
        if entry is not None:
            folds.append(entry)
    inputs = {f"f{e['fold']:02d}_{name}": path for e in folds for name, path in e.pop("inputs").items()}
    payload = {"view": VIEW, "folds_requested": args.folds, "folds": folds, "summary": summarize(folds),
               "provenance": provenance(inputs)}
    write_json(payload, args.output)
    for entry in folds:
        b = entry["bootstrap"]
        print(f"fold {entry['fold']:2d} ({entry['test_year']}): paper {entry['test_ap']['paper']:.3f} | "
              f"paper_es {entry['test_ap']['paper_es']:.3f} | Δ {entry['delta']:+.3f} "
              f"IC [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | mesmos pesos: {entry['same_weights']} | "
              f"prefixo idêntico: {entry['prefix']['identical']} ({entry['prefix']['epochs_compared']} épocas)")
    print(f"Δ médio {payload['summary']['mean_delta']:+.3f} em {len(folds)} folds → {args.output}")
    return payload


if __name__ == "__main__":
    main()
