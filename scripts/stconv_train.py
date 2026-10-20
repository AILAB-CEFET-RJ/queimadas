"""Adapter de treino/avaliação WSTS × STConvS2S (substitui as modificações perdidas do fork).

Importa apenas `model.stconvs2s.STConvS2S_C` do clone `.stconvs2s` (noemicho/stconvs2s @ noemi).
Loss, métricas, baselines e protocolo vivem aqui, versionados.

Protocolo:
- dois protocolos de split, ambos fixados antes do treino (`--split-kind`): `temporal` (fold por ano, comparável
  ao benchmark) e `spatiotemporal` (bloco × ano — teste em lugar e período novos, mais os cenários `future_only`
  e `space_only`, que decompõem a perda). Em ambos a normalização é ajustada só no treino e conferida pelas
  chaves dos eventos (`check_normalization`);
- `pos_weight` estimado na distribuição de recortes do treino;
- checkpoint pela maior AP de validação; threshold pelo maior F2 de validação;
- o teste NÃO é tocado no modo de treino. Ele é avaliado depois, com `--evaluate-test --run <nome>`,
  a partir do checkpoint e do threshold já gravados; cada avaliação é registrada num livro-razão
  append-only (`reports/test_ledger.jsonl`) e a reavaliação exige `--force`.

A saída do STConv tem `T_in` posições temporais; usamos `logits[:, :, -1:]` como previsão de t+1 —
convenção de leitura (o `conv_final` tem padding temporal simétrico), não uma saída causal por passo.
X só contém dias <= t, então não há vazamento.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import check_resources, resource_snapshot, sha256_file, write_json
from scripts.core import (
    DEFAULT_DATA_ROOT,
    DEFAULT_FOLD,
    DEFAULT_T_IN,
    DROPPABLE_GROUPS,
    SPATIOTEMPORAL_EVAL_CELLS,
    SPATIOTEMPORAL_FOLD,
    SplitSpec,
    check_normalization,
    resolve_split,
)
from scripts.env_report import provenance
from scripts.stconv_dataset import (
    DEFAULT_HALO,
    DEFAULT_PATCH,
    WSTSWindowDataset,
    receptive_radius,
    worker_init_fn,
)


STCONV_ROOT = REPO_ROOT / ".stconvs2s"
LOGIT_RANGE = (-20.0, 20.0)
FINE_BINS = 2**16
EVENT_BINS = 2**12
RUNS_REPORT_DIR = REPO_ROOT / "reports" / "stconv_runs"
DEMO_REPORT_DIR = REPO_ROOT / "reports" / "demo_runs"
SMOKE_REPORT = REPO_ROOT / "reports" / "stconv_smoke_metrics.json"
TIMING_REPORT = REPO_ROOT / "reports" / "stconv_timing.json"
TEST_LEDGER = REPO_ROOT / "reports" / "test_ledger.jsonl"
HDF5_MANIFEST = REPO_ROOT / "reports" / "hdf5_manifest.json"
SPLITS_DIR = REPO_ROOT / "config" / "splits"
VIEWS = ("all", "crop32", "new_vs_t", "new_vs_window")
BASELINES = {"persistence_all_detections_t": "fire_t", "persistence_last_pass_t": "fire_last_t", "persistence_window_union": "fire_window"}


# --------------------------------------------------------------------------- métricas em streaming


def _logit_bins(logits: np.ndarray, bins: int) -> np.ndarray:
    low, high = LOGIT_RANGE
    scaled = (np.clip(np.asarray(logits, dtype=np.float64), low, high) - low) / (high - low)
    return np.minimum((scaled * bins).astype(np.int64), bins - 1)


def average_precision_from_counts(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """AP em degraus (mesma definição do sklearn, empates agrupados por bin) a partir de histogramas."""
    n_pos = positive.sum()
    if n_pos == 0:
        return None
    tp = np.cumsum(positive[::-1])
    fp = np.cumsum(negative[::-1])
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=(tp + fp) > 0)
    delta_recall = np.diff(np.concatenate([[0.0], tp / n_pos]))
    return float(np.sum(precision * delta_recall))


class LogitHistogram:
    """Histogramas de logits por classe em bins uniformes de `LOGIT_RANGE` (fino nos escores baixos)."""

    def __init__(self, bins: int = FINE_BINS) -> None:
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)

    def update(self, logits: np.ndarray, labels: np.ndarray) -> None:
        labels = np.asarray(labels).ravel().astype(bool)
        index = _logit_bins(np.asarray(logits).ravel(), self.bins)
        self.positive += np.bincount(index[labels], minlength=self.bins)
        self.negative += np.bincount(index[~labels], minlength=self.bins)

    def threshold_of(self, bin_index: int) -> dict[str, float]:
        low, high = LOGIT_RANGE
        logit = low + (high - low) * bin_index / self.bins
        return {"bin": int(bin_index), "logit": float(logit), "probability": float(1.0 / (1.0 + np.exp(-logit)))}

    def summary(self, beta: float = 2.0) -> dict[str, Any]:
        n_pos, n_neg = int(self.positive.sum()), int(self.negative.sum())
        if n_pos == 0:
            return {"n_positive": 0, "n_negative": n_neg, "average_precision": None}
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        precision = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=(tp + fp) > 0)
        recall = tp / n_pos
        b2 = beta**2
        fbeta = np.divide((1 + b2) * precision * recall, b2 * precision + recall, out=np.zeros(len(tp)), where=(b2 * precision + recall) > 0)
        best = int(np.argmax(fbeta))
        return {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "prevalence": n_pos / (n_pos + n_neg),
            "average_precision": average_precision_from_counts(self.positive, self.negative),
            "best_f2": float(fbeta[best]),
            "best_threshold": self.threshold_of(self.bins - 1 - best),
            "precision_at_best": float(precision[best]),
            "recall_at_best": float(recall[best]),
        }

    def at_bin(self, bin_index: int, beta: float = 2.0) -> dict[str, Any]:
        """Métricas prevendo positivo para todo logit no bin `bin_index` ou acima."""
        tp, fp = int(self.positive[bin_index:].sum()), int(self.negative[bin_index:].sum())
        fn, tn = int(self.positive[:bin_index].sum()), int(self.negative[:bin_index].sum())
        return {"threshold": self.threshold_of(bin_index), **confusion_metrics(tp, fp, fn, tn, beta)}


def confusion_metrics(tp: int, fp: int, fn: int, tn: int, beta: float = 2.0) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta**2
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f2": fbeta}


class EvaluationAccumulator:
    """Modelo e baselines nos mesmos pixels, por vista, com contagens por evento para IC e comparações pareadas.

    Por evento e vista guarda: histograma grosso de logits por classe (`EVENT_BINS`) e a matriz de confusão
    de cada baseline. Isso permite IC da AP por bootstrap de eventos, IC pareado modelo × persistência no
    threshold de validação e, a partir do `.npz` salvo, bootstrap pareado entre configurações.
    """

    def __init__(self) -> None:
        self.model = {view: LogitHistogram() for view in VIEWS}
        self.baselines = {view: {name: np.zeros(4, dtype=np.int64) for name in BASELINES} for view in VIEWS}
        self.per_event: dict[str, dict[int, np.ndarray]] = {
            view: defaultdict(lambda: np.zeros((2, EVENT_BINS), dtype=np.int64)) for view in VIEWS
        }
        self.per_event_baselines: dict[str, dict[int, np.ndarray]] = {
            view: defaultdict(lambda: np.zeros((len(BASELINES), 4), dtype=np.int64)) for view in VIEWS
        }

    def _views(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        valid = batch["mask"].astype(bool)
        return {
            "all": valid,
            "crop32": valid & batch["crop32"].astype(bool),
            "new_vs_t": valid & ~batch["fire_t"].astype(bool),
            "new_vs_window": valid & ~batch["fire_window"].astype(bool),
        }

    @staticmethod
    def _confusion(predicted: np.ndarray, truth: np.ndarray) -> list[int]:
        return [
            int(np.count_nonzero(predicted & truth)),
            int(np.count_nonzero(predicted & ~truth)),
            int(np.count_nonzero(~predicted & truth)),
            int(np.count_nonzero(~predicted & ~truth)),
        ]

    def update(self, logits: np.ndarray, batch: dict[str, np.ndarray]) -> None:
        target = batch["y"].astype(bool)
        views = self._views(batch)
        events = np.asarray(batch["event"]).reshape(-1)
        for view, selected in views.items():
            self.model[view].update(logits[selected], target[selected])
            for sample, event in enumerate(events):
                chosen = selected[sample]
                labels = target[sample][chosen]
                index = _logit_bins(logits[sample][chosen], EVENT_BINS)
                counts = self.per_event[view][int(event)]
                counts[0] += np.bincount(index[labels], minlength=EVENT_BINS)
                counts[1] += np.bincount(index[~labels], minlength=EVENT_BINS)
                for position, key in enumerate(BASELINES.values()):
                    confusion = self._confusion(batch[key][sample].astype(bool)[chosen], labels)
                    self.per_event_baselines[view][int(event)][position] += confusion
                    self.baselines[view][list(BASELINES)[position]] += confusion

    def _stacked(self, view: str) -> tuple[list[int], np.ndarray, np.ndarray]:
        events = sorted(self.per_event[view])
        return (
            events,
            np.stack([self.per_event[view][e] for e in events]) if events else np.zeros((0, 2, EVENT_BINS), dtype=np.int64),
            np.stack([self.per_event_baselines[view][e] for e in events]) if events else np.zeros((0, len(BASELINES), 4), dtype=np.int64),
        )

    def bootstrap_ap(self, view: str, n_boot: int = 1000, seed: int = 0) -> dict[str, Any] | None:
        """IC 95% da AP por bootstrap de eventos (bins grossos; a AP pontual vem do histograma fino)."""
        _, stacked, _ = self._stacked(view)
        if len(stacked) < 3:
            return None
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(n_boot):
            sample = stacked[rng.integers(0, len(stacked), len(stacked))].sum(axis=0)
            ap = average_precision_from_counts(sample[0], sample[1])
            if ap is not None:
                values.append(ap)
        if not values:
            return None
        low, high = np.percentile(values, [2.5, 97.5])
        return {"n_events": len(stacked), "n_boot": n_boot, "ci95": [float(low), float(high)]}

    def bootstrap_vs_persistence(self, view: str, threshold_bin: int, n_boot: int = 1000, seed: int = 0) -> dict[str, Any] | None:
        """IC pareado (eventos) de F2(modelo no threshold de validação) − F2(persistência de t)."""
        _, stacked, baselines = self._stacked(view)
        if len(stacked) < 3:
            return None
        coarse = min(threshold_bin * EVENT_BINS // FINE_BINS, EVENT_BINS - 1)
        model_counts = np.stack([
            stacked[:, 0, coarse:].sum(axis=1), stacked[:, 1, coarse:].sum(axis=1),
            stacked[:, 0, :coarse].sum(axis=1), stacked[:, 1, :coarse].sum(axis=1),
        ], axis=1)
        persistence = baselines[:, 0]

        def delta(index: np.ndarray) -> float:
            model = confusion_metrics(*map(int, model_counts[index].sum(axis=0)))["f2"]
            base = confusion_metrics(*map(int, persistence[index].sum(axis=0)))["f2"]
            return model - base

        rng = np.random.default_rng(seed)
        values = [delta(rng.integers(0, len(stacked), len(stacked))) for _ in range(n_boot)]
        low, high = np.percentile(values, [2.5, 97.5])
        return {
            "comparison": "F2(modelo, threshold de validação em bins grossos) − F2(persistência de todas as detecções de t)",
            "point": delta(np.arange(len(stacked))),
            "ci95": [float(low), float(high)],
            "model_beats_persistence": bool(low > 0),
            "n_events": len(stacked),
        }

    def save_per_event(self, path: Path) -> Path:
        """Persiste contagens por evento (para bootstrap pareado entre configurações)."""
        arrays: dict[str, np.ndarray] = {"baseline_names": np.array(list(BASELINES))}
        for view in VIEWS:
            events, stacked, baselines = self._stacked(view)
            arrays[f"{view}_events"] = np.asarray(events)
            arrays[f"{view}_logit_histograms"] = stacked
            arrays[f"{view}_baseline_confusion"] = baselines
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def summary(self, threshold_bin: int | None = None, *, bootstrap: int = 0) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for view in VIEWS:
            model = self.model[view].summary()
            entry: dict[str, Any] = {
                "model": model,
                "prevalence_baseline_ap": model.get("prevalence"),
                "baselines": {name: confusion_metrics(*map(int, counts)) for name, counts in self.baselines[view].items()},
            }
            if threshold_bin is not None:
                entry["model_at_validation_threshold"] = self.model[view].at_bin(threshold_bin)
            if bootstrap:
                entry["ap_bootstrap_events"] = self.bootstrap_ap(view, bootstrap)
                if threshold_bin is not None:
                    entry["f2_vs_persistence_paired_bootstrap"] = self.bootstrap_vs_persistence(view, threshold_bin, bootstrap)
            payload[view] = entry
        return payload


# --------------------------------------------------------------------------- treino


def masked_bce_with_logits(logits: Any, target: Any, mask: Any, pos_weight: float) -> Any:
    import torch
    import torch.nn.functional as F

    weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def estimate_pos_weight(dataset: WSTSWindowDataset, max_samples: int, seed: int) -> dict[str, Any]:
    """Negativos/positivos sob a mesma distribuição de recortes do treino."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    positive = negative = 0
    for index in indices[:max_samples]:
        item = dataset[index]
        valid = item["mask"].bool()
        positive += int(item["y"][valid].sum())
        negative += int(valid.sum()) - int(item["y"][valid].sum())
    if positive == 0:
        raise ValueError(f"Nenhum positivo em {min(max_samples, len(indices))} recortes de treino; pos_weight indefinido.")
    return {"samples": min(max_samples, len(indices)), "positive": positive, "negative": negative, "pos_weight": negative / positive}


def build_model(n_channels: int, t_in: int, config: dict[str, Any], device: Any) -> Any:
    if str(STCONV_ROOT) not in sys.path:
        sys.path.insert(0, str(STCONV_ROOT))
    from model.stconvs2s import STConvS2S_C

    return STConvS2S_C(
        input_size=(config["batch_size"], n_channels, t_in, config["patch"], config["patch"]),
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        kernel_size=3,
        device=device,
        dropout_rate=config["dropout"],
        step=t_in,
        output_channels=1,
    ).to(device)


def evaluate(model: Any, loader: Iterable[dict[str, Any]], device: Any, use_amp: bool) -> EvaluationAccumulator:
    import torch

    accumulator = EvaluationAccumulator()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch["x"].to(device, non_blocking=True))[:, :, -1:]
            arrays = {k: v.numpy() if hasattr(v, "numpy") else np.asarray(v) for k, v in batch.items() if k != "x"}
            accumulator.update(logits.float().cpu().numpy(), arrays)
    return accumulator


def configure_determinism(seed: int) -> dict[str, Any]:
    # Necessário para kernels cuBLAS determinísticos; precisa estar definido antes da primeira chamada CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "use_deterministic_algorithms": "warn_only",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "note": "conv3d em CUDA pode não ter kernel determinístico; reprodutibilidade bit a bit não é garantida",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina/avalia STConvS2S no WildfireSpreadTS (próximo dia).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold oficial. Padrão: 2 no protocolo temporal, 6 no espaço-temporal.")
    parser.add_argument("--split-kind", choices=("temporal", "spatiotemporal"), default="temporal",
                        help="`temporal`: fold por ano, comparável ao benchmark. `spatiotemporal`: bloco × ano, "
                             "com teste em lugar e período novos e os cenários `future_only`/`space_only`.")
    parser.add_argument("--split-definition", type=Path, default=None,
                        help="Padrão: config/splits/<split-kind>_split.json; se existir, o split recomputado é "
                             "conferido contra ela e qualquer divergência é erro.")
    parser.add_argument("--evaluate-train-cell", action="store_true",
                        help="No teste do protocolo espaço-temporal, mede também a célula de treino (dentro da "
                             "amostra, logo um teto otimista) para fechar a interação dos deslocamentos.")
    parser.add_argument("--normalization", type=Path, default=None, help="Padrão: <data-root>/stats/fold<F>_train_normalization.json")
    parser.add_argument("--t-in", type=int, default=DEFAULT_T_IN)
    parser.add_argument("--patch", type=int, default=DEFAULT_PATCH)
    parser.add_argument("--halo", type=int, default=DEFAULT_HALO)
    parser.add_argument("--include-forecast-precip", action="store_true")
    parser.add_argument("--drop-groups", nargs="*", default=[], choices=sorted(DROPPABLE_GROUPS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pos-weight-samples", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=0, help="0 recomendado nesta máquina (RAM ~2 GiB livre).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-windows", type=int, default=None, help="Smoke: limita janelas de treino/validação.")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Demonstração: usa só os N primeiros eventos de cada ano. Inelegível para o teste.")
    parser.add_argument("--timing-only", action="store_true", help="Mede 1 época no conjunto completo; inelegível para o teste.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--evaluate-test", action="store_true", help="Não treina: avalia o teste do run --run-name.")
    parser.add_argument("--force", action="store_true", help="Permite sobrescrever relatório/reavaliar teste (registrado).")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=TEST_LEDGER, help="Livro-razão append-only de avaliações do teste.")
    return parser.parse_args(argv)


def sample_events(events: list[Any], limit: int | None) -> list[Any]:
    """Amostra determinística de eventos para demonstração: os `limit` primeiros de cada ano.

    `list_events` já devolve ordenado por ano e nome, então a amostra é reproduzível e cobre todos os
    anos do conjunto. Um run que use isto nunca pode avaliar o teste.
    """
    if limit is None:
        return events
    by_year: dict[int, list[Any]] = {}
    for event in events:
        by_year.setdefault(event.year, []).append(event)
    return [event for year in sorted(by_year) for event in by_year[year][:limit]]


def _loaders(args: argparse.Namespace, dataset: WSTSWindowDataset, *, shuffle: bool, batch_size: int, device: Any) -> Any:
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(args.seed) if shuffle else None,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
    )


def ledger_entries(ledger: Path, run_name: str) -> list[dict[str, Any]]:
    """Linhas do livro-razão referentes a um run (avaliações anteriores do teste)."""
    if not ledger.exists():
        return []
    entries = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [entry for entry in entries if entry.get("run_name") == run_name]


def split_paths(args: argparse.Namespace) -> tuple[int, Path, Path | None]:
    """Resolve fold, arquivo de normalização e definição versionada conforme o protocolo pedido.

    Cada protocolo tem o seu fold padrão e o seu arquivo de normalização: usar o do outro passaria pelo cheque de
    ano e ainda assim estaria ajustado em outro conjunto de eventos. Resolver os três juntos aqui é o que impede a
    combinação errada de chegar ao treino — `check_normalization` então confere pelas chaves dos eventos.
    """
    fold = args.fold if args.fold is not None else (
        SPATIOTEMPORAL_FOLD if args.split_kind == "spatiotemporal" else DEFAULT_FOLD)
    default_stats = (f"spatiotemporal_fold{fold}_train_normalization.json"
                     if args.split_kind == "spatiotemporal" else f"fold{fold}_train_normalization.json")
    normalization_path = args.normalization or args.data_root / "stats" / default_stats
    definition_path = args.split_definition
    if definition_path is None and args.split_kind == "spatiotemporal":
        definition_path = SPLITS_DIR / "spatiotemporal_split.json"
    return fold, normalization_path, definition_path


def resolve_protocol_split(hdf5_root: Path, kind: str, fold: int, definition_path: Path | None) -> SplitSpec:
    """Devolve o `SplitSpec` do protocolo, conferido contra a definição versionada quando ela existe."""
    definition = (json.loads(definition_path.read_text(encoding="utf-8"))
                  if definition_path is not None and definition_path.exists() else None)
    if kind == "spatiotemporal" and definition is None:
        raise SystemExit(
            "O protocolo espaço-temporal exige a definição versionada do split "
            "(`config/splits/spatiotemporal_split.json`). Gere-a com `scripts/make_splits.py`: "
            "sem ela o split pode mudar entre execuções e os runs deixam de ser comparáveis."
        )
    return resolve_split(hdf5_root, kind, fold, definition=definition)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.max_windows is not None:
        kind = "smoke"
    elif args.timing_only:
        kind = "timing"
    elif args.max_events is not None:
        kind = "demo"
    else:
        kind = "full"
    default_report = {
        "smoke": SMOKE_REPORT,
        "timing": TIMING_REPORT,
        "demo": DEMO_REPORT_DIR / f"{run_name}.json",
        "full": RUNS_REPORT_DIR / f"{run_name}.json",
    }[kind]
    report_path = args.report or default_report
    checkpoint = args.data_root / "runs" / run_name / "best.pt"
    # Um run cujo teste já foi olhado nunca é retreinado, nem com --force (regra do teste único).
    full_report = RUNS_REPORT_DIR / f"{run_name}.json"
    tested = ledger_entries(args.ledger, run_name) or (
        full_report.exists() and "test" in json.loads(full_report.read_text(encoding="utf-8"))
    )
    if tested:
        raise SystemExit(f"O run {run_name!r} já teve o teste avaliado; escolha outro --run-name.")
    if (report_path.exists() or checkpoint.exists()) and not args.force:
        raise SystemExit(f"{report_path} ou {checkpoint} já existe; use outro --run-name ou --force.")
    if args.timing_only:
        args.epochs = 1
    check_resources(min_available_ram_gib=0.5, min_disk_free_gib=2)
    determinism = configure_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"
    radius = receptive_radius(args.num_layers)
    if args.halo < radius:
        raise SystemExit(f"halo={args.halo} menor que o raio receptivo do modelo ({radius}).")
    fold, normalization_path, definition_path = split_paths(args)
    args.fold = fold
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if kind == "full" and not HDF5_MANIFEST.exists():
        raise SystemExit(f"Execução completa exige o manifesto de conversão {HDF5_MANIFEST}.")
    hdf5_root = args.data_root / "hdf5"
    common = {
        "t_in": args.t_in,
        "patch": args.patch,
        "halo": args.halo,
        "include_forecast_precip": args.include_forecast_precip,
        "drop_groups": args.drop_groups,
    }
    spec = resolve_protocol_split(hdf5_root, args.split_kind, fold, definition_path)
    # Cheque pelas chaves dos eventos, não pelo ano: é o que pega uma normalização do outro protocolo.
    check_normalization(normalization, spec)
    split = (spec.metadata["years"] if args.split_kind == "temporal"
             else {name: sorted({e.year for e in items}) for name, items in spec.events.items()})
    train_events = sample_events(spec.events["train"], args.max_events)
    val_events = sample_events(spec.events["validation"], args.max_events)
    train_set = WSTSWindowDataset(train_events, normalization, mode="train", seed=args.seed, max_windows=args.max_windows, **common)
    val_set = WSTSWindowDataset(val_events, normalization, mode="eval", max_windows=args.max_windows, **common)
    pos_weight = estimate_pos_weight(train_set, args.pos_weight_samples, args.seed)
    train_loader = _loaders(args, train_set, shuffle=True, batch_size=args.batch_size, device=device)
    val_loader = _loaders(args, val_set, shuffle=False, batch_size=args.eval_batch_size, device=device)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    model = build_model(len(train_set.channel_names), args.t_in, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    history, best_ap, best_epoch, best_validation = [], -1.0, None, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        losses = []
        epoch_started = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch["x"].to(device, non_blocking=True))[:, :, -1:]
            loss = masked_bce_with_logits(logits.float(), batch["y"].to(device), batch["mask"].to(device), pos_weight["pos_weight"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        train_seconds = time.perf_counter() - epoch_started
        train_set.close()
        check_resources(min_available_ram_gib=0.3, min_disk_free_gib=1)
        validation = evaluate(model, val_loader, device, use_amp)
        val_set.close()
        if args.timing_only:
            # Pré-registro: o orçamento de épocas é fixado sem olhar métricas de validação.
            timing = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": "timing",
                "eligible_for_test": False,
                "n_train_windows": len(train_set.windows),
                "n_validation_windows": len(val_set.windows),
                "train_seconds_per_epoch": round(train_seconds, 1),
                "validation_seconds": round(time.perf_counter() - epoch_started - train_seconds, 1),
                "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
                "config": config,
                "provenance": provenance({"normalization": normalization_path, "hdf5_manifest": HDF5_MANIFEST}),
                "note": "métricas de validação calculadas apenas para medir tempo e descartadas sem registro",
            }
            write_json(timing, report_path)
            print(f"Tempo: treino {timing['train_seconds_per_epoch']} s/época, validação {timing['validation_seconds']} s")
            return timing
        summary = validation.model["all"].summary()
        if summary["average_precision"] is None:
            raise SystemExit("Validação sem positivos: AP e threshold indefinidos.")
        ap = summary["average_precision"]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_ap": ap,
            "val_ap_new_vs_t": validation.model["new_vs_t"].summary()["average_precision"],
            "train_seconds": round(train_seconds, 1),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 1),
        })
        print(f"época {epoch}: loss={history[-1]['train_loss']:.4f} AP_val={ap:.4f} ({history[-1]['epoch_seconds']} s)", flush=True)
        if ap > best_ap:
            best_ap, best_epoch, best_validation = ap, epoch, validation
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, checkpoint)

    threshold_bin = best_validation.model["all"].summary()["best_threshold"]["bin"]
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "eligible_for_test": kind == "full",
        "is_demonstration": kind == "demo",
        "demonstration_note": (
            f"Amostra de {args.max_events} eventos por ano, usada só para demonstrar que o fluxo roda ponta a "
            "ponta. Estas métricas NÃO são evidência: o conjunto é pequeno e o teste não é avaliado."
        ) if kind == "demo" else None,
        "run_name": run_name,
        "protocol": {
            "fold": fold,
            "split_kind": args.split_kind,
            "split": split,
            "split_cells": spec.describe() if args.split_kind != "temporal" else None,
            "split_reading": (
                "o teste é lugar novo e período novo ao mesmo tempo; `future_only` (blocos de treino, ano de "
                "teste) isola o deslocamento temporal e `space_only` (blocos de teste, anos de treino) o "
                "espacial, ambos avaliados com este mesmo modelo"
            ) if args.split_kind == "spatiotemporal" else "anos disjuntos; regiões compartilhadas",
            "t_in": args.t_in,
            "target": "fogo ativo binário em t+1",
            "readout": "logits[:, :, -1:] (convenção de leitura)",
            "checkpoint_selection": "maior AP de validação",
            "threshold_selection": "maior F2 de validação (índice de bin no histograma de logits)",
            "ap_definition": f"AP em degraus sobre histograma de logits ({FINE_BINS} bins em {LOGIT_RANGE}); "
            f"IC por bootstrap de eventos com {EVENT_BINS} bins",
            "views": {
                "all": "todos os pixels",
                "crop32": "center_crop_x32 (comparável ao artigo)",
                "new_vs_t": "pixels sem fogo em t",
                "new_vs_window": "pixels sem fogo em t−T_in+1..t",
            },
            "baselines": BASELINES,
            "baseline_notes": "persistência 'última passagem' usa a hora máxima de t na imagem inteira (oficial: no tensor "
            "já recortado); difere só na vista crop32 quando o pixel de hora máxima cai fora do recorte",
            "logit_precision": "logits calculados sob AMP (fp16) e convertidos para fp32; espaçamento ~0,004–0,008 em −5..−10",
            "train_sampling": "1 recorte/janela/época, regra oficial de 10 sorteios, sem augmentation",
            "eval_tiling": f"tiles {args.patch} px com halo {args.halo} px (raio receptivo {radius}); cada pixel avaliado "
            "uma vez; a entrada de cada tile é idêntica à da imagem inteira, mas a saída só coincide com a inferência da "
            "imagem inteira a >= raio receptivo da borda (BatchNorm/LeakyReLU fora da imagem)",
            "channels": train_set.channel_names,
            "drop_groups": args.drop_groups,
            "determinism": determinism,
        },
        "config": config,
        "data_root": str(args.data_root),
        "provenance": provenance({"normalization": normalization_path, "hdf5_manifest": HDF5_MANIFEST,
                                  **({"split_definition": definition_path} if definition_path else {})}),
        "normalization_file": normalization_path.relative_to(args.data_root).as_posix()
        if normalization_path.resolve().is_relative_to(args.data_root.resolve()) else str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "events": {"train": [e.key for e in train_events], "validation": [e.key for e in val_events]},
        "device": str(device),
        "n_train_windows": len(train_set.windows),
        "n_validation_windows": len(val_set.windows),
        "pos_weight": pos_weight,
        "history": history,
        "best_epoch": best_epoch,
        "validation_threshold_bin": threshold_bin,
        "validation": best_validation.summary(threshold_bin, bootstrap=args.bootstrap),
        "checkpoint": checkpoint.relative_to(args.data_root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_per_event_counts": best_validation.save_per_event(checkpoint.parent / "validation_per_event.npz")
        .relative_to(args.data_root).as_posix(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
        "resources": resource_snapshot(),
    }
    write_json(report, report_path)
    print(f"Melhor época {best_epoch} | AP validação {best_ap:.4f} | relatório {report_path}")
    return report


def shift_decomposition(measured: dict[str, Any]) -> dict[str, Any]:
    """Decompõe a perda do teste nos dois deslocamentos, a partir das células já medidas.

    Com AP(bloco, ano), o teste é AP(teste, teste) e os cenários são AP(treino, teste) — só futuro — e
    AP(teste, treino) — só locais novos. Duas diferenças são identificáveis sem nenhuma célula extra:

    - `cost_of_new_places`: só futuro − teste, o que custa mudar de lugar **dado** o ano novo;
    - `cost_of_new_period`: só locais novos − teste, o que custa mudar de período **dado** o lugar novo.

    A interação exigiria a célula sem deslocamento (blocos de treino × anos de treino), que só existe dentro da
    amostra. Quando ela foi medida (`--evaluate-train-cell`), o valor é publicado com o aviso de que o
    referencial é um teto otimista, e não uma estimativa de generalização.
    """

    def ap(cell: str) -> float | None:
        value = measured.get(cell)
        return value["metrics"]["all"]["model"]["average_precision"] if value else None

    test, future_only, space_only, in_sample = ap("test"), ap("future_only"), ap("space_only"), ap("train")
    out: dict[str, Any] = {
        "metric": "average_precision (vista `all`)",
        "average_precision": {"test": test, "future_only": future_only, "space_only": space_only,
                              "train_cell_in_sample": in_sample},
        "cost_of_new_places": round(future_only - test, 6) if None not in (future_only, test) else None,
        "cost_of_new_period": round(space_only - test, 6) if None not in (space_only, test) else None,
        "interaction": None,
        "note": (
            "`cost_of_new_places` é medido no ano de teste e `cost_of_new_period` nos blocos de teste: cada um é "
            "o efeito de um deslocamento na presença do outro. Os três números vêm do mesmo checkpoint e do "
            "mesmo limiar de validação, numa única abertura do teste."
        ),
    }
    if None not in (in_sample, future_only, space_only, test):
        out["interaction"] = round(future_only + space_only - in_sample - test, 6)
        out["interaction_note"] = (
            "(só futuro + só locais novos) − (treino + teste), com a célula de treino como referencial sem "
            "deslocamento. Ela é dentro da amostra, logo um teto otimista: a interação é indicativa, não uma "
            "estimativa de generalização."
        )
    parts = [f"teste AP={test}" if test is not None else "teste indisponível"]
    if out["cost_of_new_places"] is not None:
        parts.append(f"lugar novo custa {out['cost_of_new_places']:+.4f} de AP (só futuro {future_only})")
    if out["cost_of_new_period"] is not None:
        parts.append(f"período novo custa {out['cost_of_new_period']:+.4f} (só locais novos {space_only})")
    out["reading"] = "; ".join(parts)
    return out


def run_test_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Avalia o teste uma vez a partir do checkpoint e do threshold de validação de um run."""
    if args.max_windows is not None:
        raise SystemExit("Smoke (--max-windows) não pode avaliar o teste.")
    if not args.run_name:
        raise SystemExit("--evaluate-test exige --run-name de um treino completo.")
    report_path = args.report or RUNS_REPORT_DIR / f"{args.run_name}.json"
    if not report_path.exists():
        demo = " É um run de demonstração (`--max-events`), que nunca avalia o teste." if (
            DEMO_REPORT_DIR / f"{args.run_name}.json").exists() else ""
        raise SystemExit(f"Não há relatório de treino completo em {report_path}.{demo}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("kind") != "full" or not report.get("eligible_for_test"):
        raise SystemExit(
            f"Só runs completos podem ser avaliados no teste; {args.run_name!r} é do tipo {report.get('kind')!r}."
        )
    previous = ledger_entries(args.ledger, args.run_name)
    previously_evaluated = "test" in report or bool(previous)
    if previously_evaluated and not args.force:
        raise SystemExit(f"O teste do run {args.run_name!r} já foi avaliado ({len(previous)} registro(s) no livro-razão); "
                         "use --force (fica registrado).")
    data_root = Path(report["data_root"])
    checkpoint = data_root / report["checkpoint"]
    if sha256_file(checkpoint) != report["checkpoint_sha256"]:
        raise SystemExit(f"SHA-256 de {checkpoint} diverge do registrado no treino.")
    normalization_path = data_root / report["normalization_file"]
    if sha256_file(normalization_path) != report["normalization_sha256"]:
        raise SystemExit(f"SHA-256 de {normalization_path} diverge do registrado no treino.")

    import torch

    config = report["config"]
    check_resources(min_available_ram_gib=0.5, min_disk_free_gib=1)
    configure_determinism(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"
    split_kind = config.get("split_kind", "temporal")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    definition_path = (Path(config["split_definition"]) if config.get("split_definition")
                       else (SPLITS_DIR / f"{split_kind}_split.json"
                             if split_kind != "temporal" else None))
    spec = resolve_protocol_split(data_root / "hdf5", split_kind, config["fold"], definition_path)
    check_normalization(normalization, spec)
    # Uma única abertura do teste mede **todas** as células de avaliação: o teste (lugar e período novos) e os
    # cenários que isolam um deslocamento cada. Separá-las em execuções diferentes seria olhar o teste mais de
    # uma vez, e a decomposição da perda só vale se as três vierem do mesmo checkpoint e do mesmo limiar.
    cells = ["test"]
    if split_kind == "spatiotemporal":
        cells = list(SPATIOTEMPORAL_EVAL_CELLS)
        if args.evaluate_train_cell:
            cells.append("train")
    model = None
    threshold_bin = report["validation_threshold_bin"]
    evaluated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = evaluated_at[:19].replace(":", "")
    measured: dict[str, Any] = {}
    for cell in cells:
        events = spec.events[cell]
        dataset = WSTSWindowDataset(
            events, normalization, mode="eval", t_in=config["t_in"], patch=config["patch"], halo=config["halo"],
            include_forecast_precip=config["include_forecast_precip"], drop_groups=config["drop_groups"],
        )
        if model is None:
            model = build_model(len(dataset.channel_names), config["t_in"], config, device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
        loader = _loaders(args, dataset, shuffle=False, batch_size=config["eval_batch_size"], device=device)
        accumulator = evaluate(model, loader, device, use_amp)
        measured[cell] = {
            "events": [e.key for e in events],
            "n_windows": len(dataset.windows),
            "years": sorted({e.year for e in events}),
            "metrics": accumulator.summary(threshold_bin, bootstrap=config["bootstrap"]),
            "per_event_counts": accumulator.save_per_event(
                checkpoint.parent / f"{cell}_per_event_{stamp}.npz").relative_to(data_root).as_posix(),
            "in_sample": cell == "train",
        }
        dataset.close()

    report.setdefault("test_history", [])
    if "test" in report:
        report["test_history"].append(report["test"])
    report["test"] = {
        "evaluated_at_utc": evaluated_at,
        "split_kind": split_kind,
        "events": measured["test"]["events"],
        "n_test_windows": measured["test"]["n_windows"],
        "threshold_bin_from_validation": threshold_bin,
        "metrics": measured["test"]["metrics"],
        "per_event_counts": measured["test"]["per_event_counts"],
        "provenance": provenance(),
    }
    if split_kind == "spatiotemporal":
        report["test"]["scenarios"] = {cell: value for cell, value in measured.items() if cell != "test"}
        report["test"]["shift_decomposition"] = shift_decomposition(measured)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps({
            "evaluated_at_utc": evaluated_at,
            "run_name": args.run_name,
            "forced_reevaluation": previously_evaluated,
            "previous_evaluations": len(previous),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "drop_groups": config["drop_groups"],
            "seed": config["seed"],
            "split_kind": split_kind,
            "cells_evaluated": cells,
            "test_ap": report["test"]["metrics"]["all"]["model"]["average_precision"],
        }, ensure_ascii=False) + "\n")
    write_json(report, report_path)
    headline = report["test"]["metrics"]["all"]["model"]["average_precision"]
    print(f"Teste avaliado: AP={headline} | {report_path}")
    for cell, value in (report["test"].get("scenarios") or {}).items():
        label = " (dentro da amostra)" if value["in_sample"] else ""
        cell_ap = value["metrics"]["all"]["model"]["average_precision"]
        print(f"  cenário {cell}: AP={cell_ap} em {value['n_windows']} janelas{label}")
    decomposition = report["test"].get("shift_decomposition")
    if decomposition:
        print("  " + decomposition["reading"])
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return run_test_evaluation(args) if args.evaluate_test else run_training(args)


if __name__ == "__main__":
    main()
