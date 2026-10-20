"""EDA do WildfireSpreadTS em streaming (um evento, um dia por vez), em duas passadas.

Protocolo: o fold é fixado antes da EDA.

- Passada 1 (todos os anos, sem alvo): inventário, shapes, lacunas, janelas, NaN e faixas físicas.
  Treino e validação também recebem histogramas por canal e médias por ano; o treino acumula as
  estatísticas de normalização. O teste recebe só inventário, NaN e faixas, sem o canal de fogo (22).
- Passada 2 (somente anos de treino): prevalência, transições, persistência, distância das ignições
  novas à frente, associação canal × ignição no anel da frente (por evento-dia), cossenos vento/subida
  × frente, rotação da grade (aspect × gradiente), look-ahead do NDVI/EVI2 (diferença-em-diferenças
  alinhada às trocas de composto), defasagem GFS × GRIDMET contra hipóteses de horizonte, horas de detecção.
- Sobreposição espacial validação/teste × treino pelas pegadas das imagens (sem alvo).

O JSON é gravado antes das figuras; figuras e comparação com a tabela oficial não derrubam a EDA.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
import time
import traceback
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import resource_snapshot, write_json
from scripts.core import (
    ACTIVE_FIRE_CHANNEL,
    CHANNEL_NAMES,
    CONTINUOUS_CHANNELS,
    DEFAULT_DATA_ROOT,
    DEFAULT_FOLD,
    DEFAULT_T_IN,
    N_LANDCOVER_CLASSES,
    N_RAW_CHANNELS,
    PIXEL_SIZE_M,
    SOURCE_CRS,
    SPATIOTEMPORAL_FOLD,
    EventRef,
    SplitSpec,
    StreamingChannelStats,
    date_gaps,
    fire_mask,
    grid_geometry,
    iter_event_days,
    list_events,
    normalization_payload,
    parse_dates,
    resolve_split,
    split_normalization_payload,
    spatiotemporal_fold_table,
    split_years,
    to_grid_components,
    transport_vector,
    upslope_grid_from_elevation,
    valid_window_starts,
)
from scripts.env_report import provenance


DEFAULT_REPORT = Path("reports/eda_summary.json")
DEFAULT_FIGURES = Path("reports/figures")
FRONT_RING_PX = 2
DISTANCE_EDGES = (1.0, 2.0, 3.0, 5.0, 10.0)
HISTOGRAM_BINS = 64
# Canais cuja média por evento é plotada no mapa de variáveis (2x3 pequenos múltiplos).
MAP_CHANNELS = ("elevation", "ndvi", "energy_release_component", "wind_speed", "specific_humidity", "temperature_max")
BOOTSTRAP = 1000
SMD_CAVEATS = {
    "ndvi": "contaminado pelo look-ahead do composto VNP13A1",
    "evi2": "contaminado pelo look-ahead do composto VNP13A1",
    **{name: "forçante de ~4 km reamostrada: variância quase nula dentro do anel" for name in (
        "precipitation", "wind_speed", "temperature_min", "temperature_max", "energy_release_component",
        "specific_humidity", "pdsi", "forecast_precipitation", "forecast_wind_speed", "forecast_temperature",
        "forecast_specific_humidity",
    )},
}
MIN_EVENTS_FOR_INFERENCE = 10
COMPOSITE_SPAN_DAYS = 15
PHYSICAL_RANGES: dict[int, tuple[float, float]] = {
    0: (-100.0, 16000.0),
    1: (-100.0, 16000.0),
    2: (-100.0, 16000.0),
    3: (-1e4, 1e4),
    4: (-1e4, 1e4),
    5: (0.0, 500.0),
    6: (0.0, 60.0),
    7: (0.0, 360.0),
    8: (200.0, 340.0),
    9: (200.0, 340.0),
    10: (0.0, 200.0),
    11: (0.0, 0.05),
    12: (0.0, 90.0),
    13: (0.0, 360.0),
    14: (-100.0, 5000.0),
    15: (-10.0, 10.0),
    16: (1.0, float(N_LANDCOVER_CLASSES)),
    17: (0.0, 500.0),
    18: (0.0, 60.0),
    19: (-90.0, 90.0),
    20: (-60.0, 60.0),
    21: (0.0, 0.05),
    22: (0.0, 23.0),
}
# Pesos por dia (t..t+9) dos horizontes F010–F239 do GFS0P25 se o filtro por string for lexicográfico:
# F010–F023 no dia t (14 h), F024–F119 nos dias t+1..t+4 (24 h cada), F120 + F123–F237 (trihorários).
GFS_LONG_HORIZON_WEIGHTS = np.array([14, 24, 24, 24, 24, 8, 8, 8, 8, 8], dtype=float) / 150.0
PALETTE = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#e4e3df",
    "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    "positive": "#2a78d6",
    "negative": "#e34948",
}


# --------------------------------------------------------------------------- utilitários estatísticos


def describe(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {}
    q = np.percentile(array, [0, 25, 50, 75, 100])
    return {"n": int(array.size), "min": q[0], "p25": q[1], "median": q[2], "p75": q[3], "max": q[4], "mean": float(array.mean())}


def bootstrap_events(
    groups: np.ndarray, statistic: Callable[[np.ndarray], Any], n_boot: int = BOOTSTRAP, seed: int = 0
) -> list[float] | list[list[float]] | None:
    """IC 95% reamostrando eventos inteiros (unidade de inferência = evento)."""
    unique = np.unique(groups)
    if len(unique) < 3:
        return None
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = np.concatenate([index[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        value = statistic(sample)
        if value is not None and np.all(np.isfinite(value)):
            values.append(value)
    if not values:
        return None
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5], axis=0)
    return np.stack([low, high]).T.tolist() if np.ndim(low) else [float(low), float(high)]


def event_mean_test(values: Sequence[float], n_perm: int = 5000, seed: int = 0) -> dict[str, Any]:
    """Média entre eventos com teste de sinal e permutação por troca de sinal (H0: média 0)."""
    from scipy import stats

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"n_events": 0, "conclusion": "inconclusivo (sem eventos)"}
    rng = np.random.default_rng(seed)
    observed = float(array.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (flips * np.abs(array)).mean(axis=1)
    p_perm = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    negatives = int(np.sum(array < 0))
    p_sign = float(stats.binomtest(negatives, n, 0.5).pvalue)
    effect = observed / array.std(ddof=1) if n > 1 and array.std(ddof=1) > 0 else None
    boot = rng.choice(array, size=(2000, n), replace=True).mean(axis=1) if n >= 3 else None
    ci = [float(v) for v in np.percentile(boot, [2.5, 97.5])] if boot is not None else None
    if n < MIN_EVENTS_FOR_INFERENCE:
        conclusion = "inconclusivo (poucos eventos)"
    else:
        conclusion = "efeito detectado" if p_perm < 0.05 else "sem efeito detectável"
    return {
        "n_events": n,
        "mean": observed,
        "median": float(np.median(array)),
        "share_negative": negatives / n,
        "p_value_sign_flip": p_perm,
        "p_value_sign_test": p_sign,
        "standardized_effect": effect,
        "ci95_mean_bootstrap_events": ci,
        "conclusion": conclusion,
    }


# --------------------------------------------------------------------------- passada 1: inventário (todos os anos)


def event_inventory(event: EventRef, t_in: int = DEFAULT_T_IN) -> dict[str, Any]:
    dates = parse_dates(event.dates)
    west, south, east, north = footprint_lonlat(event)
    return {
        "key": event.key,
        "year": event.year,
        "n_days": event.n_days,
        "height": event.height,
        "width": event.width,
        "lon": event.lnglat[0],
        "lat": event.lnglat[1],
        "footprint_lonlat": [west, south, east, north],
        "first_month": int(str(dates[0])[5:7]),
        "date_gaps": len(date_gaps(dates)),
        "windows_contiguous": len(valid_window_starts(dates, t_in=t_in, t_out=1)),
        "windows_official": max(event.n_days - t_in, 0),
    }


class EventAggregates:
    """Médias por evento dos canais do mapa e contagem de pixels de fogo.

    A contagem de fogo é o alvo, então só é acumulada fora do ano de teste: no teste o evento entra
    no mapa apenas como localização.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def update(self, event: EventRef, frame: np.ndarray, *, with_target: bool) -> None:
        row = self.rows.setdefault(event.key, {"n_days": 0, "fire_pixels": 0 if with_target else None})
        row["n_days"] += 1
        for name in MAP_CHANNELS:
            with warnings.catch_warnings():
                # Um canal inteiramente NaN neste dia apenas não contribui para a média do evento.
                warnings.simplefilter("ignore", RuntimeWarning)
                value = float(np.nanmean(frame[CHANNEL_NAMES.index(name)]))
            if np.isfinite(value):
                row[name] = row.get(name, 0.0) + value
                row[f"_{name}_n"] = row.get(f"_{name}_n", 0) + 1
        if with_target:
            row["fire_pixels"] += int(fire_mask(frame).sum())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        result = {}
        for key, row in self.rows.items():
            entry = {"n_days": row["n_days"], "fire_pixels": row["fire_pixels"]}
            for name in MAP_CHANNELS:
                count = row.get(f"_{name}_n", 0)
                entry[name] = row[name] / count if count else None
            result[key] = entry
        return result


class ChannelInventory:
    """NaN, faixas físicas, min/max e (opcionalmente) histogramas e médias por canal, sem alvo."""

    def __init__(self, channels: Sequence[int], *, histograms: bool) -> None:
        self.channels = tuple(channels)
        self.histograms = histograms
        n = len(self.channels)
        self.total = 0
        self.nan = np.zeros(n, dtype=np.int64)
        self.out_of_range = np.zeros(n, dtype=np.int64)
        self.min = np.full(n, np.inf)
        self.max = np.full(n, -np.inf)
        self.sum = np.zeros(n)
        self.count = np.zeros(n, dtype=np.int64)
        self.hist = np.zeros((n, HISTOGRAM_BINS), dtype=np.int64)

    def update(self, frame: np.ndarray) -> None:
        pixels = frame.shape[-1] * frame.shape[-2]
        self.total += pixels
        for position, channel in enumerate(self.channels):
            values = frame[channel]
            finite = np.isfinite(values)
            self.nan[position] += pixels - int(finite.sum())
            if not finite.any():
                continue
            valid = values[finite].astype(np.float64)
            self.min[position] = min(self.min[position], float(valid.min()))
            self.max[position] = max(self.max[position], float(valid.max()))
            low, high = PHYSICAL_RANGES[channel]
            self.out_of_range[position] += int(np.count_nonzero((valid < low) | (valid > high)))
            if self.histograms:
                self.sum[position] += valid.sum()
                self.count[position] += valid.size
                self.hist[position] += np.histogram(np.clip(valid, low, high), bins=HISTOGRAM_BINS, range=(low, high))[0]

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for position, channel in enumerate(self.channels):
            entry: dict[str, Any] = {
                "missing_rate": float(self.nan[position] / self.total) if self.total else None,
                "min": float(self.min[position]) if np.isfinite(self.min[position]) else None,
                "max": float(self.max[position]) if np.isfinite(self.max[position]) else None,
                "physical_range": list(PHYSICAL_RANGES[channel]),
                "out_of_range": int(self.out_of_range[position]),
            }
            if self.histograms:
                entry["mean"] = float(self.sum[position] / self.count[position]) if self.count[position] else None
                entry["histogram_counts"] = self.hist[position].tolist()
            result[CHANNEL_NAMES[channel]] = entry
        return result


def footprint_lonlat(event: EventRef) -> tuple[float, float, float, float]:
    """Pegada aproximada do evento em graus (oeste, sul, leste, norte), para desenhar no mapa."""
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", SOURCE_CRS, always_xy=True)
    to_deg = Transformer.from_crs(SOURCE_CRS, "EPSG:4326", always_xy=True)
    x, y = to_utm.transform(*event.lnglat)
    half_w, half_h = event.width * PIXEL_SIZE_M / 2, event.height * PIXEL_SIZE_M / 2
    west, south = to_deg.transform(x - half_w, y - half_h)
    east, north = to_deg.transform(x + half_w, y + half_h)
    return west, south, east, north


def footprint(event: EventRef) -> tuple[float, float, float, float]:
    """Retângulo aproximado da imagem em EPSG:32610 (centro `lnglat`, 375 m de grade por pixel)."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", SOURCE_CRS, always_xy=True).transform(*event.lnglat)
    half_w, half_h = event.width * PIXEL_SIZE_M / 2, event.height * PIXEL_SIZE_M / 2
    return x - half_w, y - half_h, x + half_w, y + half_h


def spatial_overlap(train: Sequence[EventRef], others: Sequence[EventRef]) -> dict[str, Any]:
    """Fração da área de cada evento coberta pela pegada de algum evento de treino (máximo por par)."""
    if not train or not others:
        return {"n_events": len(others)}
    train_boxes = [footprint(e) for e in train]
    fractions = []
    for event in others:
        x0, y0, x1, y1 = footprint(event)
        area = (x1 - x0) * (y1 - y0)
        best = 0.0
        for a0, b0, a1, b1 in train_boxes:
            overlap = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
            best = max(best, overlap / area)
        fractions.append(best)
    array = np.asarray(fractions)
    return {
        "n_events": len(array),
        "share_with_any_overlap": float(np.mean(array > 0)),
        "share_overlap_gt_50pct": float(np.mean(array > 0.5)),
        "max_overlap_fraction": describe(array),
        "note": "pegada = retângulo de H×W pixels de 375 m centrado em lnglat, na grade EPSG:32610",
    }


# --------------------------------------------------------------------------- passada 2: alvo e frente (treino)


def distance_labels() -> list[str]:
    edges = (0.0, *DISTANCE_EDGES)
    labels = [f"({a:g}, {b:g}] px" for a, b in zip(edges[:-1], edges[1:])]
    return labels + [f"> {DISTANCE_EDGES[-1]:g} px", "sem fogo na referência"]


def bin_distances(distances: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(DISTANCE_EDGES) + 2, dtype=np.int64)
    finite = np.isfinite(distances)
    counts[-1] += int(np.count_nonzero(~finite))
    bins = np.digitize(distances[finite], DISTANCE_EDGES, right=True)
    counts[: len(DISTANCE_EDGES) + 1] += np.bincount(bins, minlength=len(DISTANCE_EDGES) + 1)
    return counts


class FrontAccumulator:
    """Alvo, transições, persistência, distâncias, SMD por evento-dia e cossenos no anel da frente."""

    def __init__(self) -> None:
        self.pixels = self.positives_next = self.new_pixels = self.new_in_ring = 0
        self.event_days = self.event_days_with_fire_next = 0
        self.persistence = np.zeros(3, dtype=np.int64)
        self.transitions: dict[str, list[int]] = {"ignition": [], "persistence": [], "extinguishment": []}
        self.distance_counts = {
            "to_fire_t": np.zeros(len(DISTANCE_EDGES) + 2, dtype=np.int64),
            "to_window_union": np.zeros(len(DISTANCE_EDGES) + 2, dtype=np.int64),
        }
        self.distance_km: list[float] = []
        self.ring_pixels = self.ring_ignitions = 0
        self.smd_rows: list[tuple[int, np.ndarray]] = []
        self.cos_rows: list[tuple[int, float, float]] = []

    def update(
        self,
        event_id: int,
        frame_t: np.ndarray,
        frame_next: np.ndarray,
        window_union: np.ndarray,
        gamma_deg: float,
        ground_pixel_m: float,
    ) -> None:
        from scipy import ndimage

        fire_t, fire_next = fire_mask(frame_t).astype(bool), fire_mask(frame_next).astype(bool)
        self.event_days += 1
        self.pixels += fire_next.size
        self.positives_next += int(fire_next.sum())
        self.event_days_with_fire_next += int(fire_next.any())
        self.persistence += [
            np.count_nonzero(fire_t & fire_next),
            np.count_nonzero(fire_t & ~fire_next),
            np.count_nonzero(~fire_t & fire_next),
        ]
        new = fire_next & ~fire_t
        self.new_pixels += int(new.sum())
        if fire_next.any():
            self.transitions["ignition"].append(int(new.sum()))
        if fire_t.any():
            self.transitions["persistence"].append(int(np.count_nonzero(fire_t & fire_next)))
            self.transitions["extinguishment"].append(int(np.count_nonzero(fire_t & ~fire_next)))
        reference = window_union | fire_t
        if new.any():
            if reference.any():
                self.distance_counts["to_window_union"] += bin_distances(ndimage.distance_transform_edt(~reference)[new])
            else:
                self.distance_counts["to_window_union"][-1] += int(new.sum())
        if not fire_t.any():
            self.distance_counts["to_fire_t"][-1] += int(new.sum())
            return
        distance, (near_row, near_col) = ndimage.distance_transform_edt(~fire_t, return_indices=True)
        if new.any():
            self.distance_counts["to_fire_t"] += bin_distances(distance[new])
            self.distance_km += (distance[new] * ground_pixel_m / 1000.0).tolist()
        ring = (~fire_t) & (distance <= FRONT_RING_PX)
        if not ring.any():
            return
        labels = fire_next[ring]
        self.ring_pixels += int(ring.sum())
        self.ring_ignitions += int(labels.sum())
        self.new_in_ring += int(np.count_nonzero(new & ring))
        if labels.all() or not labels.any():
            return
        # SMD por evento-dia (desvio agrupado não ponderado dentro do dia).
        smd = np.full(len(CONTINUOUS_CHANNELS), np.nan)
        for position, channel in enumerate(CONTINUOUS_CHANNELS):
            values = frame_t[channel][ring].astype(np.float64)
            a, b = values[labels & np.isfinite(values)], values[~labels & np.isfinite(values)]
            if a.size >= 2 and b.size >= 2:
                pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
                if pooled > 0:
                    smd[position] = (a.mean() - b.mean()) / pooled
        self.smd_rows.append((event_id, smd))
        # Cossenos: vetor frente→candidato × transporte do vento e × subida (ambos na grade).
        rows, cols = np.nonzero(ring)
        d_row = (rows - near_row[rows, cols]).astype(np.float64)
        d_col = (cols - near_col[rows, cols]).astype(np.float64)
        norm = np.hypot(d_row, d_col)
        east, north = transport_vector(frame_t[6][ring], frame_t[7][ring])
        wind_col, wind_row = to_grid_components(east, north, gamma_deg)
        up_col_full, up_row_full = upslope_grid_from_elevation(frame_t[14], ground_pixel_m)

        def cosine(vc: np.ndarray, vr: np.ndarray) -> np.ndarray:
            size = np.hypot(vc, vr) * norm
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(size > 0, (vc * d_col + vr * d_row) / size, np.nan)

        def difference(cos: np.ndarray) -> float:
            a, b = cos[labels & np.isfinite(cos)], cos[~labels & np.isfinite(cos)]
            return float(a.mean() - b.mean()) if a.size and b.size else np.nan

        self.cos_rows.append((
            event_id,
            difference(cosine(wind_col, wind_row)),
            difference(cosine(up_col_full[ring], up_row_full[ring])),
        ))

    def _smd_summary(self) -> dict[str, Any]:
        if not self.smd_rows:
            return {}
        groups = np.array([g for g, _ in self.smd_rows])
        matrix = np.vstack([row for _, row in self.smd_rows])
        summary = {}
        for position, channel in enumerate(CONTINUOUS_CHANNELS):
            column = matrix[:, position]

            def mean_of_event_means(index: np.ndarray, column: np.ndarray = column) -> float | None:
                g, v = groups[index], column[index]
                keep = np.isfinite(v)
                if not keep.any():
                    return None
                return float(np.mean([v[keep][g[keep] == e].mean() for e in np.unique(g[keep])]))

            finite = np.isfinite(column)
            summary[CHANNEL_NAMES[channel]] = {
                "smd_mean_of_event_means": mean_of_event_means(np.arange(len(groups))),
                "ci95_bootstrap_events": bootstrap_events(groups, mean_of_event_means),
                "n_event_days": int(finite.sum()),
                "share_event_days_with_within_ring_variance": float(finite.mean()) if len(finite) else None,
                "n_events": int(len(np.unique(groups[finite]))),
                "caveat": SMD_CAVEATS.get(CHANNEL_NAMES[channel]),
            }
        return summary

    def _cosine_summary(self) -> dict[str, Any]:
        if not self.cos_rows:
            return {}
        groups = np.array([g for g, _, _ in self.cos_rows])
        result = {}
        for position, name in ((1, "wind_transport"), (2, "upslope_from_elevation")):
            column = np.array([row[position] for row in self.cos_rows])
            per_event = [
                float(np.nanmean(column[groups == e]))
                for e in np.unique(groups)
                if np.isfinite(column[groups == e]).any()
            ]
            result[name] = event_mean_test(per_event)
        return result

    def to_dict(self) -> dict[str, Any]:
        tp, fp, fn = map(int, self.persistence)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        labels = distance_labels()
        return {
            "event_days": self.event_days,
            "event_days_with_fire_next": self.event_days_with_fire_next,
            "pixels": self.pixels,
            "positives_next": self.positives_next,
            "prevalence_next": self.positives_next / self.pixels if self.pixels else None,
            "new_pixels": self.new_pixels,
            "new_fraction_of_positives": self.new_pixels / self.positives_next if self.positives_next else None,
            "persistence_baseline_all_pairs": {
                "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f2": f2,
                "note": "todos os pares consecutivos; não comparável à persistência do STConv (janelas T_in)",
            },
            "transitions_per_event_day": {name: describe(values) for name, values in self.transitions.items()},
            "new_pixel_distance": {
                "definition": "distância euclidiana (px de grade) de cada pixel novo ao pixel ativo mais próximo",
                "to_fire_t": dict(zip(labels, self.distance_counts["to_fire_t"].tolist())),
                "to_window_union": dict(zip(labels, self.distance_counts["to_window_union"].tolist())),
                "window_union_definition": "união das detecções nos T_in−1 dias anteriores a t e em t",
                "km_to_fire_t": describe(self.distance_km),
            },
            "front_ring": {
                "ring_px": FRONT_RING_PX,
                "pixels": self.ring_pixels,
                "ignitions": self.ring_ignitions,
                "ignition_rate": self.ring_ignitions / self.ring_pixels if self.ring_pixels else None,
                "share_of_new_pixels_in_ring": self.new_in_ring / self.new_pixels if self.new_pixels else None,
                "smd_ignite_vs_not_by_event_day": self._smd_summary(),
                "cosine_front_direction_ignite_minus_not": self._cosine_summary(),
            },
        }


class WindAxisCheck:
    """Eixo do vento previsto (canal 19, atan(v/u), anti-horário desde leste) × transporte GRIDMET (canal 7)."""

    def __init__(self, bins: int = 18) -> None:
        self.edges = np.linspace(0.0, 90.0, bins + 1)
        self.counts = np.zeros(bins, dtype=np.int64)
        self.cos2_sum = 0.0
        self.n = 0

    def update(self, frame: np.ndarray) -> None:
        th, forecast = frame[7], frame[19]
        valid = np.isfinite(th) & np.isfinite(forecast)
        if not valid.any():
            return
        transport_math = 90.0 - (th[valid] + 180.0)
        delta = np.abs(((transport_math - forecast[valid]) + 90.0) % 180.0 - 90.0)
        self.counts += np.histogram(delta, bins=self.edges)[0]
        self.cos2_sum += float(np.cos(np.deg2rad(2.0 * delta)).sum())
        self.n += int(valid.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pixels": self.n,
            "mean_cos_2delta": self.cos2_sum / self.n if self.n else None,
            "interpretation": "1 = eixos alinhados; 0 = sem relação; -1 = perpendiculares",
            "histogram_edges_deg": self.edges.tolist(),
            "histogram_counts": self.counts.tolist(),
        }


class AspectGridCheck:
    """Rotação grade × norte verdadeiro: offset (aspect − encosta abaixo na grade) contra −γ por evento."""

    def __init__(self, min_slope_deg: float = 2.0) -> None:
        self.min_slope_deg = min_slope_deg
        self.rows: list[dict[str, Any]] = []

    def update(self, event: EventRef, frame: np.ndarray, gamma_deg: float) -> None:
        slope, aspect = frame[12], frame[13]
        up_col, up_row = upslope_grid_from_elevation(frame[14])
        # Encosta abaixo como azimute da grade (horário desde o norte da grade): leste = −up_col, norte = +up_row.
        downslope_grid = np.degrees(np.arctan2(-up_col, up_row))
        valid = np.isfinite(downslope_grid) & np.isfinite(aspect) & (slope > self.min_slope_deg)
        if valid.sum() < 50:
            return
        mean_vector = np.mean(np.exp(1j * np.deg2rad(aspect[valid] - downslope_grid[valid])))
        self.rows.append({
            "key": event.key,
            "offset_deg": float(np.degrees(np.angle(mean_vector))),
            "resultant_length": float(np.abs(mean_vector)),
            "expected_offset_deg": -gamma_deg,
            "n_pixels": int(valid.sum()),
        })

    def to_dict(self) -> dict[str, Any]:
        if len(self.rows) < 3:
            return {"n_events": len(self.rows)}
        offset = np.array([r["offset_deg"] for r in self.rows])
        expected = np.array([r["expected_offset_deg"] for r in self.rows])
        groups = np.arange(len(self.rows))

        def fit(index: np.ndarray) -> list[float] | None:
            if np.ptp(expected[index]) == 0:
                return None
            slope, intercept = np.polyfit(expected[index], offset[index], 1)
            return [float(slope), float(intercept)]

        fitted = fit(groups)
        if fitted is None:
            return {"n_events": len(self.rows), "error": "γ sem variação entre eventos", "events": self.rows}
        slope, intercept = fitted
        ci = bootstrap_events(groups, fit)
        return {
            "n_events": len(self.rows),
            "regression_offset_on_minus_gamma": {
                "slope": slope,
                "intercept_deg": intercept,
                "ci95_slope": ci[0] if ci else None,
                "ci95_intercept_deg": ci[1] if ci else None,
                "slope_one_inside_ci": bool(ci and ci[0][0] <= 1.0 <= ci[0][1]),
            },
            "corr_offset_vs_minus_gamma": float(np.corrcoef(offset, expected)[0, 1]),
            "residual_deg": describe(offset - expected),
            "abs_offset_without_rotation_deg": describe(np.abs(offset)),
            "events": self.rows,
        }


class CompositeLookAheadCheck:
    """Look-ahead do NDVI/EVI2 por diferença-em-diferenças alinhada às trocas de composto.

    Dias contados pela data (`dia = data − primeira data do evento`), não pelo índice da imagem.
    Troca de composto = dia em que mais da metade dos pixels válidos muda (evita trocas espúrias).
    Um composto observado a partir do dia s cobre [s, s+15]; o composto do dia 0 tem início
    desconhecido e é tratado como terminando no dia 15 (limite superior, conservador para o "pré").
    Para pixels com primeira detecção em d0: `post` = composto vigente em d0−1 cuja janela contém d0;
    `pre` = composto mais recente que termina antes de d0. Controle = pixels a ≤ 10 px dos queimados
    que não queimam em [d0, d0+15]. DiD = Δ(queimados) − Δ(controle), agregado por evento.

    Um DiD negativo não identifica look-ahead sozinho: o composto pós também contém dias antes de d0,
    em que a secagem pré-fogo reduz o índice. Por isso são reportados (i) um DiD placebo (composto pré
    × o anterior a ele, sem dias pós-fogo) e (ii) a dose-resposta do DiD pelo número de dias pós-d0
    contidos no composto pós.
    """

    CHANNELS = {"ndvi": 3, "evi2": 4}
    SWITCH_MIN_FRACTION = 0.5

    def __init__(self) -> None:
        self.groups: dict[str, list[dict[str, float]]] = {name: [] for name in self.CHANNELS}
        self.intervals: list[int] = []
        self.burned_pixels_total = 0
        self.burned_pixels_used = {name: 0 for name in self.CHANNELS}
        self.event_id = 0
        self._reset()

    def _reset(self) -> None:
        self.composites: dict[str, list[tuple[int, np.ndarray]]] = {name: [] for name in self.CHANNELS}
        self.first_fire: np.ndarray | None = None

    def update_day(self, day: int, frame: np.ndarray) -> None:
        fire = fire_mask(frame).astype(bool)
        if self.first_fire is None:
            self.first_fire = np.full(fire.shape, -1, dtype=np.int32)
        self.first_fire[(self.first_fire < 0) & fire] = day
        for name, channel in self.CHANNELS.items():
            values = frame[channel]
            history = self.composites[name]
            if not history:
                history.append((day, values.copy()))
                continue
            previous = history[-1][1]
            both = np.isfinite(values) & np.isfinite(previous)
            if both.any() and np.mean(values[both] != previous[both]) > self.SWITCH_MIN_FRACTION:
                if name == "ndvi":
                    self.intervals.append(day - history[-1][0])
                history.append((day, values.copy()))

    def finish_event(self) -> None:
        from scipy import ndimage

        if self.first_fire is None:
            return
        first = self.first_fire
        self.burned_pixels_total += int(np.count_nonzero(first >= 1))
        for name, history in self.composites.items():
            starts = [start for start, _ in history]
            ends = [starts[0] + COMPOSITE_SPAN_DAYS if k == 0 else start + COMPOSITE_SPAN_DAYS for k, start in enumerate(starts)]
            for d0 in np.unique(first[first >= 1]):
                post = max((k for k, start in enumerate(starts) if start <= d0 - 1), default=None)
                pre_candidates = [k for k, end in enumerate(ends) if end < d0]
                if post is None or not pre_candidates or ends[post] < d0:
                    continue
                pre = pre_candidates[-1]
                if pre == post:
                    continue
                burned = first == d0
                control = ndimage.binary_dilation(burned, iterations=10) & ((first < 0) | (first > d0 + COMPOSITE_SPAN_DAYS))
                did = self._did(history[post][1] - history[pre][1], burned, control)
                if did is None:
                    continue
                placebo = self._did(history[pre][1] - history[pre - 1][1], burned, control) if pre >= 1 else None
                self.groups[name].append({
                    "event": self.event_id,
                    "did": did,
                    "placebo": np.nan if placebo is None else placebo,
                    "post_days_after_d0": float(ends[post] - d0 + 1),
                    "n_burned": float(burned.sum()),
                })
                self.burned_pixels_used[name] += int(burned.sum())
        self.event_id += 1
        self._reset()

    @staticmethod
    def _did(change: np.ndarray, burned: np.ndarray, control: np.ndarray) -> float | None:
        a, b = change[burned & np.isfinite(change)], change[control & np.isfinite(change)]
        return float(a.mean() - b.mean()) if a.size and b.size else None

    def _per_event(self, rows: list[dict[str, float]], key: str) -> list[float]:
        values = []
        for event in sorted({int(r["event"]) for r in rows}):
            chosen = [r for r in rows if r["event"] == event and np.isfinite(r[key])]
            if chosen:
                values.append(float(np.average([r[key] for r in chosen], weights=[r["n_burned"] for r in chosen])))
        return values

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "design": "diferença-em-diferenças por data: composto pós (contém d0) − composto pré (termina antes de d0), "
            "queimados × controle a <= 10 px sem queima em 15 dias; média ponderada por evento; placebo e dose-resposta",
            "unit": "índice x1e4",
            "composite_switch_interval_days": describe(self.intervals),
            "burned_pixels_with_first_detection_after_day0": self.burned_pixels_total,
        }
        for name, rows in self.groups.items():
            groups = np.array([r["event"] for r in rows]) if rows else np.zeros(0)
            dose = np.array([r["post_days_after_d0"] for r in rows])
            did = np.array([r["did"] for r in rows])

            def dose_response(index: np.ndarray) -> float | None:
                from scipy import stats

                if len(index) < 5 or np.ptp(dose[index]) == 0 or np.ptp(did[index]) == 0:
                    return None
                return float(stats.spearmanr(dose[index], did[index]).statistic)

            result[name] = {
                "n_burn_day_groups": len(rows),
                "burned_pixels_used": self.burned_pixels_used[name],
                "coverage_of_burned_pixels": self.burned_pixels_used[name] / self.burned_pixels_total if self.burned_pixels_total else None,
                "did": event_mean_test(self._per_event(rows, "did")),
                "placebo_did": event_mean_test(self._per_event(rows, "placebo")),
                "dose_response_spearman_days_after_d0_vs_did": {
                    "value": dose_response(np.arange(len(rows))) if rows else None,
                    "ci95_bootstrap_events": bootstrap_events(groups, dose_response) if rows else None,
                    "expected_if_look_ahead": "negativo (mais dias pós-fogo no composto → queda maior)",
                },
            }
        result["note"] = ("look-ahead é sustentado por DiD negativo + placebo nulo + dose-resposta negativa; "
                          "nulo não exclui look-ahead (CV-MVC) e nada aqui dispensa o ablation")
        return result


class ForecastHorizonCheck:
    """Perfil corr(GFS_t, GRIDMET_{t+k}) contra hipóteses de horizonte.

    H0: previsão do próprio dia t (perfil = autocorrelação do GRIDMET); H1: média ponderada dos dias
    t..t+9 (filtro lexicográfico F010–F239). Séries = médias espaciais diárias, centradas por evento,
    alinhadas por data. Vento excluído (|média(u, v)| não é comparável à velocidade média do GRIDMET).
    """

    VARIABLES = {"temperature": (20, (8, 9), -273.15), "specific_humidity": (21, (11,), 0.0)}

    def __init__(self, max_lag: int = 5) -> None:
        self.max_lag = max_lag
        self.series: dict[str, list[tuple[int, dict[Any, tuple[float, float]]]]] = {name: [] for name in self.VARIABLES}
        self._current: dict[str, dict[Any, tuple[float, float]]] = {}

    def update_day(self, date: np.datetime64, frame: np.ndarray) -> None:
        for name, (forecast, observed, offset) in self.VARIABLES.items():
            obs = np.nanmean([np.nanmean(frame[c]) for c in observed]) + offset
            self._current.setdefault(name, {})[date] = (float(np.nanmean(frame[forecast])), float(obs))

    def finish_event(self, event_id: int) -> None:
        for name, by_date in self._current.items():
            self.series[name].append((event_id, by_date))
        self._current = {}

    def _rows(self, name: str) -> list[tuple[int, np.ndarray]]:
        rows = []
        horizon = len(GFS_LONG_HORIZON_WEIGHTS)
        span = max(horizon, self.max_lag + 1)
        for event_id, by_date in self.series[name]:
            records = []
            for d in sorted(by_date):
                future = [d + np.timedelta64(k, "D") for k in range(span)]
                if not all(f in by_date and np.all(np.isfinite(by_date[f])) for f in future):
                    continue
                obs = [by_date[f][1] for f in future]
                h1 = float(np.dot(GFS_LONG_HORIZON_WEIGHTS, obs[:horizon]))
                records.append([by_date[d][0], h1] + obs[: self.max_lag + 1])
            if len(records) >= 2:
                matrix = np.asarray(records, dtype=float)
                rows.append((event_id, matrix - np.nanmean(matrix, axis=0)))
        return rows

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"hypotheses": {"H0": "previsão do dia t", "H1": "média ponderada t..t+9 (F010–F239)"}}
        k = self.max_lag + 1
        for name in self.VARIABLES:
            rows = self._rows(name)
            if len(rows) < 3:
                result[name] = {"n_events": len(rows)}
                continue
            groups = np.concatenate([np.full(len(m), e) for e, m in rows])
            stacked = np.vstack([m for _, m in rows])

            def profiles(index: np.ndarray) -> np.ndarray | None:
                data = stacked[index]
                gfs, h1, obs = data[:, 0], data[:, 1], data[:, 2:]
                if np.nanstd(gfs) == 0 or np.nanstd(obs[:, 0]) == 0:
                    return None
                corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
                observed = np.array([corr(gfs, obs[:, j]) for j in range(k)])
                h0 = np.array([corr(obs[:, 0], obs[:, j]) for j in range(k)])
                h1p = np.array([corr(h1, obs[:, j]) for j in range(k)])
                return np.concatenate([observed, h0, h1p, [np.sum((observed - h0) ** 2), np.sum((observed - h1p) ** 2),
                                                          np.nanstd(gfs) / np.nanstd(obs[:, 0])]])

            point = profiles(np.arange(len(groups)))
            if point is None:
                result[name] = {"n_events": len(rows), "error": "série constante"}
                continue
            ci = bootstrap_events(groups, profiles)
            unique = np.unique(groups)
            rng = np.random.default_rng(1)
            closer, argmax = [], []
            for _ in range(500):
                chosen = rng.choice(unique, size=len(unique), replace=True)
                sample = profiles(np.concatenate([np.flatnonzero(groups == g) for g in chosen]))
                if sample is not None:
                    closer.append(sample[3 * k + 1] < sample[3 * k])
                    argmax.append(int(np.argmax(sample[:k])))
            result[name] = {
                "n_events": int(len(unique)),
                "n_days": int(len(groups)),
                "corr_observed_by_lag": point[:k].tolist(),
                "corr_observed_ci95": ci[:k] if ci else None,
                "profile_H0": point[k : 2 * k].tolist(),
                "profile_H1": point[2 * k : 3 * k].tolist(),
                "sq_distance_to_H0": float(point[3 * k]),
                "sq_distance_to_H1": float(point[3 * k + 1]),
                "bootstrap_share_closer_to_H1": float(np.mean(closer)) if closer else None,
                "argmax_lag_observed": int(np.argmax(point[:k])),
                "argmax_lag_bootstrap_distribution": {str(j): float(np.mean(np.array(argmax) == j)) for j in range(k)} if argmax else None,
                "argmax_lag_expected": {"H0": int(np.argmax(point[k : 2 * k])), "H1": int(np.argmax(point[2 * k : 3 * k]))},
                "std_ratio_gfs_over_gridmet": float(point[3 * k + 2]),
            }
        return result


# --------------------------------------------------------------------------- comparações externas


def official_statistics(train_years: Sequence[int], repo: Path = REPO_ROOT / ".wildfirespreadts") -> dict[str, Any] | None:
    """Tabela oficial de médias/desvios (`src/dataloader/utils.py`), carregada sem importar o pacote."""
    path = repo / "src" / "dataloader" / "utils.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("wsts_official_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    means, stds, missing = module.get_means_stds_missing_values(list(train_years))
    return {"means": np.asarray(means).tolist(), "stds": np.asarray(stds).tolist(), "missing": np.asarray(missing).tolist()}


def compare_with_official(ours: dict[str, Any], official: dict[str, Any] | None) -> dict[str, Any] | None:
    if official is None:
        return None
    rows = {}
    for channel in CONTINUOUS_CHANNELS:
        entry = ours["channels"][CHANNEL_NAMES[channel]]
        mean_off, std_off = official["means"][channel], official["stds"][channel]
        rows[CHANNEL_NAMES[channel]] = {
            "mean_ours": entry["mean"],
            "mean_official": mean_off,
            "mean_rel_diff": (entry["mean"] - mean_off) / abs(mean_off) if mean_off else None,
            "std_ours": entry["std"],
            "std_official": std_off,
            "std_rel_diff": (entry["std"] - std_off) / abs(std_off) if std_off else None,
            "missing_ours": entry["missing_rate"],
            "missing_official": official["missing"][channel],
        }
    return rows


def block_split_summary(spec: SplitSpec, inventory_rows: Sequence[dict[str, Any]],
                        role_by_event: dict[str, str]) -> dict[str, Any]:
    """Inventário de um split por blocos, derivado das linhas já coletadas — nenhuma leitura extra de pixel.

    Serve aos dois protocolos por bloco. Cada conjunto (ou célula) é descrito por eventos, imagens, janelas e
    **composição por ano**: é a coluna que diz o que o split separa. No split espacial todos os anos aparecem em
    todos os conjuntos, porque ele isola espaço; no espaço-temporal cada célula tem os anos do seu grupo, porque
    ali as duas separações estão cruzadas.
    """
    by_key = {row["key"]: row for row in inventory_rows}
    per_split: dict[str, Any] = {}
    for name, events in spec.events.items():
        rows = [by_key[e.key] for e in events if e.key in by_key]
        years: dict[str, int] = {}
        for event in events:
            years[str(event.year)] = years.get(str(event.year), 0) + 1
        per_split[name] = {
            "events": len(events),
            "images": sum(r["n_days"] for r in rows),
            "windows_contiguous": sum(r["windows_contiguous"] for r in rows),
            "days_per_event": describe([r["n_days"] for r in rows]),
            "events_by_year": dict(sorted(years.items())),
        }
    meta = spec.metadata
    dropped_rows = [by_key[k] for k in meta.get("dropped", []) if k in by_key]
    summary = {
        "kind": spec.kind,
        "per_split": per_split,
        "role_by_event": role_by_event,
        "dropped": {
            "events": len(meta.get("dropped", [])),
            "images": sum(r["n_days"] for r in dropped_rows),
            "windows_contiguous": sum(r["windows_contiguous"] for r in dropped_rows),
            "reason": "pegada atravessa o corredor de exclusão entre dois conjuntos",
        },
        "parameters": meta.get("parameters"),
        "blocks": meta.get("blocks"),
        "verification": meta.get("verification"),
        "realized_shares": meta.get("realized_shares"),
    }
    for key in ("fold", "years", "cells", "grid", "viability", "evaluation_cells", "train_period"):
        if key in meta:
            summary[key] = meta[key]
    if "sealed_cells" in meta:
        sealed_rows = {name: [by_key[k] for k in keys if k in by_key]
                       for name, keys in meta["sealed_cells"].items()}
        summary["sealed"] = {
            "windows_contiguous": meta["windows"]["sealed"],
            "by_cell": {name: {"events": len(meta["sealed_cells"][name]),
                               "windows_contiguous": sum(r["windows_contiguous"] for r in rows)}
                        for name, rows in sealed_rows.items()},
            "reason": (
                "célula sem papel no protocolo. Selecionar modelo em (blocos de validação × ano de teste) "
                "calibraria o threshold no período de teste; em (blocos de teste × ano de validação), na região "
                "de teste. Ficam fora de treino, seleção e relatório."
            ),
        }
    return summary


def split_comparison(temporal_years: dict[str, list[int]], inventory: dict[str, Any],
                     spatial: SplitSpec, spatiotemporal: SplitSpec, *,
                     temporal_overlap: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compara os três protocolos no que decide a leitura de qualquer métrica.

    Três colunas importam. **Janelas de treino**, porque se um protocolo treinasse com muito menos dados, uma
    queda de desempenho seria volume e não generalização. **Sobreposição de pegada** entre teste e treino, que é
    a propriedade que motivou sair do split por ano. E **anos por conjunto**, que diz o que cada um separa.
    """
    per_year = inventory["per_year"]
    temporal = {
        name: {
            "events": sum(per_year[str(y)]["events"] for y in years),
            "windows_contiguous": sum(per_year[str(y)]["windows_contiguous"] for y in years),
            "years": sorted(years),
        }
        for name, years in temporal_years.items()
    }

    def by_cell(spec: SplitSpec, names: Sequence[str]) -> dict[str, Any]:
        return {
            name: {
                "events": len(spec.events[name]),
                "windows_contiguous": spec.metadata["windows"][name],
                "years": sorted({e.year for e in spec.events[name]}),
            }
            for name in names
        }

    train_reference = temporal["train"]["windows_contiguous"]
    ratios = {
        kind: round(spec.metadata["windows"]["train"] / train_reference, 4) if train_reference else None
        for kind, spec in (("spatial", spatial), ("spatiotemporal", spatiotemporal))
    }
    return {
        "temporal": temporal,
        "spatial": by_cell(spatial, ("train", "validation", "test")),
        "spatiotemporal": by_cell(spatiotemporal,
                                  ("train", "validation", "test", "future_only", "space_only")),
        "train_windows_relative_to_temporal": ratios,
        "spatial_train_windows_relative_to_temporal": ratios["spatial"],
        "test_overlap_with_train": {
            "temporal": ((temporal_overlap or {}).get("test") or {}).get("share_with_any_overlap"),
            "spatial": (spatial.metadata["verification"]["test"] or {}).get("share_with_any_overlap_of_train"),
            "spatiotemporal": (spatiotemporal.metadata["verification"]["test"] or {})
            .get("share_with_any_overlap_of_train"),
        },
        "isolates": {
            "temporal": "tempo (anos disjuntos, regiões compartilhadas)",
            "spatial": "espaço (regiões disjuntas, anos compartilhados)",
            "spatiotemporal": "espaço e tempo (bloco × ano; o teste é lugar novo e período novo)",
        },
        "reading": (
            "O protocolo temporal responde 'generaliza para um ano novo?' e é o único comparável com o benchmark "
            "publicado. O espacial responde 'generaliza para uma região nova?' e serve de diagnóstico da "
            "sobreposição — não é protocolo de avaliação, porque treina com o ano de teste. O espaço-temporal é "
            "a estimativa conservadora, e é dele que sai o número reportado: as células `future_only` e "
            "`space_only` dizem quanto da perda é tempo, quanto é lugar, e quanto é interação entre os dois."
        ),
    }


# --------------------------------------------------------------------------- orquestração


def run_eda(
    hdf5_root: Path,
    fold: int = DEFAULT_FOLD,
    *,
    t_in: int = DEFAULT_T_IN,
    spatial_definition: dict[str, Any] | None = None,
    spatiotemporal_definition: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Executa as duas passadas; devolve (resumo, normalização temporal, normalização espaço-temporal).

    As duas normalizações são acumuladas **no mesmo passe** sobre os dados. A passada 1 já lê todos os dias de
    todos os eventos, então manter um segundo acumulador para a célula de treino do protocolo espaço-temporal
    custa aritmética e nenhuma leitura extra — reler 48 GB por causa de um segundo protocolo seria desnecessário.

    Cada acumulador só recebe o treino do **seu** protocolo. Isso importa porque os dois conjuntos de treino não
    são aninhados: o fold temporal treina em 2018+2020 e o espaço-temporal em 2019+2020, restrito aos blocos de
    treino. O split espacial puro não ganha normalização porque não é protocolo de treino — entra aqui só como
    diagnóstico de sobreposição e contraparte de volume.
    """
    started = time.perf_counter()
    split = split_years(fold)
    role_by_year = {str(y): role for role, ys in split.items() for y in ys}
    events_by_year = {str(y): list_events(hdf5_root, [y]) for y in sorted(int(y) for y in role_by_year)}
    train_events = [e for y in split["train"] for e in events_by_year[str(y)]]

    spatial = resolve_split(hdf5_root, "spatial", definition=spatial_definition)
    spatial_role = {e.key: name for name, items in spatial.events.items() for e in items}
    spatiotemporal = resolve_split(hdf5_root, "spatiotemporal", definition=spatiotemporal_definition)
    spatiotemporal_role = {e.key: name for name, items in spatiotemporal.events.items() for e in items}
    for name, keys in spatiotemporal.metadata.get("sealed_cells", {}).items():
        for key in keys:
            spatiotemporal_role.setdefault(key, "sealed")
    spatiotemporal_train_keys = {e.key for e in spatiotemporal.events["train"]}
    # Conservador: um evento que é avaliação em **qualquer** protocolo não entra em mapa nem em agregado de
    # alvo. São o ano de teste temporal, o teste espacial e as três células de avaliação do espaço-temporal —
    # inclusive `space_only`, que é do período de treino: sem isso, olhar o rótulo de 2020 nos blocos de teste
    # exporia justamente o cenário que deve medir generalização para lugar novo.
    held_out = {e.key for e in events_by_year[str(split["test"][0])]}
    held_out |= {e.key for e in spatial.events["test"]}
    for cell in spatiotemporal.metadata["evaluation_cells"]:
        held_out |= {e.key for e in spatiotemporal.events[cell]}

    # Passada 1 — todos os anos, sem alvo.
    normalization_stats = StreamingChannelStats()
    spatiotemporal_normalization_stats = StreamingChannelStats()
    aggregates = EventAggregates()
    inventory: dict[str, Any] = {"per_year": {}, "events": []}
    channels_by_year: dict[str, Any] = {}
    for year, events in events_by_year.items():
        role = role_by_year[year]
        channels = [c for c in range(N_RAW_CHANNELS) if role != "test" or c != ACTIVE_FIRE_CHANNEL]
        tracker = ChannelInventory(channels, histograms=role != "test")
        rows = [event_inventory(e, t_in=t_in) for e in events]
        inventory["events"] += rows
        for event in events:
            for _, block in iter_event_days(event):
                tracker.update(block[0])
                aggregates.update(event, block[0], with_target=event.key not in held_out)
                if role == "train":
                    normalization_stats.update(block)
                if event.key in spatiotemporal_train_keys:
                    spatiotemporal_normalization_stats.update(block)
        channels_by_year[year] = tracker.to_dict()
        inventory["per_year"][year] = {
            "role": role,
            "events": len(events),
            "images": sum(r["n_days"] for r in rows),
            "days_per_event": describe([r["n_days"] for r in rows]),
            "height": describe([r["height"] for r in rows]),
            "width": describe([r["width"] for r in rows]),
            "events_with_date_gaps": sum(r["date_gaps"] > 0 for r in rows),
            "windows_contiguous": sum(r["windows_contiguous"] for r in rows),
            "windows_official": sum(r["windows_official"] for r in rows),
        }
    all_rows = inventory["events"]
    inventory["min_height"] = min((r["height"] for r in all_rows), default=None)
    inventory["min_width"] = min((r["width"] for r in all_rows), default=None)
    inventory["max_pixels"] = max((r["height"] * r["width"] for r in all_rows), default=None)
    normalization = normalization_payload(normalization_stats, fold, train_events)
    spatiotemporal_normalization = split_normalization_payload(
        spatiotemporal_normalization_stats, spatiotemporal)

    temporal_overlap = {
        name: spatial_overlap(train_events, [e for y in split[name] for e in events_by_year[str(y)]])
        for name in ("validation", "test")
    }

    # Passada 2 — somente treino.
    front = FrontAccumulator()
    wind = WindAxisCheck()
    aspect = AspectGridCheck()
    composite = CompositeLookAheadCheck()
    horizon = ForecastHorizonCheck()
    fire_hours = np.zeros(24, dtype=np.int64)
    gammas = []
    for event_id, event in enumerate(train_events):
        geometry = grid_geometry(event.lnglat)
        gammas.append(geometry["gamma_deg"])
        dates = parse_dates(event.dates)
        previous = None
        history: deque[np.ndarray] = deque(maxlen=t_in)
        for index, block in iter_event_days(event):
            frame = block[0]
            wind.update(frame)
            composite.update_day(int((dates[index] - dates[0]).astype(int)), frame)
            horizon.update_day(dates[index], frame)
            hours = frame[ACTIVE_FIRE_CHANNEL]
            fire_hours += np.bincount(hours[hours > 0].astype(np.int64), minlength=24)[:24]
            if index == 0:
                aspect.update(event, frame, geometry["gamma_deg"])
            consecutive = previous is not None and dates[index] - dates[index - 1] == np.timedelta64(1, "D")
            if not consecutive:
                history.clear()
            else:
                # União das detecções em t−T_in+1..t−1 (o dia t entra como `fire_t` em `front.update`).
                previous_days = list(history)[:-1]
                union = np.any(np.stack(previous_days), axis=0) if previous_days else np.zeros(frame.shape[1:], dtype=bool)
                front.update(event_id, previous, frame, union, geometry["gamma_deg"], geometry["ground_pixel_m"])
            history.append(fire_mask(frame).astype(bool))
            previous = frame
        composite.finish_event()
        horizon.finish_event(event_id)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "fold": fold,
            "train_years": split["train"],
            "validation_years": split["validation"],
            "test_years": split["test"],
            "role_by_year": role_by_year,
            "t_in": t_in,
            "rule": "alvo e diagnósticos só no treino; validação/teste só inventário (teste sem canal de fogo)",
        },
        "inventory": inventory,
        "event_aggregates": aggregates.to_dict(),
        "map_channels": list(MAP_CHANNELS),
        "channels_by_year": channels_by_year,
        "normalization_train": {k: v for k, v in normalization.items() if k != "events"},
        "spatial_split": block_split_summary(spatial, inventory["events"], spatial_role),
        "spatiotemporal_split": {
            **block_split_summary(spatiotemporal, inventory["events"], spatiotemporal_role),
            # Por que a tabela de folds entra no resumo: o fold do protocolo espaço-temporal não é o do
            # temporal, e a única justificativa aceitável para isso é a contagem de células — que fica aqui,
            # ao lado do split, em vez de só na mensagem de um commit.
            "fold_choice": spatiotemporal_fold_table(
                [e for items in events_by_year.values() for e in items], t_in=t_in),
        },
        "split_comparison": split_comparison(split, inventory, spatial, spatiotemporal,
                                             temporal_overlap=temporal_overlap),
        "normalization_train_spatiotemporal": {
            k: v for k, v in spatiotemporal_normalization.items()
            if k not in ("events", "train_event_keys")
        },
        "train_target": front.to_dict(),
        "wind_axis_check": wind.to_dict(),
        "aspect_grid_rotation_check": aspect.to_dict(),
        "composite_look_ahead_check": composite.to_dict(),
        "forecast_horizon_check": horizon.to_dict(),
        "active_fire_hour_histogram_utc_train": fire_hours.tolist(),
        "grid_gamma_deg_train": describe(gammas),
        "spatial_overlap_with_train": temporal_overlap,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "resources": resource_snapshot(),
    }
    return summary, normalization, spatiotemporal_normalization


# --------------------------------------------------------------------------- figuras


def _style(ax: Any, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(PALETTE["surface"])
    ax.set_title(title, loc="left", fontsize=11, color=PALETTE["text"], pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=PALETTE["text_secondary"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["text_secondary"], labelsize=8)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def _signed_colors(values: Sequence[float]) -> list[str]:
    return [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in values]


def make_figures(summary: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Gera as figuras; cada figura falha isoladamente (erro registrado, EDA preservada)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    errors: dict[str, str] = {}
    roles = summary["protocol"]["role_by_year"]
    colors = {"train": PALETTE["series"][0], "validation": PALETTE["series"][1], "test": PALETTE["series"][2]}
    label_style = {"fontsize": 8, "color": PALETTE["text_secondary"]}

    def inventory() -> Any:
        years = sorted(summary["inventory"]["per_year"])
        images = [summary["inventory"]["per_year"][y]["images"] for y in years]
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        bars = ax.barh(years, images, color=[colors[roles[y]] for y in years], height=0.6)
        for bar, year in zip(bars, years):
            events = summary["inventory"]["per_year"][year]["events"]
            text = f"  {int(bar.get_width()):,} imagens · {events} eventos · {roles[year]}".replace(",", ".")
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, text, va="center", **label_style)
        ax.invert_yaxis()
        ax.set_xlim(0, max(max(images), 1) * 1.6)
        _style(ax, "Imagens diárias por ano", f"Fold {summary['protocol']['fold']} — cor indica o papel do ano no split")
        return fig

    def missing() -> Any:
        years = sorted(summary["channels_by_year"])
        names = [n for n in CHANNEL_NAMES if n != "landcover"]
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for offset, year in zip(np.linspace(-0.3, 0.3, len(years)), years):
            channels = summary["channels_by_year"][year]
            rates = [100 * (channels[n]["missing_rate"] or 0) if n in channels else np.nan for n in names]
            ax.scatter(rates, np.arange(len(names)) + offset, s=14, color=colors[roles[year]], label=f"{year} ({roles[year]})", zorder=3)
        ax.set_yticks(np.arange(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlabel("pixels com NaN (%)", **label_style)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        _style(ax, "Dados faltantes por canal e ano", "Fogo ativo: NaN = sem detecção; canal omitido no ano de teste")
        return fig

    def distances() -> Any:
        block = summary["train_target"]["new_pixel_distance"]
        labels = list(block["to_fire_t"])
        y = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for shift, key, name, color in (
            (-0.18, "to_fire_t", "ao fogo em t", PALETTE["series"][0]),
            (0.18, "to_window_union", "à união da janela de entrada", PALETTE["series"][1]),
        ):
            counts = np.array([block[key][label] for label in labels], dtype=float)
            ax.barh(y + shift, 100 * counts / max(counts.sum(), 1), height=0.34, color=color, label=name)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("% dos pixels novos em t+1", **label_style)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        _style(ax, "Distância das ignições novas ao fogo ativo mais próximo", f"Treino {summary['protocol']['train_years']} · 1 px = 375 m de grade")
        return fig

    def smd() -> Any:
        rows = summary["train_target"]["front_ring"]["smd_ignite_vs_not_by_event_day"]
        items = sorted(
            ((k, v) for k, v in rows.items() if v["smd_mean_of_event_means"] is not None),
            key=lambda kv: kv[1]["smd_mean_of_event_means"],
        )
        if not items:
            return None
        values = [v["smd_mean_of_event_means"] for _, v in items]
        y = np.arange(len(items))
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.barh(y, values, color=_signed_colors(values), height=0.6)
        for i, (_, v) in enumerate(items):
            if v["ci95_bootstrap_events"]:
                ax.plot(v["ci95_bootstrap_events"], [i, i], color=PALETTE["text_secondary"], linewidth=1)
        ax.set_yticks(y, [k for k, _ in items])
        ax.axvline(0, color=PALETTE["text_secondary"], linewidth=0.8)
        ax.set_xlabel("SMD (inflama − não inflama), média de eventos; linha = IC 95% por eventos", **label_style)
        _style(ax, f"Canais no anel da frente (≤ {FRONT_RING_PX} px) em t", "Associação descritiva estratificada por evento-dia — não é causalidade")
        return fig

    def wind_axis() -> Any:
        wind = summary["wind_axis_check"]
        if not wind["n_pixels"]:
            return None
        edges = np.asarray(wind["histogram_edges_deg"])
        counts = np.asarray(wind["histogram_counts"], dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        ax.bar(edges[:-1], 100 * counts / counts.sum(), width=np.diff(edges) * 0.9, align="edge", color=PALETTE["series"][0])
        ax.set_xlabel("diferença de eixo |Δ| (graus, módulo 180°)", **label_style)
        ax.set_ylabel("% dos pixels", **label_style)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
        _style(ax, "Vento previsto (canal 19) × GRIDMET (canal 7)", f"média de cos 2Δ = {wind['mean_cos_2delta']:.3f} · 1 = eixos alinhados")
        return fig

    def horizon() -> Any:
        block = summary["forecast_horizon_check"].get("temperature", {})
        if "corr_observed_by_lag" not in block:
            return None
        lags = np.arange(len(block["corr_observed_by_lag"]))
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        for name, key, color in (
            ("observado (GFS)", "corr_observed_by_lag", PALETTE["series"][0]),
            ("H0: dia t", "profile_H0", PALETTE["series"][1]),
            ("H1: média t..t+9", "profile_H1", PALETTE["series"][2]),
        ):
            ax.plot(lags, block[key], marker="o", markersize=4, linewidth=2, color=color, label=name)
        ax.set_xlabel("defasagem k (dias): correlação com GRIDMET em t+k", **label_style)
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
        _style(ax, "Horizonte efetivo da previsão de temperatura (GFS)", "Perfil observado × perfis esperados sob cada hipótese (treino)")
        return fig

    def official() -> Any:
        rows = summary.get("official_statistics_comparison")
        if not rows:
            return None
        items = [(k, 100 * v["mean_rel_diff"]) for k, v in rows.items() if v["mean_rel_diff"] is not None]
        values = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        ax.barh([k for k, _ in items], values, color=_signed_colors(values), height=0.6)
        ax.axvline(0, color=PALETTE["text_secondary"], linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("diferença relativa da média (%)", **label_style)
        _style(ax, "Médias ajustadas no treino × tabela oficial", f"Anos {summary['protocol']['train_years']} · oficial: src/dataloader/utils.py")
        return fig

    def _map_axes(fig, position, extent):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        spec = position if isinstance(position, tuple) else (position,)
        ax = fig.add_subplot(*spec, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#b9b8b2")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#8a8984")
        ax.set_facecolor(PALETTE["surface"])
        return ax

    def _extent(rows, margin=1.5):
        lons = [r["lon"] for r in rows]
        lats = [r["lat"] for r in rows]
        return [min(lons) - margin, max(lons) + margin, min(lats) - margin, max(lats) + margin]

    def event_map() -> Any:
        import cartopy.crs as ccrs
        from matplotlib.patches import Rectangle

        rows = summary["inventory"]["events"]
        if not rows or "lon" not in rows[0]:
            return None
        fig = plt.figure(figsize=(6.4, 5.6))
        ax = _map_axes(fig, 111, _extent(rows))
        for row in rows:
            west, south, east, north = row["footprint_lonlat"]
            role = roles[str(row["year"])]
            ax.add_patch(Rectangle((west, south), east - west, north - south, transform=ccrs.PlateCarree(),
                                   facecolor=colors[role], edgecolor="none", alpha=0.35))
        handles = [plt.Line2D([], [], marker="s", linestyle="", color=colors[r],
                              label=f"{r} ({sum(1 for x in rows if roles[str(x['year'])] == r)} eventos)")
                   for r in ("train", "validation", "test") if any(roles[str(x["year"])] == r for x in rows)]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower left")
        ax.set_title("Pegada dos incêndios por papel no split", loc="left", fontsize=11, color=PALETTE["text"], pad=18)
        ax.text(0, 1.02, "Retângulos sobrepostos entre cores indicam que o split por ano não separa regiões",
                transform=ax.transAxes, **label_style)
        return fig

    # Cores das células do protocolo espaço-temporal: o conjunto herda a cor do seu papel, e os dois cenários
    # herdam a cor do **grupo de blocos** de onde vêm, hachurados, porque é isso que os distingue do conjunto
    # homônimo — `future_only` são blocos de treino noutro ano, `space_only` são blocos de teste noutro ano.
    cell_style = {
        "train": (colors["train"], None),
        "validation": (colors["validation"], None),
        "test": (colors["test"], None),
        "future_only": (colors["train"], "///"),
        "space_only": (colors["test"], "\\\\\\"),
        "sealed": (PALETTE["grid"], None),
        "dropped": (PALETTE["text_secondary"], None),
    }
    cell_label = {
        "train": "treino",
        "validation": "validação",
        "test": "teste",
        "future_only": "só futuro",
        "space_only": "só locais novos",
        "sealed": "selada",
        "dropped": "descartada (corredor)",
    }
    group_label = {"train": "treino", "validation": "validação", "test": "teste"}

    def split_maps() -> Any:
        """Os dois protocolos lado a lado. A comparação é o argumento: à esquerda as cores se misturam."""
        import cartopy.crs as ccrs
        from matplotlib.patches import Patch, Rectangle

        rows = summary["inventory"]["events"]
        spatiotemporal = summary.get("spatiotemporal_split") or {}
        role_by_event = spatiotemporal.get("role_by_event") or {}
        if not rows or "lon" not in rows[0] or not role_by_event:
            return None
        temporal_overlap = (summary.get("split_comparison") or {}).get("test_overlap_with_train") or {}
        st_overlap = (spatiotemporal.get("verification") or {}).get("test") or {}
        extent = _extent(rows)
        fig = plt.figure(figsize=(10.6, 5.4))
        years = spatiotemporal.get("years") or {}
        test_years = ", ".join(str(year) for year in years.get("test", []))
        share = temporal_overlap.get("temporal")
        panels = (
            ("Protocolo temporal (fold 2, por ano)",
             lambda row: roles[str(row["year"])],
             ("train", "validation", "test"),
             (f"{share:.0%} dos eventos de teste caem sobre pegada de treino" if share is not None
              else "o teste repete as regiões do treino")),
            (f"Protocolo espaço-temporal (fold {spatiotemporal.get('fold')}, bloco × ano)",
             lambda row: role_by_event.get(row["key"], "dropped"),
             ("train", "validation", "test", "future_only", "space_only", "sealed", "dropped"),
             f"teste = {test_years} em blocos sem treino"
             + (f"; folga mínima {st_overlap.get('min_gap_to_train_km')} km" if st_overlap else "")),
        )
        for position, (title, role_of, order, subtitle) in enumerate(panels, start=1):
            ax = _map_axes(fig, (1, 2, position), extent)
            for row in rows:
                role = role_of(row)
                face, hatch = cell_style.get(role, (PALETTE["text_secondary"], None))
                west, south, east, north = row["footprint_lonlat"]
                ax.add_patch(Rectangle(
                    (west, south), east - west, north - south, transform=ccrs.PlateCarree(),
                    facecolor=face, edgecolor=face if hatch else "none", hatch=hatch, linewidth=0.3,
                    alpha=0.18 if role in ("dropped", "sealed") else 0.40))
            handles = []
            for role in order:
                count = sum(1 for row in rows if role_of(row) == role)
                if not count:
                    continue
                face, hatch = cell_style[role]
                # A hachura precisa aparecer na legenda: `treino` e `só futuro` têm a mesma cor de propósito
                # (são os mesmos blocos) e só a hachura os distingue.
                handles.append(Patch(facecolor=face, edgecolor=face, hatch=hatch, alpha=0.6,
                                     label=f"{cell_label[role]} ({count})"))
            ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower left")
            ax.set_title(title, loc="left", fontsize=10.5, color=PALETTE["text"], pad=16)
            ax.text(0, 1.02, subtitle, transform=ax.transAxes, **label_style)
        fig.suptitle("O que cada protocolo separa", x=0.01, ha="left", fontsize=11.5, color=PALETTE["text"])
        shares = {kind: temporal_overlap.get(kind) for kind in ("temporal", "spatiotemporal")}
        if all(value is not None for value in shares.values()):
            fig.text(0.01, 0.015,
                     f"Sobreposição de pegada entre teste e treino: {shares['temporal']:.0%} no temporal, "
                     f"{shares['spatiotemporal']:.0%} no espaço-temporal (verificada, não suposta)",
                     **label_style)
        return fig

    def spatial_blocks_map() -> Any:
        """A grade de blocos e o conjunto de cada um — mostra que a atribuição é por região, não por evento."""
        import cartopy.crs as ccrs

        rows = summary["inventory"]["events"]
        spatial = summary.get("spatial_split") or {}
        role_by_event = spatial.get("role_by_event") or {}
        split_by_block = (spatial.get("blocks") or {}).get("split_by_block") or {}
        if not rows or "lon" not in rows[0] or not split_by_block:
            return None
        fig = plt.figure(figsize=(6.6, 5.8))
        ax = _map_axes(fig, 111, _extent(rows))
        for row in rows:
            role = role_by_event.get(row["key"], "dropped")
            ax.plot(row["lon"], row["lat"], marker="o", markersize=3.2, linestyle="",
                    color=colors.get(role, PALETTE["text_secondary"]),
                    alpha=0.35 if role == "dropped" else 0.9, transform=ccrs.PlateCarree())
        ax.set_title("Blocos espaciais e o grupo de cada um", loc="left", fontsize=11,
                     color=PALETTE["text"], pad=18)
        parameters = spatial.get("parameters") or {}
        ax.text(0, 1.02,
                f"blocos de {parameters.get('block_m', 0) / 1000:.0f} km, corredor de exclusão de "
                f"{parameters.get('buffer_m', 0) / 1000:.0f} km; pontos são centros de evento",
                transform=ax.transAxes, **label_style)
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors.get(r, PALETTE["text_secondary"]),
                              label=f"{r} ({sum(1 for v in split_by_block.values() if v == r)} blocos)"
                              if r != "dropped" else "descartado (corredor)")
                   for r in ("train", "validation", "test", "dropped")]
        ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower left")
        return fig

    def spatiotemporal_grid() -> Any:
        """A grade 3×3 do protocolo: é onde se lê, de uma vez, o que é treino, o que é cenário e o que é selado."""
        from matplotlib.patches import Rectangle

        spatiotemporal = summary.get("spatiotemporal_split") or {}
        grid = spatiotemporal.get("grid") or {}
        cells = spatiotemporal.get("cells") or {}
        years = spatiotemporal.get("years") or {}
        if not grid or not cells:
            return None
        role_at = {(spec["blocks"], year_group): name
                   for name, spec in cells.items()
                   for year_group in ("train", "validation", "test")
                   if sorted(years.get(year_group, [])) == spec["years"]}
        groups = ("train", "validation", "test")
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for col, year_group in enumerate(groups):
            for row_index, block_group in enumerate(groups):
                row = 2 - row_index
                counts = grid[block_group][year_group]
                role = role_at.get((block_group, year_group), "sealed")
                face, hatch = cell_style[role]
                ax.add_patch(Rectangle((col + 0.03, row + 0.05), 0.94, 0.90, facecolor=face, hatch=hatch,
                                       edgecolor=PALETTE["surface"], linewidth=2,
                                       alpha=0.22 if role == "sealed" else 0.45))
                ax.text(col + 0.5, row + 0.66, cell_label[role], ha="center", va="center", fontsize=9.5,
                        color=PALETTE["text"])
                ax.text(col + 0.5, row + 0.44, f"{counts['windows']:,} janelas".replace(",", "."),
                        ha="center", va="center", **label_style)
                ax.text(col + 0.5, row + 0.27, f"{counts['events']} eventos",
                        ha="center", va="center", **label_style)
        ax.set_xticks([c + 0.5 for c in range(3)])
        ax.set_xticklabels([f"anos de {group_label[name]}\n{', '.join(str(y) for y in years.get(name, []))}"
                            for name in groups], fontsize=9, color=PALETTE["text_secondary"])
        ax.set_yticks([2.5, 1.5, 0.5])
        ax.set_yticklabels([f"blocos de\n{group_label[name]}" for name in groups], fontsize=9,
                           color=PALETTE["text_secondary"])
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        dropped = spatiotemporal.get("dropped") or {}
        ax.set_title(f"Protocolo espaço-temporal: as nove células (fold {spatiotemporal.get('fold')})",
                     loc="left", fontsize=11, color=PALETTE["text"], pad=18)
        ax.text(0, 1.02,
                f"fora da grade: {dropped.get('events', 0)} eventos "
                f"({dropped.get('windows_contiguous', 0):,} janelas) no corredor de exclusão".replace(",", "."),
                transform=ax.transAxes, **label_style)
        return fig

    def split_windows() -> Any:
        """Volume de treino dos protocolos: é o que impede ler uma queda de desempenho como falta de dados."""
        comparison = summary.get("split_comparison") or {}
        if not comparison or "spatiotemporal" not in comparison:
            return None
        names = ("train", "validation", "test")
        kinds = (("temporal", "temporal (por ano)"), ("spatial", "espacial (por bloco)"),
                 ("spatiotemporal", "espaço-temporal (bloco × ano)"))
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        width = 0.27
        positions = np.arange(len(names))
        for offset, (kind, rotulo) in enumerate(kinds):
            values = [comparison[kind][name]["windows_contiguous"] for name in names]
            bars = ax.bar(positions + (offset - 1) * width, values, width,
                          color=PALETTE["series"][offset], edgecolor=PALETTE["surface"], linewidth=0.8,
                          label=rotulo)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}".replace(",", "."),
                        ha="center", va="bottom", fontsize=7, color=PALETTE["text_secondary"])
        cells = comparison["spatiotemporal"]
        ax.set_xticks(positions)
        ax.set_xticklabels(["treino", "validação", "teste"])
        ax.set_ylabel("janelas contíguas", **label_style)
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
        ratios = comparison.get("train_windows_relative_to_temporal") or {}
        def thousands(value: int) -> str:
            return f"{value:,}".replace(",", ".")

        subtitle = (f"treino em {ratios.get('spatiotemporal', 0):.0%} do fold temporal; fora do eixo: só futuro "
                    f"{thousands(cells['future_only']['windows_contiguous'])}, só locais novos "
                    f"{thousands(cells['space_only']['windows_contiguous'])}")
        _style(ax, "Janelas por conjunto nos três protocolos", subtitle)
        return fig

    def fire_map() -> Any:
        import cartopy.crs as ccrs

        rows = [r for r in summary["inventory"]["events"] if "lon" in r]
        aggregates = summary.get("event_aggregates", {})
        burning = [(r, aggregates.get(r["key"], {}).get("fire_pixels")) for r in rows]
        burning = [(r, n) for r, n in burning if n]
        if not burning:
            return None
        # Layout `constrained` (e não o `tight_layout` global): a barra de cor rouba espaço de um eixo de
        # aspecto travado, e só o motor restrito reconcilia os dois sem espremer o mapa a um canto.
        fig = plt.figure(figsize=(7.2, 5.15), layout="constrained")
        ax = _map_axes(fig, 111, _extent(rows))
        counts = np.array([n for _, n in burning], dtype=float)
        scatter = ax.scatter([r["lon"] for r, _ in burning], [r["lat"] for r, _ in burning],
                             s=12 + 180 * counts / counts.max(), c=counts, cmap="YlOrRd",
                             norm=__import__("matplotlib").colors.LogNorm(), alpha=0.75,
                             edgecolors="#52514e", linewidths=0.3, transform=ccrs.PlateCarree())
        bar = fig.colorbar(scatter, ax=ax, shrink=0.88, pad=0.02, aspect=28)
        bar.set_label("pixels de fogo no evento (escala log)", fontsize=8, color=PALETTE["text_secondary"])
        bar.ax.tick_params(labelsize=7, colors=PALETTE["text_secondary"])
        ax.set_title("Atividade de fogo por evento", loc="left", fontsize=11, color=PALETTE["text"], pad=30)
        ax.text(0, 1.02,
                "\n".join(textwrap.wrap(
                    "A contagem de fogo é o alvo, então o mapa exclui toda célula de avaliação de qualquer "
                    "protocolo: o ano de teste, o teste espacial e as três células do espaço-temporal", 92)),
                transform=ax.transAxes, va="bottom", **label_style)
        return fig

    def variable_maps() -> Any:
        import cartopy.crs as ccrs

        rows = [r for r in summary["inventory"]["events"] if "lon" in r]
        aggregates = summary.get("event_aggregates", {})
        channels = summary.get("map_channels", [])
        if not rows or not channels:
            return None
        extent = _extent(rows)
        fig = plt.figure(figsize=(9.6, 6.0))
        for position, name in enumerate(channels[:6], start=1):
            values = [(r, aggregates.get(r["key"], {}).get(name)) for r in rows]
            values = [(r, v) for r, v in values if v is not None and np.isfinite(v)]
            ax = _map_axes(fig, (2, 3, position), extent)
            if values:
                data = np.array([v for _, v in values], dtype=float)
                scatter = ax.scatter([r["lon"] for r, _ in values], [r["lat"] for r, _ in values],
                                     s=14, c=data, cmap="viridis", alpha=0.85, edgecolors="none",
                                     transform=ccrs.PlateCarree())
                bar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
                bar.ax.tick_params(labelsize=6, colors=PALETTE["text_secondary"])
            ax.set_title(name, loc="left", fontsize=9, color=PALETTE["text"])
        fig.suptitle("Média por evento das forçantes, no espaço", x=0.01, ha="left", fontsize=11, color=PALETTE["text"])
        return fig

    def channel_histograms() -> Any:
        years = sorted(summary["channels_by_year"])
        names = [n for n in CHANNEL_NAMES if n not in ("landcover", "active_fire")]
        available = [n for n in names if any("histogram_counts" in summary["channels_by_year"][y].get(n, {})
                                             for y in years)]
        if not available:
            return None
        columns = 4
        rows_n = int(np.ceil(len(available) / columns))
        fig, axes = plt.subplots(rows_n, columns, figsize=(10.5, 2.0 * rows_n))
        for ax, name in zip(axes.ravel(), available):
            for year in years:
                entry = summary["channels_by_year"][year].get(name, {})
                counts = entry.get("histogram_counts")
                if not counts:
                    continue
                low, high = entry["physical_range"]
                edges = np.linspace(low, high, len(counts) + 1)
                total = max(sum(counts), 1)
                ax.plot(edges[:-1], np.array(counts) / total, linewidth=1.2,
                        color=colors[roles[year]], alpha=0.85, label=year)
            ax.set_title(name, fontsize=8, loc="left", color=PALETTE["text"])
            ax.tick_params(labelsize=6, colors=PALETTE["text_secondary"])
            ax.set_facecolor(PALETTE["surface"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes.ravel()[len(available):]:
            ax.set_visible(False)
        axes.ravel()[0].legend(fontsize=6, frameon=False)
        fig.suptitle("Distribuição de cada canal por ano (densidade)", x=0.01, ha="left",
                     fontsize=11, color=PALETTE["text"])
        return fig

    def seasonality() -> Any:
        rows = [r for r in summary["inventory"]["events"] if "first_month" in r]
        if not rows:
            return None
        years = sorted({str(r["year"]) for r in rows})
        months = np.arange(1, 13)
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        bottom = np.zeros(12)
        for year in years:
            counts = np.array([sum(1 for r in rows if str(r["year"]) == year and r["first_month"] == m)
                               for m in months], dtype=float)
            ax.bar(months, counts, bottom=bottom, color=colors[roles[year]], edgecolor=PALETTE["surface"],
                   linewidth=0.8, label=f"{year} ({roles[year]})")
            bottom += counts
        ax.set_xticks(months)
        ax.set_xlabel("mês do primeiro dia do evento", **label_style)
        ax.set_ylabel("eventos", **label_style)
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
        _style(ax, "Sazonalidade dos incêndios", "Empilhado por ano; a cor indica o papel no split")
        return fig

    def event_sizes() -> Any:
        rows = summary["inventory"]["events"]
        if not rows:
            return None
        years = sorted({str(r["year"]) for r in rows})
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
        for ax, key, title, unit in (
            (axes[0], "n_days", "Duração do evento", "dias"),
            (axes[1], None, "Área da imagem", "milhares de pixels"),
        ):
            data = [[(r["height"] * r["width"] / 1000 if key is None else r[key])
                     for r in rows if str(r["year"]) == year] for year in years]
            parts = ax.boxplot(data, labels=years, patch_artist=True, widths=0.6,
                               medianprops={"color": PALETTE["text"], "linewidth": 1.2},
                               flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
            for patch, year in zip(parts["boxes"], years):
                patch.set_facecolor(colors[roles[year]])
                patch.set_alpha(0.7)
                patch.set_edgecolor(PALETTE["text_secondary"])
            ax.set_ylabel(unit, **label_style)
            _style(ax, title)
            ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
        return fig

    builders = {
        "wsts_mapa_splits": split_maps,
        "wsts_grade_espacotemporal": spatiotemporal_grid,
        "wsts_mapa_blocos_espaciais": spatial_blocks_map,
        "wsts_janelas_por_split": split_windows,
        "wsts_mapa_eventos": event_map,
        "wsts_mapa_atividade_fogo": fire_map,
        "wsts_mapa_variaveis": variable_maps,
        "wsts_histogramas_canais": channel_histograms,
        "wsts_sazonalidade": seasonality,
        "wsts_tamanho_eventos": event_sizes,
        "wsts_inventario_por_ano": inventory,
        "wsts_faltantes_por_canal": missing,
        "wsts_distancia_ignicoes_frente": distances,
        "wsts_smd_anel_frente": smd,
        "wsts_convencao_vento": wind_axis,
        "wsts_horizonte_previsao": horizon,
        "wsts_estatisticas_vs_oficial": official,
    }
    for name, builder in builders.items():
        try:
            fig = builder()
            if fig is None:
                continue
            fig.patch.set_facecolor(PALETTE["surface"])
            if fig.get_layout_engine() is None:  # a figura que já escolheu um motor fica com o dela
                fig.tight_layout()
            path = out_dir / f"{name}.png"
            fig.savefig(path, dpi=130)
            written[name] = path.as_posix()
        except Exception:  # noqa: BLE001 - uma figura não pode derrubar a EDA
            errors[name] = traceback.format_exc(limit=3)
        finally:
            plt.close("all")
    return {"written": written, "errors": errors}


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    return resolved.relative_to(REPO_ROOT).as_posix() if resolved.is_relative_to(REPO_ROOT) else resolved.as_posix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA em streaming do WildfireSpreadTS (protocolo por fold).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument("--t-in", type=int, default=DEFAULT_T_IN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--conversion-manifest", type=Path, default=Path("reports/hdf5_manifest.json"))
    parser.add_argument("--spatial-split", type=Path, default=Path("config/splits/spatial_split.json"),
                        help="Definição versionada do split espacial; se existir, o recomputado é conferido "
                             "contra ela e qualquer divergência é erro.")
    parser.add_argument("--spatiotemporal-split", type=Path,
                        default=Path("config/splits/spatiotemporal_split.json"),
                        help="Definição versionada do split espaço-temporal, conferida do mesmo modo.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    stats_path = args.data_root / "stats" / f"fold{args.fold}_train_normalization.json"

    def read_definition(path: Path) -> dict[str, Any] | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    spatiotemporal_definition = read_definition(args.spatiotemporal_split)
    spatiotemporal_fold = (spatiotemporal_definition or {}).get("fold", SPATIOTEMPORAL_FOLD)
    st_stats_path = (args.data_root / "stats"
                     / f"spatiotemporal_fold{spatiotemporal_fold}_train_normalization.json")
    summary, normalization, spatiotemporal_normalization = run_eda(
        args.data_root / "hdf5", args.fold, t_in=args.t_in,
        spatial_definition=read_definition(args.spatial_split),
        spatiotemporal_definition=spatiotemporal_definition)
    write_json(normalization, stats_path)
    write_json(spatiotemporal_normalization, st_stats_path)
    summary["normalization_file"] = _relative(stats_path)
    summary["normalization_file_spatiotemporal"] = _relative(st_stats_path)
    if args.conversion_manifest.exists():
        manifest = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
        summary["lost_detections_hhmm_lt_100"] = manifest.get("lost_detections_hhmm_lt_100")
    summary["provenance"] = provenance({
        "normalization": stats_path,
        "normalization_spatiotemporal": st_stats_path,
        "hdf5_manifest": args.conversion_manifest,
        "spatial_split": args.spatial_split,
        "spatiotemporal_split": args.spatiotemporal_split,
    })
    try:
        official = official_statistics(summary["protocol"]["train_years"])
        summary["official_statistics_comparison"] = compare_with_official(normalization, official)
    except Exception:  # noqa: BLE001 - código de terceiros não pode derrubar a EDA
        summary["official_statistics_comparison"] = None
        summary["official_statistics_error"] = traceback.format_exc(limit=3)
    write_json(summary, args.report)
    figures = make_figures(summary, args.figures)
    summary["figures"] = {name: _relative(Path(path)) for name, path in figures["written"].items()}
    summary["figure_errors"] = figures["errors"]
    write_json(summary, args.report)
    comparison = summary["split_comparison"]
    print(f"Prevalência t+1 (treino temporal): {summary['train_target']['prevalence_next']}")
    ratios = comparison["train_windows_relative_to_temporal"]
    print(f"Janelas de treino — temporal {comparison['temporal']['train']['windows_contiguous']} | "
          f"espacial {comparison['spatial']['train']['windows_contiguous']} "
          f"({ratios['spatial']:.0%}) | espaço-temporal "
          f"{comparison['spatiotemporal']['train']['windows_contiguous']} ({ratios['spatiotemporal']:.0%})")
    cells = comparison["spatiotemporal"]
    print("Cenários (janelas): teste (lugar e período novos) "
          f"{cells['test']['windows_contiguous']} | só futuro {cells['future_only']['windows_contiguous']} | "
          f"só locais novos {cells['space_only']['windows_contiguous']}")
    print(f"Figuras: {len(summary['figures'])} | erros: {list(summary['figure_errors'])}")
    print(f"Relatório: {args.report}")
    return summary


if __name__ == "__main__":
    main()
