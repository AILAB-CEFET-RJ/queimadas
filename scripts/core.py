"""Núcleo reutilizável da frente WildfireSpreadTS (WSTS).

Contrato de canais, folds oficiais, leitura lazy de HDF5 por evento, janelas diárias
contíguas, estatísticas de normalização em streaming (somente anos de treino),
transformações de entrada e convenções vetoriais usadas nos diagnósticos de EDP.

Fontes da semântica dos canais: ``FireSpreadDataset.map_channel_index_to_features``
(repositório oficial) e ``DataPreparation/satellites/FirePred.py`` do repositório de
criação ``SebastianGer/WildfireSpreadTSCreateDataset`` @ ``f6c5167``. Limitações
temporais conhecidas de cada fonte estão documentadas em ``README.md``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

import numpy as np

if TYPE_CHECKING:
    import h5py


DEFAULT_DATA_ROOT = Path(os.environ.get("WSTS_DATA_ROOT", "dados_geoespaciais/wildfirespreadts"))
YEARS = (2018, 2019, 2020, 2021)
N_RAW_CHANNELS = 23
LANDCOVER_CHANNEL = 16
N_LANDCOVER_CLASSES = 17
ACTIVE_FIRE_CHANNEL = 22
FORECAST_PRECIP_CHANNEL = 17
DEFAULT_FOLD = 2
DEFAULT_T_IN = 5
SOURCE_CRS = "EPSG:32610"
PIXEL_SIZE_M = 375.0
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Ordem idêntica ao oficial: data_fold_id -> (treino, treino, validação, teste).
FOLDS: tuple[tuple[int, int, int, int], ...] = (
    (2018, 2019, 2020, 2021),
    (2018, 2019, 2021, 2020),
    (2018, 2020, 2019, 2021),
    (2018, 2020, 2021, 2019),
    (2018, 2021, 2019, 2020),
    (2018, 2021, 2020, 2019),
    (2019, 2020, 2018, 2021),
    (2019, 2020, 2021, 2018),
    (2019, 2021, 2018, 2020),
    (2019, 2021, 2020, 2018),
    (2020, 2021, 2018, 2019),
    (2020, 2021, 2019, 2018),
)


def _channel(index: int, name: str, source: str, unit: str, kind: str, missing_group: str | None) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "source": source,
        "unit": unit,
        "kind": kind,
        "missing_group": missing_group,
    }


GFS_SOURCE = "GFS0P25 rodada 00Z(t); filtro por string no system:index (provável média F010-F239)"
RAW_CHANNELS: list[dict[str, Any]] = [
    _channel(0, "viirs_m11", "VNP09GA (mediana diária)", "reflectance x1e4", "continuous", "viirs_reflectance"),
    _channel(1, "viirs_i2", "VNP09GA (mediana diária)", "reflectance x1e4", "continuous", "viirs_reflectance"),
    _channel(2, "viirs_i1", "VNP09GA (mediana diária)", "reflectance x1e4", "continuous", "viirs_reflectance"),
    _channel(
        3, "ndvi", "VNP13A1 composto 16 dias iniciado em [t-7, t] (look-ahead até t+15)",
        "index x1e4", "continuous", "vegetation_index",
    ),
    _channel(
        4, "evi2", "VNP13A1 composto 16 dias iniciado em [t-7, t] (look-ahead até t+15)",
        "index x1e4", "continuous", "vegetation_index",
    ),
    _channel(5, "precipitation", "GRIDMET pr", "mm/day", "continuous", "gridmet"),
    _channel(6, "wind_speed", "GRIDMET vs", "m/s", "continuous", "gridmet"),
    _channel(
        7, "wind_direction", "GRIDMET th",
        "deg clockwise from TRUE north, direction wind comes FROM", "angle", "gridmet",
    ),
    _channel(8, "temperature_min", "GRIDMET tmmn", "K", "continuous", "gridmet"),
    _channel(9, "temperature_max", "GRIDMET tmmx", "K", "continuous", "gridmet"),
    _channel(10, "energy_release_component", "GRIDMET erc", "NFDRS index", "continuous", "gridmet"),
    _channel(11, "specific_humidity", "GRIDMET sph", "kg/kg", "continuous", "gridmet"),
    _channel(12, "slope", "SRTM ee.Terrain.slope", "deg", "continuous", "topography"),
    _channel(
        13, "aspect", "SRTM ee.Terrain.aspect",
        "deg clockwise from TRUE north, downslope-facing direction", "angle", "topography",
    ),
    _channel(14, "elevation", "SRTM", "m", "continuous", "topography"),
    _channel(15, "pdsi", "GRIDMET/DROUGHT pdsi (pentad contendo t)", "dimensionless", "continuous", "pdsi"),
    _channel(16, "landcover", "MCD12Q1 LC_Type1 (IGBP), ano do fogo", "class 1..17", "categorical", None),
    _channel(17, "forecast_precipitation", GFS_SOURCE, "mm (definição muda em 2019-11-07)", "continuous", "gfs"),
    _channel(18, "forecast_wind_speed", GFS_SOURCE, "m/s", "continuous", "gfs"),
    _channel(
        19, "forecast_wind_direction", GFS_SOURCE + ", atan(v/u)",
        "deg in [-90, 90]: eixo do vetor (u, v), anti-horário desde leste, quadrante perdido",
        "axial_angle", "gfs",
    ),
    _channel(20, "forecast_temperature", GFS_SOURCE, "degC", "continuous", "gfs"),
    _channel(21, "forecast_specific_humidity", GFS_SOURCE, "kg/kg", "continuous", "gfs"),
    _channel(
        22, "active_fire", "VNP14IMG acq_date == t (UTC), confiança != baixa",
        "UTC hour hh (0 = sem detecção)", "fire_hour", None,
    ),
]
CHANNEL_NAMES = [channel["name"] for channel in RAW_CHANNELS]
CONTINUOUS_CHANNELS = tuple(c["index"] for c in RAW_CHANNELS if c["kind"] == "continuous")
ANGLE_CHANNELS = tuple(c["index"] for c in RAW_CHANNELS if c["kind"] == "angle")
AXIAL_CHANNELS = tuple(c["index"] for c in RAW_CHANNELS if c["kind"] == "axial_angle")
MISSING_GROUPS: dict[str, tuple[int, ...]] = {}
for _entry in RAW_CHANNELS:
    if _entry["missing_group"] is not None:
        MISSING_GROUPS[_entry["missing_group"]] = MISSING_GROUPS.get(_entry["missing_group"], ()) + (_entry["index"],)
LANDCOVER_CLASSES = (
    "evergreen_needleleaf_forests",
    "evergreen_broadleaf_forests",
    "deciduous_needleleaf_forests",
    "deciduous_broadleaf_forests",
    "mixed_forests",
    "closed_shrublands",
    "open_shrublands",
    "woody_savannas",
    "savannas",
    "grasslands",
    "permanent_wetlands",
    "croplands",
    "urban_and_built_up_lands",
    "cropland_natural_vegetation_mosaics",
    "permanent_snow_and_ice",
    "barren",
    "water_bodies",
)


# Pré-registro do experimento (`README.md`), fixado antes de qualquer métrica em dados completos. Fica
# aqui para ter **uma** definição, consumida pelo executor e pelos geradores de resultado, de modo que nenhum
# deles possa afrouxar o critério — ou esticar o orçamento — depois de ver os números.
ABLATION_SIGMA_MULTIPLE = 2.0
ABLATION_MIN_ABS_AP = 0.01
PREREGISTERED_EPOCHS = 20
PREREGISTERED_RUNS = 9  # 3 configurações × 3 sementes


def ablation_is_relevant(effect: float, pooled_sd: float) -> bool | None:
    """Aplica a regra pré-registrada: |ΔAP| > 2 × desvio agrupado **e** > 0,01.

    Devolve `None` quando o desvio agrupado não é estimável (uma única semente por configuração), porque
    nesse caso a regra não tem resposta — e "não estimável" não é a mesma coisa que "não relevante".
    O terceiro critério do pré-registro, o IC pareado por eventos excluindo zero, depende dos vetores por
    evento e é verificado onde eles existem; esta função cobre os dois critérios computáveis a partir das médias.
    """
    if pooled_sd != pooled_sd or effect != effect:
        return None
    return bool(abs(effect) > ABLATION_SIGMA_MULTIPLE * pooled_sd and abs(effect) > ABLATION_MIN_ABS_AP)


def split_years(fold_id: int = DEFAULT_FOLD) -> dict[str, list[int]]:
    if not 0 <= fold_id < len(FOLDS):
        raise ValueError(f"fold_id deve estar em [0, {len(FOLDS) - 1}], recebido {fold_id}.")
    fold = FOLDS[fold_id]
    return {"train": list(fold[:2]), "validation": [fold[2]], "test": [fold[3]]}


# --------------------------------------------------------------------------- split espacial
#
# Por que existe: o split por ano separa **tempo**, não espaço. Na EDA, 96% dos eventos de teste caem sobre a
# pegada de algum evento de treino e 67% com mais de metade da área — os mesmos incêndios recorrem nas mesmas
# regiões, então o modelo vê no teste quase a mesma topografia e vegetação que viu no treino. Qualquer métrica
# ali responde "generaliza para um ano novo?", nunca "generaliza para uma região nova?".
#
# Por que blocos e não componentes conexas: a pegada mediana de um evento é de 93 km. Agrupar eventos por
# sobreposição transitiva junta 98,4% das janelas num único aglomerado, mesmo com buffer zero — não dá split.
# A alternativa padrão na literatura de validação espacial é o **bloco**: uma grade grosseira sobre a área de
# estudo, blocos inteiros atribuídos a cada conjunto, e um corredor de exclusão que descarta os eventos que
# atravessam a fronteira entre dois conjuntos. O preço é perder esses eventos; o ganho é que a independência
# espacial passa a ser verificável, e não suposta.
ALL_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021)
SPATIAL_BLOCK_M = 600_000.0
SPATIAL_BUFFER_M = 10_000.0
SPATIAL_SHARES: dict[str, float] = {"train": 0.60, "validation": 0.15, "test": 0.25}
SPATIAL_MIN_BLOCKS = 2
SPLIT_KINDS = ("temporal", "spatial", "spatiotemporal")


def event_footprint(event: "EventRef") -> tuple[float, float, float, float]:
    """Retângulo da imagem em EPSG:32610: (oeste, sul, leste, norte) em metros.

    Aproximação deliberada: o HDF5 guarda o centro em lon/lat e o tamanho em pixels, não o canto da grade.
    `H×W` pixels de 375 m centrados em `lnglat` reproduzem a pegada com erro muito abaixo do buffer de
    exclusão, que é a escala que importa aqui.
    """
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", SOURCE_CRS, always_xy=True).transform(*event.lnglat)
    half_w, half_h = event.width * PIXEL_SIZE_M / 2, event.height * PIXEL_SIZE_M / 2
    return x - half_w, y - half_h, x + half_w, y + half_h


def boxes_intersect(a: Sequence[float], b: Sequence[float], buffer_m: float = 0.0) -> bool:
    """Dois retângulos se tocam, dilatados por `buffer_m`."""
    return bool(a[0] - buffer_m <= b[2] and b[0] <= a[2] + buffer_m
                and a[1] - buffer_m <= b[3] and b[1] <= a[3] + buffer_m)


def overlap_fraction(box: Sequence[float], others: Sequence[Sequence[float]]) -> float:
    """Maior fração da área de `box` coberta por um único retângulo de `others`."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area <= 0:
        return 0.0
    best = 0.0
    for other in others:
        width = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
        height = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
        best = max(best, width * height / area)
    return best


def box_gap_m(box: Sequence[float], others: Sequence[Sequence[float]]) -> float | None:
    """Menor folga (em metros) entre `box` e os retângulos de `others`; 0 quando há sobreposição."""
    if not others:
        return None
    return min(max(0.0, max(o[0] - box[2], box[0] - o[2], o[1] - box[3], box[1] - o[3])) for o in others)


def event_window_counts(events: Sequence["EventRef"], *, t_in: int = DEFAULT_T_IN, t_out: int = 1) -> np.ndarray:
    """Janelas válidas por evento. É a unidade que o modelo de fato consome, então é o peso do balanceamento."""
    return np.array([len(valid_window_starts(parse_dates(e.dates), t_in=t_in, t_out=t_out)) for e in events],
                    dtype=np.int64)


def spatial_blocks(events: Sequence["EventRef"], block_m: float = SPATIAL_BLOCK_M) -> tuple[list[str], dict[str, list[int]]]:
    """Rotula cada evento com o bloco da grade que contém o **centro** da sua pegada.

    A origem da grade é o canto sudoeste dos centros observados, o que torna o rótulo determinístico para um
    dado conjunto de eventos. Usar o centro (e não a pegada toda) é o que permite um rótulo único por evento;
    quem atravessa fronteira é tratado depois, pelo corredor de exclusão.
    """
    boxes = [event_footprint(e) for e in events]
    xs = np.array([(b[0] + b[2]) / 2 for b in boxes])
    ys = np.array([(b[1] + b[3]) / 2 for b in boxes])
    if not len(xs):
        return [], {}
    cols = np.floor((xs - xs.min()) / block_m).astype(int)
    rows = np.floor((ys - ys.min()) / block_m).astype(int)
    labels = [f"{c}_{r}" for c, r in zip(cols, rows)]
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    return labels, grouped


def assign_blocks(grouped: dict[str, list[int]], weights: np.ndarray, shares: dict[str, float],
                  min_blocks: int = SPATIAL_MIN_BLOCKS) -> dict[str, str]:
    """Atribui blocos inteiros a conjuntos, aproximando as frações de `shares` em peso.

    Greedy determinístico: o bloco mais pesado primeiro, para o conjunto com o maior déficit relativo ao seu
    alvo. Depois um **reparo**: se algum conjunto ficou com menos de `min_blocks` blocos, recebe o menor bloco
    do conjunto mais rico em blocos. O reparo custa quase nada em peso e evita que um conjunto seja uma única
    região contígua — um limiar calibrado numa região só não transfere.
    """
    if not grouped:
        return {}
    total = float(weights.sum()) or 1.0
    block_weight = {block: float(weights[index].sum()) for block, index in grouped.items()}
    order = sorted(grouped, key=lambda b: (-block_weight[b], b))
    assigned: dict[str, str] = {}
    accumulated = {name: 0.0 for name in shares}
    for block in order:
        target = min(shares, key=lambda name: (accumulated[name] - shares[name] * total) / shares[name])
        assigned[block] = target
        accumulated[target] += block_weight[block]

    for _ in range(len(order)):
        counts = {name: sum(1 for b, s in assigned.items() if s == name) for name in shares}
        needy = [name for name in shares if counts[name] < min_blocks]
        donors = [name for name in shares if counts[name] > min_blocks]
        if not needy or not donors:
            break
        donor = max(donors, key=lambda name: counts[name])
        moved = min((b for b, s in assigned.items() if s == donor), key=lambda b: (block_weight[b], b))
        assigned[moved] = min(needy, key=lambda name: (counts[name], name))
    return assigned


def spatial_split(
    events: Sequence["EventRef"],
    *,
    block_m: float = SPATIAL_BLOCK_M,
    buffer_m: float = SPATIAL_BUFFER_M,
    shares: dict[str, float] | None = None,
    min_blocks: int = SPATIAL_MIN_BLOCKS,
    t_in: int = DEFAULT_T_IN,
) -> dict[str, Any]:
    """Split por blocos espaciais, com corredor de exclusão entre conjuntos.

    Devolve as chaves de evento por conjunto, as chaves descartadas e os metadados que tornam o resultado
    auditável: blocos, pesos, frações realizadas e a verificação de sobreposição residual. Todos os anos
    aparecem em todos os conjuntos de propósito — este split isola **espaço**, e misturar as duas separações
    tornaria impossível dizer qual delas explica uma diferença de desempenho.
    """
    shares = dict(shares or SPATIAL_SHARES)
    labels, grouped = spatial_blocks(events, block_m)
    weights = event_window_counts(events, t_in=t_in)
    assigned = assign_blocks(grouped, weights, shares, min_blocks)
    split_of = [assigned[label] for label in labels]
    boxes = [event_footprint(e) for e in events]

    # Corredor de exclusão: um evento cuja pegada toca (dilatada pelo buffer) a de um evento de outro
    # conjunto é descartado. É o que transforma "blocos disjuntos" em "eventos espacialmente disjuntos".
    keep = [True] * len(events)
    for i in range(len(events)):
        for j in range(len(events)):
            if i != j and split_of[i] != split_of[j] and boxes_intersect(boxes[i], boxes[j], buffer_m):
                keep[i] = False
                break

    assignment = {name: [] for name in shares}
    for index, event in enumerate(events):
        if keep[index]:
            assignment[split_of[index]].append(event.key)
    dropped = [event.key for index, event in enumerate(events) if not keep[index]]
    total_weight = float(weights.sum()) or 1.0

    kept_boxes = {name: [boxes[i] for i in range(len(events)) if keep[i] and split_of[i] == name] for name in shares}
    verification = {}
    for name in ("validation", "test"):
        gaps = [box_gap_m(box, kept_boxes["train"]) for box in kept_boxes.get(name, [])]
        gaps = [g for g in gaps if g is not None]
        verification[name] = {
            "share_with_any_overlap_of_train": float(np.mean([
                overlap_fraction(box, kept_boxes["train"]) > 0 for box in kept_boxes.get(name, [])
            ])) if kept_boxes.get(name) else None,
            "min_gap_to_train_km": round(min(gaps) / 1000, 1) if gaps else None,
            "median_gap_to_train_km": round(float(np.median(gaps)) / 1000, 1) if gaps else None,
        }

    return {
        "kind": "spatial",
        "assignment": assignment,
        "dropped": dropped,
        "parameters": {"block_m": block_m, "buffer_m": buffer_m, "shares": shares,
                       "min_blocks": min_blocks, "t_in": t_in},
        "blocks": {
            "n_blocks": len(grouped),
            "split_by_block": assigned,
            "windows_by_block": {b: int(weights[i].sum()) for b, i in sorted(grouped.items())},
            "blocks_per_split": {name: sum(1 for s in assigned.values() if s == name) for name in shares},
        },
        "windows": {
            **{name: int(sum(weights[i] for i in range(len(events)) if keep[i] and split_of[i] == name))
               for name in shares},
            "dropped": int(sum(weights[i] for i in range(len(events)) if not keep[i])),
            "total": int(total_weight),
        },
        "realized_shares": {
            name: round(float(sum(weights[i] for i in range(len(events)) if keep[i] and split_of[i] == name))
                        / total_weight, 4) for name in shares
        },
        "n_events": {name: len(keys) for name, keys in assignment.items()},
        "n_dropped_events": len(dropped),
        "verification": verification,
        "note": (
            "Todos os anos aparecem em todos os conjuntos: este split isola espaço, não tempo. Os eventos "
            "descartados são os que atravessam o corredor de exclusão entre conjuntos — é o preço de tornar a "
            "independência espacial verificável."
        ),
    }


# --------------------------------------------------------------------------- split espaço-temporal
#
# Por que existe: nenhum dos dois splits anteriores responde à pergunta que a aplicação faz. O temporal separa
# anos mas repete regiões (96% dos eventos de teste caem sobre pegada de treino); o espacial separa regiões mas
# repete anos, de modo que o modelo já viu a estação de fogo do ano em que é avaliado. Uma previsão operacional
# enfrenta as duas coisas ao mesmo tempo: um período que ainda não aconteceu, num lugar sem histórico de treino.
#
# Como é construído: as duas separações são **cruzadas**. Os blocos espaciais vêm de `spatial_split` (grade
# grosseira + corredor de exclusão); os anos vêm de um fold oficial do benchmark (`split_years`: dois anos de
# treino, um de validação, um de teste, como recomendam os autores do dataset). Cada evento tem um grupo de
# bloco e um grupo de ano, portanto cai em exatamente uma das nove células da grade 3×3:
#
#                          anos de treino      ano de validação     ano de teste
#   blocos de treino       TREINO              (selada)             future_only
#   blocos de validação    (selada)            VALIDAÇÃO            (selada)
#   blocos de teste        space_only          (selada)             TESTE
#
# `TESTE` é a estimativa conservadora: lugar novo **e** período novo. `future_only` e `space_only` são os dois
# cenários intermediários, avaliados com o **mesmo** modelo treinado na célula de treino, e existem porque sem
# eles uma queda no teste não tem causa atribuível: `future_only` mede o deslocamento temporal com a geografia
# familiar, `space_only` mede o deslocamento espacial dentro do período de treino. O que a queda do teste tem
# além da soma das duas é a interação — o quanto os dois deslocamentos se amplificam.
#
# As quatro células sem papel são declaradas e contadas no inventário em vez de desaparecerem: selecionar
# modelo em (blocos de validação × ano de teste) calibraria o threshold no período de teste, e em (blocos de
# teste × ano de validação) calibraria na região de teste. Ambas ficam seladas.
SPATIOTEMPORAL_CELLS: dict[str, tuple[str, str]] = {
    # célula -> (grupo de blocos, grupo de anos)
    "train": ("train", "train"),
    "validation": ("validation", "validation"),
    "test": ("test", "test"),
    "future_only": ("train", "test"),
    "space_only": ("test", "train"),
}
SPATIOTEMPORAL_EVAL_CELLS = ("test", "future_only", "space_only")
# Gates de viabilidade de célula, fixados **antes** de olhar qualquer rótulo: só contagens de evento e de janela
# entram aqui. A validação precisa de eventos suficientes para calibrar threshold e escolher época; o teste,
# para que um IC por bootstrap de eventos não seja vazio.
SPATIOTEMPORAL_GATES: dict[str, dict[str, int]] = {
    "validation": {"windows": 300, "events": 10},
    "test": {"windows": 300, "events": 20},
}
# Fold escolhido para o protocolo espaço-temporal (ver `spatiotemporal_fold_table`): é o único fold viável que
# mantém 2021 como ano de teste, o mesmo do fold 2 do protocolo temporal, e o de maior volume de treino entre
# os viáveis. Ano de teste comum aos dois protocolos é o que faz a comparação entre eles isolar **uma**
# diferença — a restrição espacial — em vez de duas.
SPATIOTEMPORAL_FOLD = 6


def spatiotemporal_split(
    events: Sequence["EventRef"],
    *,
    fold: int = SPATIOTEMPORAL_FOLD,
    t_in: int = DEFAULT_T_IN,
    **spatial_kwargs: Any,
) -> dict[str, Any]:
    """Cruza os blocos espaciais com os anos de um fold oficial; devolve as cinco células e a auditoria.

    Os eventos descartados pelo corredor de exclusão continuam descartados: a independência espacial entre as
    células vem inteira de `spatial_split`, e aqui só se acrescenta a separação temporal. O resultado é
    determinístico dado (eventos, fold, parâmetros de bloco).
    """
    base = spatial_split(events, t_in=t_in, **spatial_kwargs)
    block_group = {key: name for name, keys in base["assignment"].items() for key in keys}
    years = split_years(fold)
    year_group = {year: name for name, items in years.items() for year in items}
    weights = event_window_counts(events, t_in=t_in)
    weight_of = {event.key: int(w) for event, w in zip(events, weights)}
    by_key = {event.key: event for event in events}

    groups = ("train", "validation", "test")
    unknown = sorted({e.year for e in events} - set(year_group))
    if unknown:
        raise ValueError(
            f"Eventos nos anos {unknown}, que não estão no fold {fold} ({years}). O cruzamento exige que todo "
            "evento tenha um grupo de ano: um ano fora do fold ficaria invisível em vez de cair numa célula."
        )
    grid_keys: dict[str, dict[str, list[str]]] = {b: {y: [] for y in groups} for b in groups}
    for event in events:
        if event.key in block_group:
            grid_keys[block_group[event.key]][year_group[event.year]].append(event.key)

    assignment = {cell: sorted(grid_keys[b][y]) for cell, (b, y) in SPATIOTEMPORAL_CELLS.items()}
    used = set(SPATIOTEMPORAL_CELLS.values())
    sealed = {
        f"{b}_blocks_x_{y}_years": sorted(grid_keys[b][y])
        for b in groups for y in groups if (b, y) not in used
    }

    def totals(keys: Sequence[str]) -> dict[str, int]:
        return {"events": len(keys), "windows": sum(weight_of[k] for k in keys)}

    windows = {cell: totals(keys)["windows"] for cell, keys in assignment.items()}
    windows["sealed"] = sum(totals(keys)["windows"] for keys in sealed.values())
    windows["dropped"] = base["windows"]["dropped"]
    windows["total"] = base["windows"]["total"]

    viability = {}
    for name, gate in SPATIOTEMPORAL_GATES.items():
        measured = totals(assignment[name])
        viability[name] = {
            **measured,
            "gate": gate,
            "passes": bool(measured["windows"] >= gate["windows"] and measured["events"] >= gate["events"]),
        }

    boxes = {cell: [event_footprint(by_key[k]) for k in keys] for cell, keys in assignment.items()}
    dates = {cell: [parse_dates(by_key[k].dates) for k in keys] for cell, keys in assignment.items()}
    train_boxes, train_dates = boxes["train"], dates["train"]
    train_first = min((d.min() for d in train_dates), default=None)
    train_last = max((d.max() for d in train_dates), default=None)

    verification: dict[str, Any] = {}
    for cell in ("validation", "test", "future_only", "space_only"):
        cell_boxes, cell_dates = boxes[cell], dates[cell]
        gaps = [g for g in (box_gap_m(box, train_boxes) for box in cell_boxes) if g is not None]
        # Folga temporal: dias entre o último dia da célula de treino e o primeiro dia da célula avaliada.
        # Negativa quando os períodos se sobrepõem — o esperado em `space_only`, que é do próprio período de
        # treino — e positiva quando a célula é inteiramente futura.
        cell_first = min((d.min() for d in cell_dates), default=None)
        gap_days = (int((cell_first - train_last) / np.timedelta64(1, "D"))
                    if cell_first is not None and train_last is not None else None)
        verification[cell] = {
            **totals(assignment[cell]),
            "years": sorted({by_key[k].year for k in assignment[cell]}),
            "share_with_any_overlap_of_train": float(np.mean([
                overlap_fraction(box, train_boxes) > 0 for box in cell_boxes])) if cell_boxes else None,
            "min_gap_to_train_km": round(min(gaps) / 1000, 1) if gaps else None,
            "median_gap_to_train_km": round(float(np.median(gaps)) / 1000, 1) if gaps else None,
            "days_from_last_train_day_to_first_day": gap_days,
            "fully_after_train": bool(gap_days is not None and gap_days > 0),
        }

    return {
        "kind": "spatiotemporal",
        "fold": fold,
        "years": years,
        "assignment": assignment,
        "sealed_cells": sealed,
        "dropped": base["dropped"],
        "parameters": {**base["parameters"], "fold": fold},
        "blocks": base["blocks"],
        "grid": {b: {y: totals(keys) for y, keys in row.items()} for b, row in grid_keys.items()},
        "cells": {cell: {"blocks": b, "years": sorted(years[y]), **totals(assignment[cell])}
                  for cell, (b, y) in SPATIOTEMPORAL_CELLS.items()},
        "windows": windows,
        "n_events": {cell: len(keys) for cell, keys in assignment.items()},
        "n_dropped_events": base["n_dropped_events"],
        "train_period": {"first_day": str(train_first) if train_first is not None else None,
                         "last_day": str(train_last) if train_last is not None else None},
        "viability": viability,
        "verification": verification,
        "evaluation_cells": list(SPATIOTEMPORAL_EVAL_CELLS),
        "note": (
            "O teste é lugar novo e período novo ao mesmo tempo; `future_only` (blocos de treino, ano de teste) "
            "e `space_only` (blocos de teste, anos de treino) decompõem a perda, avaliados com o mesmo modelo. "
            "As células seladas não entram em treino, seleção nem relatório de desempenho."
        ),
    }


def spatiotemporal_fold_table(events: Sequence["EventRef"], *, t_in: int = DEFAULT_T_IN,
                              **spatial_kwargs: Any) -> dict[str, Any]:
    """Tamanho das cinco células em cada um dos 12 folds oficiais, com o gate de viabilidade aplicado.

    Serve a duas perguntas que não se responde sem a tabela: qual fold usar (nem todos sobrevivem ao
    cruzamento — o fold 2, do protocolo temporal, deixa a validação com 6 eventos) e quantos folds restam para
    uma validação cruzada entre anos, que é a recomendação formal dos autores do dataset.
    """
    base = spatial_split(events, t_in=t_in, **spatial_kwargs)
    block_group = {key: name for name, keys in base["assignment"].items() for key in keys}
    weights = event_window_counts(events, t_in=t_in)
    cell_weight: dict[tuple[str, int], int] = {}
    cell_events: dict[tuple[str, int], int] = {}
    for event, weight in zip(events, weights):
        group = block_group.get(event.key)
        if group is None:
            continue
        cell_weight[(group, event.year)] = cell_weight.get((group, event.year), 0) + int(weight)
        cell_events[(group, event.year)] = cell_events.get((group, event.year), 0) + 1

    def cell(block: str, years: Sequence[int]) -> dict[str, int]:
        return {"events": sum(cell_events.get((block, y), 0) for y in years),
                "windows": sum(cell_weight.get((block, y), 0) for y in years)}

    folds: dict[str, Any] = {}
    for fold_id in range(len(FOLDS)):
        years = split_years(fold_id)
        cells = {name: cell(block, years[year_group])
                 for name, (block, year_group) in SPATIOTEMPORAL_CELLS.items()}
        passes = all(cells[name]["windows"] >= gate["windows"] and cells[name]["events"] >= gate["events"]
                     for name, gate in SPATIOTEMPORAL_GATES.items())
        folds[str(fold_id)] = {"years": years, "cells": cells, "passes_gates": passes}
    viable = [int(key) for key, value in folds.items() if value["passes_gates"]]
    return {
        "gates": SPATIOTEMPORAL_GATES,
        "folds": folds,
        "viable_folds": viable,
        "viable_with_test_year_2021": [f for f in viable if split_years(f)["test"] == [2021]],
        "chosen": SPATIOTEMPORAL_FOLD,
        "criterion": (
            "entre os folds que passam os gates, manter 2021 como ano de teste (o mesmo do fold 2 temporal, "
            "para que a comparação entre protocolos isole a restrição espacial) e, entre esses, o de maior "
            "volume de treino. Só contagem de eventos, data e pegada entram no critério — nenhum rótulo."
        ),
        "note": (
            "A partição de blocos é a mesma em todos os folds, então esta tabela varia apenas o tempo. Uma "
            "validação cruzada que também rotacionasse os blocos é extensão; não está implementada."
        ),
    }

@dataclass(frozen=True)
class SplitSpec:
    """Um split resolvido em listas de eventos, qualquer que seja o critério que o gerou.

    Existe para que EDA, treino e diagnósticos não precisem saber se a separação é por ano ou por região:
    todos pedem `spec.events["train"]`. `identity` é o que entra na proveniência e no cheque de normalização —
    o conjunto de chaves de treino, que é a única coisa que realmente importa para detectar vazamento.
    """

    kind: str
    fold: int | None
    events: dict[str, list["EventRef"]]
    metadata: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "split_kind": self.kind,
            "fold": self.fold,
            "n_events": {name: len(items) for name, items in self.events.items()},
            "train_event_keys": sorted(e.key for e in self.events["train"]),
        }

    def describe(self) -> dict[str, Any]:
        """Resumo para relatório, sem a lista inteira de eventos."""
        return {
            "kind": self.kind,
            "fold": self.fold,
            "n_events": {name: len(items) for name, items in self.events.items()},
            "years_by_split": {name: sorted({e.year for e in items}) for name, items in self.events.items()},
            **{k: v for k, v in self.metadata.items()
               if k not in ("assignment", "sealed_cells", "dropped")},
        }


def resolve_split(
    hdf5_root: str | Path,
    kind: str = "temporal",
    fold: int | None = None,
    *,
    definition: dict[str, Any] | None = None,
    **spatial_kwargs: Any,
) -> SplitSpec:
    """Resolve o split pedido em listas de `EventRef`.

    `temporal` usa os anos do fold oficial. `spatial` usa blocos geográficos sobre **todos** os anos.
    `spatiotemporal` cruza os dois: blocos de treino nos anos de treino, blocos de teste no ano de teste, mais
    as células `future_only` e `space_only`, que decompõem a perda. Quando `definition` é passado (o JSON
    versionado do split), o resultado recomputado é conferido contra ele e qualquer divergência é erro — é o
    que impede o split de mudar de baixo de um experimento já rodado.
    """
    if kind not in SPLIT_KINDS:
        raise ValueError(f"kind deve ser um de {SPLIT_KINDS}, recebido {kind!r}.")
    if fold is None:
        # Cada protocolo tem o seu fold padrão: o temporal mantém o fold 2, fixado antes da EDA e comparável
        # com o benchmark; o espaço-temporal usa o fold 6, o único viável que preserva 2021 como ano de teste.
        fold = SPATIOTEMPORAL_FOLD if kind == "spatiotemporal" else DEFAULT_FOLD
    if kind == "temporal":
        years = split_years(fold)
        return SplitSpec(
            kind="temporal",
            fold=fold,
            events={name: list_events(hdf5_root, value) for name, value in years.items()},
            metadata={"years": years, "note": "separa anos; não separa regiões (ver `spatial_overlap_with_train`)"},
        )

    events = list_events(hdf5_root, ALL_YEARS)
    if kind == "spatial":
        result = spatial_split(events, **spatial_kwargs)
    else:
        result = spatiotemporal_split(events, fold=fold, **spatial_kwargs)
    if definition is not None:
        recorded = {name: sorted(keys) for name, keys in definition.get("assignment", {}).items()}
        computed = {name: sorted(keys) for name, keys in result["assignment"].items()}
        if recorded != computed:
            raise ValueError(
                f"O split {kind} recomputado difere do versionado em `{kind}_split.json`. Isso significa que o "
                "conjunto de eventos ou os parâmetros mudaram: um experimento já rodado não pode ser comparado "
                "com este. Regenere o split de propósito e registre o desvio e o motivo."
            )
    by_key = {e.key: e for e in events}
    return SplitSpec(
        kind=kind,
        fold=result.get("fold"),
        events={name: [by_key[k] for k in keys] for name, keys in result["assignment"].items()},
        metadata=result,
    )


# --------------------------------------------------------------------------- datas e janelas


def parse_dates(dates: Iterable[Any]) -> np.ndarray:
    """Converte `img_dates` (str/bytes `YYYY-MM-DD`) em `datetime64[D]`, validando o formato."""
    parsed = []
    for raw in dates:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        text = text.replace(".tif", "")
        if not DATE_PATTERN.match(text):
            raise ValueError(f"Data fora do formato YYYY-MM-DD: {text!r}")
        parsed.append(np.datetime64(text, "D"))
    return np.asarray(parsed, dtype="datetime64[D]")


def date_gaps(dates: np.ndarray) -> list[dict[str, Any]]:
    """Intervalos entre datas consecutivas diferentes de 1 dia (inclui duplicatas/desordem)."""
    deltas = np.diff(np.asarray(dates, dtype="datetime64[D]")).astype(int)
    return [
        {"after_index": int(i), "from": str(dates[i]), "to": str(dates[i + 1]), "delta_days": int(d)}
        for i, d in enumerate(deltas)
        if d != 1
    ]


def valid_window_starts(
    dates: np.ndarray,
    *,
    t_in: int = DEFAULT_T_IN,
    t_out: int = 1,
    skip_initial: int = 0,
    stride: int = 1,
) -> list[int]:
    """Inícios de janelas com `t_in + t_out` dias estritamente consecutivos dentro de um evento.

    Busca diária (o dataset tem um passo por dia), com `skip_initial`, que
    reproduz `n_leading_observations_test_adjustment` do oficial. Esse alinhamento de alvos entre
    `t_in` diferentes só vale sem lacunas; comparações entre `t_in` devem alinhar pela data do alvo.
    """
    if t_in <= 0 or t_out <= 0 or stride <= 0 or skip_initial < 0:
        raise ValueError("t_in, t_out e stride devem ser positivos; skip_initial >= 0.")
    dates = np.asarray(dates, dtype="datetime64[D]")
    window = t_in + t_out
    one_day = np.timedelta64(1, "D")
    return [
        start
        for start in range(skip_initial, len(dates) - window + 1, stride)
        if np.all(np.diff(dates[start : start + window]) == one_day)
    ]


# --------------------------------------------------------------------------- HDF5 por evento


@dataclass(frozen=True)
class EventRef:
    year: int
    fire_name: str
    path: Path
    n_days: int
    height: int
    width: int
    dates: tuple[str, ...]
    lnglat: tuple[float, float]

    @property
    def key(self) -> str:
        return f"{self.year}/{self.fire_name}"


def open_h5(path: str | Path, mode: str = "r", *, cache_bytes: int | None = None) -> "h5py.File":
    """Abre HDF5 sem file locking; `cache_bytes` dimensiona o cache de chunks (1 chunk = 1 dia)."""
    import h5py

    if cache_bytes is None:
        return h5py.File(path, mode, locking=False)
    return h5py.File(path, mode, locking=False, rdcc_nbytes=cache_bytes, rdcc_nslots=10007)


def read_event_ref(path: str | Path) -> EventRef:
    with open_h5(path) as handle:
        data = handle["data"]
        if data.ndim != 4 or data.shape[1] != N_RAW_CHANNELS:
            raise ValueError(f"{path}: shape inesperado {data.shape}; esperado [T, 23, H, W].")
        dates = tuple(str(d) for d in parse_dates(data.attrs["img_dates"]))
        lng, lat = (float(v) for v in data.attrs["lnglat"])
        return EventRef(
            year=int(data.attrs["year"]),
            fire_name=str(data.attrs["fire_name"]),
            path=Path(path),
            n_days=int(data.shape[0]),
            height=int(data.shape[2]),
            width=int(data.shape[3]),
            dates=dates,
            lnglat=(lng, lat),
        )


def list_events(hdf5_root: str | Path, years: Sequence[int]) -> list[EventRef]:
    """Inventário lazy (só atributos) dos eventos dos anos pedidos, ordenado por ano/nome."""
    root = Path(hdf5_root)
    events = []
    for year in years:
        for path in sorted((root / str(year)).glob("*.hdf5")):
            ref = read_event_ref(path)
            if ref.year != int(year):
                raise ValueError(f"{path}: atributo year={ref.year} diverge da pasta {year}.")
            events.append(ref)
    return events


def read_days(path: str | Path, start: int, stop: int) -> np.ndarray:
    """Lê apenas os dias `[start, stop)` de um evento como `float32 [T, 23, H, W]`."""
    with open_h5(path) as handle:
        return np.asarray(handle["data"][start:stop], dtype=np.float32)


def iter_event_days(event: EventRef, chunk_days: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Itera um evento em blocos de dias, sem carregá-lo inteiro."""
    with open_h5(event.path) as handle:
        data = handle["data"]
        for start in range(0, data.shape[0], chunk_days):
            yield start, np.asarray(data[start : start + chunk_days], dtype=np.float32)


# --------------------------------------------------------------------------- canal de fogo ativo


def preprocess_active_fire(raw_hhmm: np.ndarray) -> tuple[np.ndarray, int]:
    """Replica o oficial (NaN -> 0; hhmm -> hh por divisão inteira).

    Retorna também o número de detecções com `hhmm < 100` (00:00–00:59 UTC), que o
    pré-processamento oficial transforma em 0 e, portanto, apaga (registrar como inconsistência).
    """
    raw = np.asarray(raw_hhmm, dtype=np.float32)
    detected = ~np.isnan(raw)
    lost = int(np.count_nonzero(detected & (raw < 100)))
    hours = np.floor_divide(np.nan_to_num(raw, nan=0.0), 100).astype(np.float32)
    return hours, lost


def fire_mask(frame_or_af: np.ndarray) -> np.ndarray:
    """Máscara binária de fogo ativo a partir de um frame `[23, H, W]` ou do canal `[H, W]`."""
    array = np.asarray(frame_or_af)
    af = array[ACTIVE_FIRE_CHANNEL] if array.ndim == 3 else array
    return (np.nan_to_num(af, nan=0.0) > 0).astype(np.uint8)


# --------------------------------------------------------------------------- estatísticas em streaming


def landcover_invalid(values: np.ndarray) -> np.ndarray:
    """Máscara de pixels cujo land cover não é inteiro finito em 1..17 (NaN incluído)."""
    values = np.asarray(values)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    return ~finite | (safe != np.round(safe)) | (safe < 1) | (safe > N_LANDCOVER_CLASSES)


class StreamingChannelStats:
    """Média/desvio/min/max/faltantes por canal contínuo, acumulados em float64.

    A variância usa `E[x²] − E[x]²` em float64; o erro de cancelamento é desprezível nas
    escalas do WSTS (verificado contra a tabela oficial na EDA).
    """

    def __init__(self, channels: Sequence[int] = CONTINUOUS_CHANNELS) -> None:
        self.channels = tuple(channels)
        n = len(self.channels)
        self.count = np.zeros(n, dtype=np.int64)
        self.missing = np.zeros(n, dtype=np.int64)
        self.total = np.zeros(n, dtype=np.int64)
        self.sum = np.zeros(n, dtype=np.float64)
        self.sumsq = np.zeros(n, dtype=np.float64)
        self.min = np.full(n, np.inf)
        self.max = np.full(n, -np.inf)
        self.landcover_counts = np.zeros(N_LANDCOVER_CLASSES + 1, dtype=np.int64)

    def update(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4 or frames.shape[1] != N_RAW_CHANNELS:
            raise ValueError(f"Esperado [T, 23, H, W], recebido {frames.shape}.")
        for position, channel in enumerate(self.channels):
            values = frames[:, channel].astype(np.float64, copy=False).ravel()
            valid = values[np.isfinite(values)]
            self.total[position] += values.size
            self.missing[position] += values.size - valid.size
            if valid.size:
                self.count[position] += valid.size
                self.sum[position] += valid.sum()
                self.sumsq[position] += np.square(valid).sum()
                self.min[position] = min(self.min[position], float(valid.min()))
                self.max[position] = max(self.max[position], float(valid.max()))
        landcover = frames[:, LANDCOVER_CHANNEL].ravel()
        invalid = landcover_invalid(landcover)
        self.landcover_counts[0] += int(np.count_nonzero(invalid))
        classes = landcover[~invalid].astype(np.int64)
        self.landcover_counts[1:] += np.bincount(classes, minlength=N_LANDCOVER_CLASSES + 1)[1:]

    def to_dict(self) -> dict[str, Any]:
        mean = np.divide(self.sum, self.count, out=np.zeros_like(self.sum), where=self.count > 0)
        var = np.divide(self.sumsq, self.count, out=np.zeros_like(self.sum), where=self.count > 0) - mean**2
        std = np.sqrt(np.clip(var, 0.0, None))
        channels = {}
        for position, channel in enumerate(self.channels):
            degenerate = not std[position] > 0
            channels[CHANNEL_NAMES[channel]] = {
                "index": channel,
                "mean": float(mean[position]),
                "std": 1.0 if degenerate else float(std[position]),
                "std_replaced_by_one": degenerate,
                "min": float(self.min[position]) if self.count[position] else None,
                "max": float(self.max[position]) if self.count[position] else None,
                "valid": int(self.count[position]),
                "missing_rate": float(self.missing[position] / self.total[position]) if self.total[position] else None,
            }
        return {
            "channels": channels,
            "landcover_counts": {
                "invalid_or_nan": int(self.landcover_counts[0]),
                **{name: int(self.landcover_counts[i + 1]) for i, name in enumerate(LANDCOVER_CLASSES)},
            },
        }


def fit_normalization_for_fold(hdf5_root: str | Path, fold_id: int) -> dict[str, Any]:
    """Ajusta média/desvio só nos eventos dos anos de treino do fold, dia a dia.

    O JSON resultante registra fold, anos e a lista de eventos (com SHA-256 do sidecar de
    conversão, quando existe) para auditar que nenhum ano de validação/teste entrou.
    """
    import json

    train_years = split_years(fold_id)["train"]
    events = list_events(hdf5_root, train_years)
    stats = StreamingChannelStats()
    for event in events:
        for _, block in iter_event_days(event):
            stats.update(block)
    return normalization_payload(stats, fold_id, events)


def normalization_payload(stats: StreamingChannelStats, fold_id: int, events: Sequence[EventRef]) -> dict[str, Any]:
    """JSON de normalização a partir de estatísticas já acumuladas nos eventos de treino do fold."""
    import json

    train_years = split_years(fold_id)["train"]
    foreign = sorted({e.year for e in events} - set(train_years))
    if foreign:
        raise ValueError(f"Eventos de anos fora do treino do fold {fold_id}: {foreign}.")
    event_log = []
    for event in events:
        sidecar = event.path.with_suffix(".json")
        sha = json.loads(sidecar.read_text(encoding="utf-8")).get("sha256") if sidecar.exists() else None
        event_log.append({"key": event.key, "n_days": event.n_days, "sha256": sha})
    payload = stats.to_dict()
    payload.update({"fold": fold_id, "fit_years": train_years, "n_events": len(events), "events": event_log})
    return payload


def check_normalization_fold(normalization: dict[str, Any], fold_id: int) -> None:
    expected = split_years(fold_id)["train"]
    if normalization.get("fit_years") != expected or normalization.get("fold") != fold_id:
        raise ValueError(
            f"Normalização ajustada em fold={normalization.get('fold')} anos={normalization.get('fit_years')}; "
            f"esperado fold={fold_id} anos={expected}."
        )


def fit_normalization_for_split(hdf5_root: str | Path, split: SplitSpec) -> dict[str, Any]:
    """Ajusta média/desvio só nos eventos de treino do split, dia a dia, para qualquer critério de split."""
    stats = StreamingChannelStats()
    for event in split.events["train"]:
        for _, block in iter_event_days(event):
            stats.update(block)
    return split_normalization_payload(stats, split)


def split_normalization_payload(stats: StreamingChannelStats, split: SplitSpec) -> dict[str, Any]:
    """JSON de normalização para um `SplitSpec`, com a identidade do conjunto de treino registrada.

    Para o split temporal mantém `fold`/`fit_years`, de modo que arquivos antigos e o cheque por ano
    continuam válidos. Para qualquer split grava `train_event_keys`: é a verificação forte, porque compara o
    conjunto exato de eventos usados, e não apenas o rótulo do ano.
    """
    import json

    events = split.events["train"]
    event_log = []
    for event in events:
        sidecar = event.path.with_suffix(".json")
        sha = json.loads(sidecar.read_text(encoding="utf-8")).get("sha256") if sidecar.exists() else None
        event_log.append({"key": event.key, "n_days": event.n_days, "sha256": sha})
    payload = stats.to_dict()
    payload.update({
        "split_kind": split.kind,
        "fold": split.fold,
        "fit_years": sorted({e.year for e in events}),
        "n_events": len(events),
        "events": event_log,
        "train_event_keys": sorted(e.key for e in events),
    })
    return payload


def check_normalization(normalization: dict[str, Any], split: SplitSpec) -> None:
    """Confere que a normalização foi ajustada **exatamente** nos eventos de treino deste split.

    Compara o conjunto de chaves, não os anos: um arquivo ajustado no split temporal tem anos coerentes e
    ainda assim seria errado para o split espacial, cujo treino é outro conjunto de eventos. Quando o arquivo
    é antigo e não traz `train_event_keys`, cai no cheque por ano — que só existe para o split temporal.
    """
    if normalization.get("split_kind") not in (None, split.kind):
        raise ValueError(
            f"Normalização ajustada para split {normalization.get('split_kind')!r}; esperado {split.kind!r}."
        )
    recorded = normalization.get("train_event_keys")
    if recorded is None:
        if split.kind != "temporal":
            raise ValueError(
                "Normalização sem `train_event_keys`: não é possível verificar que foi ajustada nos eventos "
                f"de treino do split {split.kind!r}. Reajuste com `fit_normalization_for_split`."
            )
        check_normalization_fold(normalization, split.fold if split.fold is not None else DEFAULT_FOLD)
        return
    expected = sorted(e.key for e in split.events["train"])
    if sorted(recorded) != expected:
        missing = sorted(set(expected) - set(recorded))[:3]
        extra = sorted(set(recorded) - set(expected))[:3]
        raise ValueError(
            f"Normalização ajustada em {len(recorded)} eventos; o treino deste split tem {len(expected)}. "
            f"Faltando (até 3): {missing}. Sobrando (até 3): {extra}."
        )


# --------------------------------------------------------------------------- transformação de entrada


# Grupos que podem ser removidos da entrada em ablations (canais brutos + flag de ausência do grupo).
DROPPABLE_GROUPS = {"vegetation_index": (3, 4), "gfs": (17, 18, 19, 20, 21)}


def input_channel_spec(
    *, include_forecast_precip: bool = False, drop_groups: Sequence[str] = ()
) -> list[tuple[str, str, Any]]:
    """Lista ordenada `(nome, tipo, origem)` dos canais de entrada do modelo.

    `drop_groups` remove grupos inteiros (`DROPPABLE_GROUPS`), incluindo a flag de ausência do grupo.
    """
    unknown = set(drop_groups) - set(DROPPABLE_GROUPS)
    if unknown:
        raise ValueError(f"Grupos desconhecidos em drop_groups: {sorted(unknown)}; válidos: {sorted(DROPPABLE_GROUPS)}.")
    dropped = {c for group in drop_groups for c in DROPPABLE_GROUPS[group]}
    if not include_forecast_precip:
        dropped.add(FORECAST_PRECIP_CHANNEL)
    spec: list[tuple[str, str, Any]] = []
    spec += [(CHANNEL_NAMES[c], "zscore", c) for c in CONTINUOUS_CHANNELS if c not in dropped]
    for c in ANGLE_CHANNELS:
        if c not in dropped:
            spec += [(f"{CHANNEL_NAMES[c]}_sin", "sin", c), (f"{CHANNEL_NAMES[c]}_cos", "cos", c)]
    for c in AXIAL_CHANNELS:
        if c not in dropped:
            spec += [(f"{CHANNEL_NAMES[c]}_sin2", "sin2", c), (f"{CHANNEL_NAMES[c]}_cos2", "cos2", c)]
    spec += [(f"landcover_{name}", "onehot", k + 1) for k, name in enumerate(LANDCOVER_CLASSES)]
    spec += [
        (f"missing_{group}", "missing", channels)
        for group, channels in MISSING_GROUPS.items()
        if group not in drop_groups
    ]
    spec.append(("active_fire_binary", "fire", ACTIVE_FIRE_CHANNEL))
    return spec


def input_channel_names(*, include_forecast_precip: bool = False, drop_groups: Sequence[str] = ()) -> list[str]:
    return [name for name, _, _ in input_channel_spec(include_forecast_precip=include_forecast_precip, drop_groups=drop_groups)]


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.float32(0.0))


def transform_frames(
    frames: np.ndarray,
    normalization: dict[str, Any],
    *,
    include_forecast_precip: bool = False,
    drop_groups: Sequence[str] = (),
) -> np.ndarray:
    """`[T, 23, H, W]` brutos -> `[C, T, H, W]` float32 no contrato de `input_channel_spec`.

    Contínuos: z-score com estatísticas do treino, não finito -> 0 (média de treino).
    Ângulos: `sin`/`cos`; eixo (canal 19): `sin 2θ`/`cos 2θ`. Land cover: one-hot 17.
    Flags de ausência por grupo marcadas antes da imputação. Fogo ativo: binário.
    A saída é pré-alocada e preenchida no lugar (pico de memória ~ 1 cópia da saída).
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[1] != N_RAW_CHANNELS:
        raise ValueError(f"Esperado [T, 23, H, W], recebido {frames.shape}.")
    spec = input_channel_spec(include_forecast_precip=include_forecast_precip, drop_groups=drop_groups)
    landcover = frames[:, LANDCOVER_CHANNEL]
    if landcover_invalid(landcover).any():
        raise ValueError(
            f"Land cover fora de 1..{N_LANDCOVER_CLASSES} ou não finito: "
            f"min={np.nanmin(landcover)}, max={np.nanmax(landcover)}."
        )
    classes = landcover.astype(np.int64)
    t, _, h, w = frames.shape
    out = np.empty((len(spec), t, h, w), dtype=np.float32)
    stats = normalization["channels"]
    for position, (name, kind, source) in enumerate(spec):
        if kind == "zscore":
            entry = stats[name]
            out[position] = _finite_or_zero((frames[:, source] - np.float32(entry["mean"])) / np.float32(entry["std"]))
        elif kind in {"sin", "cos", "sin2", "cos2"}:
            factor = 2.0 if kind.endswith("2") else 1.0
            radians = np.deg2rad(factor * frames[:, source])
            out[position] = _finite_or_zero(np.sin(radians) if kind.startswith("sin") else np.cos(radians))
        elif kind == "onehot":
            out[position] = classes == source
        elif kind == "missing":
            out[position] = (~np.isfinite(frames[:, list(source)])).any(axis=1)
        else:
            out[position] = np.nan_to_num(frames[:, source], nan=0.0) > 0
    return out


# --------------------------------------------------------------------------- geometria e convenções vetoriais


def grid_geometry(lnglat: Sequence[float], crs: str = SOURCE_CRS) -> dict[str, float]:
    """Rotação e escala da grade UTM do dataset no centro do evento.

    `gamma_deg`: azimute do norte verdadeiro medido na grade (horário a partir do norte da
    grade). Todos os TIFs do WSTS estão em EPSG:32610, inclusive eventos longe do meridiano
    central (−123°), onde |gamma| chega a ~14°. `ground_pixel_m`: tamanho do pixel no terreno
    (375 m de grade / fator de escala).
    """
    from pyproj import CRS, Proj, Transformer

    lng, lat = float(lnglat[0]), float(lnglat[1])
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = transformer.transform(lng, lat)
    x1, y1 = transformer.transform(lng, lat + 0.01)
    gamma = float(np.degrees(np.arctan2(x1 - x0, y1 - y0)))
    factors = Proj(CRS.from_user_input(crs)).get_factors(lng, lat)
    scale = float(factors.meridional_scale)
    return {"gamma_deg": gamma, "scale_factor": scale, "ground_pixel_m": PIXEL_SIZE_M / scale}


def transport_vector(speed: np.ndarray, direction_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vento meteorológico (de onde vem, horário desde o norte verdadeiro) -> transporte (leste, norte).

    Um vento de norte (θ = 0°) sopra para o sul: `(u, v) = (0, −speed)`.
    """
    theta = np.deg2rad(direction_from_deg)
    return -speed * np.sin(theta), -speed * np.cos(theta)


def upslope_vector(slope_deg: np.ndarray, aspect_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradiente de subida (leste, norte verdadeiros) com módulo `tan(slope)`.

    O aspect aponta encosta abaixo; a subida aponta para `aspect + 180°`. Em terreno plano o
    aspect do GEE vale ~0° (norte espúrio), mas o módulo `tan(slope)` anula o vetor.
    """
    magnitude = np.tan(np.deg2rad(slope_deg))
    theta = np.deg2rad(aspect_deg)
    return -magnitude * np.sin(theta), -magnitude * np.cos(theta)


def true_to_grid(east: np.ndarray, north: np.ndarray, gamma_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(leste, norte) verdadeiros -> (leste, norte) da grade: azimute_grade = azimute_verdadeiro + gamma."""
    g = np.deg2rad(gamma_deg)
    return east * np.cos(g) + north * np.sin(g), -east * np.sin(g) + north * np.cos(g)


def to_grid_components(
    east: np.ndarray, north: np.ndarray, gamma_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """(leste, norte) verdadeiros -> (d_coluna, d_linha) no raster (linha cresce para o sul da grade)."""
    grid_east, grid_north = true_to_grid(east, north, gamma_deg)
    return grid_east, -grid_north


def upslope_grid_from_elevation(elevation: np.ndarray, ground_pixel_m: float = PIXEL_SIZE_M) -> tuple[np.ndarray, np.ndarray]:
    """Gradiente de subida (d_coluna, d_linha) direto na grade, em m/m (independe da convenção do aspect).

    `np.gradient` devolve dz/dlinha e dz/dcoluna; como a linha cresce para o sul, um valor positivo
    em d_linha significa subida para o sul — exatamente a convenção de `to_grid_components`.
    """
    d_row, d_col = np.gradient(np.asarray(elevation, dtype=np.float64))
    return d_col / ground_pixel_m, d_row / ground_pixel_m
