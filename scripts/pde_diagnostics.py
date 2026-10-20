"""Diagnósticos empíricos para candidatas a termos de EDP no WildfireSpreadTS.

EDP de referência: ∂s/∂t = ∇·(D∇s) − v·∇s + R·s(1−s) − L, com v = v_vento + v_encosta.
Este script NÃO ajusta a EDP; ele testa se o dado sustenta cada papel físico, por evento-dia:

1. Advecção — direção local de espalhamento (vetor do pixel queimando mais próximo em t até cada
   pixel novo em t+1, média de vetores unitários ponderada pelo comprimento resultante) × transporte
   do vento GRIDMET (dias t e t+1, levado à grade com a convergência γ do evento) e × subida calculada
   pelo gradiente da elevação na própria grade. Nulos: rotação aleatória do driver (isotropia) e
   permutação do driver entre dias do mesmo evento; Holm entre os testes de advecção.
2. Reação/extinção — taxa de crescimento log((A_{t+1}+1)/(A_t+1)) × ERC, umidade específica,
   precipitação, temperatura máxima e vento no fogo (dias t e t+1). Estimador dentro do evento
   (Spearman de ranks centrados por evento), correlação parcial com log(A_t+1) e com a
   observabilidade (fração de NaN nas bandas VIIRS em t+1 perto do fogo), permutação dentro do evento
   coerente com o estimador, IC por bootstrap de eventos e Holm.
3. Combustível — taxa de ignição no anel da frente × NDVI do composto "pré" (termina antes de t,
   sem dias pós-fogo), estimador dentro do evento; taxa por classe de land cover (MCD12Q1 do ano do
   fogo, classificação anual contém pós-fogo — ressalva).
4. Escalas — velocidade da frente não condicionada ao movimento (pixels novos locais / pixels da
   frente, px/dia, dias sem avanço contam zero), fração da frente que avança, distância das ignições
   novas ao fogo em t (> √2 px = mais de uma célula) e em km (pixel de 375/k m). O limite CFL
   (Courant ≤ 1) vale para esquemas explícitos.

Protocolo: somente anos de treino do fold, fixado antes da EDA. O dataset não é verdade de campo
para estatística de propagação (documentação oficial): resultados são evidência de plausibilidade,
não estimativas causais.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import check_resources, resource_snapshot, write_json
from scripts.core import (
    CHANNEL_NAMES,
    DEFAULT_DATA_ROOT,
    DEFAULT_FOLD,
    LANDCOVER_CHANNEL,
    LANDCOVER_CLASSES,
    EventRef,
    fire_mask,
    grid_geometry,
    iter_event_days,
    list_events,
    parse_dates,
    split_years,
    to_grid_components,
    transport_vector,
    upslope_grid_from_elevation,
)
from scripts.env_report import provenance


DEFAULT_REPORT = Path("reports/pde_diagnostics.json")
HDF5_MANIFEST = REPO_ROOT / "reports" / "hdf5_manifest.json"
MAX_LOCAL_SPREAD_PX = 10
SENSITIVITY_RADII_PX = (5, 10, 20)
RING_PX = 2
MIN_NEW_PIXELS = 3
OBSERVABILITY_RADIUS_PX = 10
COMPOSITE_SPAN_DAYS = 15
ONE_CELL = float(np.sqrt(2.0))
# Canais resolvidos por nome: uma renumeracao em `wsts_core.RAW_CHANNELS` falha alto, em vez de
# silenciosamente ler o canal errado.
def _channel(name: str) -> int:
    return CHANNEL_NAMES.index(name)


DRIVERS = {name: _channel(name) for name in (
    "energy_release_component", "specific_humidity", "precipitation", "temperature_max", "wind_speed",
)}
NDVI_CHANNEL = _channel("ndvi")
ELEVATION_CHANNEL = _channel("elevation")
WIND_SPEED_CHANNEL = _channel("wind_speed")
WIND_DIRECTION_CHANNEL = _channel("wind_direction")
VIIRS_CHANNELS = [_channel(name) for name in ("viirs_m11", "viirs_i2", "viirs_i1")]


# --------------------------------------------------------------------------- geometria por evento-dia


def disk(radius: int) -> np.ndarray:
    """Elemento estruturante em disco. `binary_dilation` com a cruz padrão geraria um diamante."""
    offsets = np.arange(-radius, radius + 1)
    return offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius**2


def distance_to_fire(fire_t: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Distância euclidiana ao pixel queimando mais próximo e os índices desse pixel (EDT única)."""
    from scipy import ndimage

    distance, indices = ndimage.distance_transform_edt(~fire_t.astype(bool), return_indices=True)
    return distance, (indices[0], indices[1])


def spread_vectors(
    fire_t: np.ndarray,
    fire_next: np.ndarray,
    max_px: int = MAX_LOCAL_SPREAD_PX,
    precomputed: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any] | None:
    """Direção local de espalhamento na grade (d_col, d_row) e distâncias dos pixels novos à frente.

    Para cada pixel novo (fogo em t+1 e não em t) a até `max_px` do fogo em t, toma o vetor unitário do
    pixel queimando mais próximo até ele; a direção é a média desses vetores (comprimento = coerência).
    `precomputed` reaproveita a EDT já calculada pelo chamador.
    """
    burning = fire_t.astype(bool)
    new = fire_next.astype(bool) & ~burning
    if not burning.any() or not new.any():
        return None
    distance, (near_row, near_col) = precomputed if precomputed is not None else distance_to_fire(burning)
    rows, cols = np.nonzero(new)
    distances = distance[rows, cols]
    local = distances <= max_px
    if local.sum() < MIN_NEW_PIXELS:
        return {"distances_px": distances.astype(np.float32), "direction": None, "n_local": int(local.sum())}
    d_row = (rows - near_row[rows, cols])[local].astype(np.float64)
    d_col = (cols - near_col[rows, cols])[local].astype(np.float64)
    norm = np.hypot(d_row, d_col)
    return {
        "distances_px": distances.astype(np.float32),
        "direction": (float((d_col / norm).mean()), float((d_row / norm).mean())),
        "n_local": int(local.sum()),
    }


def front_statistics(
    fire_t: np.ndarray, fire_next: np.ndarray, distance: np.ndarray | None = None
) -> dict[str, float] | None:
    """Velocidade da frente não condicionada ao movimento e fração da frente que avança (8-vizinhança)."""
    from scipy import ndimage

    burning = fire_t.astype(bool)
    if not burning.any():
        return None
    structure = np.ones((3, 3), dtype=bool)
    front = burning & ndimage.binary_dilation(~burning, structure=structure)
    new = fire_next.astype(bool) & ~burning
    if distance is None:
        distance = ndimage.distance_transform_edt(~burning)
    new_local = new & (distance <= MAX_LOCAL_SPREAD_PX)
    advancing = front & ndimage.binary_dilation(new, structure=structure)
    n_front = int(front.sum())
    return {
        "front_pixels": n_front,
        "speed_px_per_day": float(new_local.sum() / n_front) if n_front else 0.0,
        "share_front_advancing": float(advancing.sum() / n_front) if n_front else 0.0,
    }


def mean_vector(d_col: np.ndarray, d_row: np.ndarray, where: np.ndarray) -> tuple[float, float] | None:
    selected = where & np.isfinite(d_col) & np.isfinite(d_row)
    if not selected.any():
        return None
    return float(d_col[selected].mean()), float(d_row[selected].mean())


def cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    na, nb = float(np.hypot(*a)), float(np.hypot(*b))
    if na == 0 or nb == 0:
        return None
    return float((a[0] * b[0] + a[1] * b[1]) / (na * nb))


# --------------------------------------------------------------------------- estatística


def weighted_mean_cosine(spread: np.ndarray, driver: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Cosseno médio ponderado. O peso padrão é o comprimento resultante da direção de espalhamento.

    Esse peso tem viés de amostra pequena: com poucos pixels novos, E[R] ≈ 0,5 mesmo sob isotropia,
    então dias com direção mal estimada pesam demais. `analyze_advection` reporta a sensibilidade a
    pesos alternativos (uniforme e √n).
    """
    resultant = np.linalg.norm(spread, axis=1)
    cos = (spread * driver).sum(axis=1) / (resultant * np.linalg.norm(driver, axis=1))
    w = resultant if weights is None else weights
    return float(np.sum(w * cos) / np.sum(w))


def nonzero_rows(*arrays: np.ndarray) -> np.ndarray:
    """Linhas em que todos os vetores dados têm norma finita e positiva (NaN reprova)."""
    keep = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        keep &= np.linalg.norm(array, axis=1) > 0
    return keep


def rotate(driver: np.ndarray, angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.column_stack([driver[:, 0] * cos - driver[:, 1] * sin, driver[:, 0] * sin + driver[:, 1] * cos])


def rotation_pvalue(
    spread: np.ndarray, driver: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Nulo de isotropia com **um ângulo por evento**, não por dia.

    O vento tem direção predominante por evento e por região, então rotacionar dia a dia encolheria a
    variância do nulo por ~nº de dias em vez de ~nº de eventos. Em simulação sob H0 verdadeiro, a versão
    por dia rejeitava 24,7% a 5% nominal; por evento, 4,0%. O p é reportado nos dois sentidos: um
    alinhamento significativamente negativo indicaria erro de convenção, e precisa ser detectável.
    """
    observed = weighted_mean_cosine(spread, driver)
    index_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    greater = smaller = 0
    for _ in range(n_perm):
        rotated = driver.copy()
        for index in index_by_group:
            rotated[index] = rotate(driver[index], rng.uniform(0, 2 * np.pi))
        value = weighted_mean_cosine(spread, rotated)
        greater += value >= observed
        smaller += value <= observed
    return {
        "mean_cosine_weighted": observed,
        "p_value_one_sided": (greater + 1) / (n_perm + 1),
        "p_value_two_sided": min(1.0, 2 * min(greater + 1, smaller + 1) / (n_perm + 1)),
        "n_permutations": n_perm,
        "n_events": len(index_by_group),
        "unit_of_randomization": "evento",
    }


def circular_shift(values: np.ndarray, index: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Desloca circularmente as linhas de um evento, preservando a autocorrelação temporal."""
    return np.roll(values[index], int(rng.integers(len(index))), axis=0)


def shift_pvalue(
    spread: np.ndarray, driver: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Nulo de variação diária por deslocamento circular dentro do evento.

    Permutar livremente os dias destrói a autocorrelação do driver e infla o erro tipo I (13% medido a
    5% nominal com AR(1) ρ = 0,8). O deslocamento circular quebra o pareamento dia a dia mas mantém a
    estrutura temporal, então o nulo fica calibrado. Testa só a variação **dentro** do evento: preserva
    o vento predominante, logo um p alto não é evidência contra advecção.
    """
    observed = weighted_mean_cosine(spread, driver)
    index_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    exceed = 0
    for _ in range(n_perm):
        shifted = driver.copy()
        for index in index_by_group:
            shifted[index] = circular_shift(driver, index, rng)
        exceed += weighted_mean_cosine(spread, shifted) >= observed
    return {
        "mean_cosine_weighted": observed,
        "p_value": (exceed + 1) / (n_perm + 1),
        "n_permutations": n_perm,
        "n_events": len(index_by_group),
        "unit_of_randomization": "dia dentro do evento (deslocamento circular)",
        "note": "preserva o vento predominante do evento; p alto não é evidência contra advecção",
    }



def partial_fit(spread: np.ndarray, wind: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Mínimos quadrados de `spread_unit ≈ a·wind_unit + b·slope_unit` empilhando as duas componentes."""
    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.divide(v, n, out=np.zeros_like(v), where=n > 0)

    s, w, u = unit(spread), unit(wind), unit(slope)
    design = np.column_stack([w.ravel(), u.ravel()])
    coef, *_ = np.linalg.lstsq(design, s.ravel(), rcond=None)
    return coef


def bootstrap_by_group(
    statistic: Callable[[np.ndarray, np.ndarray], Any],
    groups: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """IC 95% reamostrando eventos inteiros, com réplicas rotuladas separadamente.

    `statistic` recebe `(índices, rótulos de grupo da amostra)`. Os rótulos são novos a cada réplica:
    sem isso, um evento sorteado duas vezes viraria um único grupo de tamanho dobrado e os estimadores
    dentro do evento (ranks centrados por grupo) mudariam de sentido.
    """
    unique = np.unique(groups)
    if len(unique) < 3:
        return None
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    values, failures = [], 0
    for _ in range(n_boot):
        parts, labels = [], []
        for replica, g in enumerate(rng.choice(unique, size=len(unique), replace=True)):
            parts.append(index_by_group[g])
            labels.append(np.full(len(index_by_group[g]), replica))
        value = statistic(np.concatenate(parts), np.concatenate(labels))
        if value is None or not np.all(np.isfinite(value)):
            failures += 1
            continue
        values.append(value)
    if not values:
        return {"ci95": None, "n_valid": 0, "n_boot": n_boot, "failures": failures}
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5], axis=0)
    ci = [low.tolist(), high.tolist()] if np.ndim(low) else [float(low), float(high)]
    return {"ci95": ci, "n_valid": len(values), "n_boot": n_boot, "failures": failures}


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy import stats

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5 or np.ptp(x[finite]) == 0 or np.ptp(y[finite]) == 0:
        return None
    return float(stats.spearmanr(x[finite], y[finite]).statistic)


def centered_ranks(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Rank **global** (0..1) centrado na média de cada grupo; NaN preservado.

    O rank é global de propósito. Ranquear dentro de cada evento parece mais natural — remove o nível
    do evento junto com a escala — mas introduz um viés de amostra pequena: com poucos dias por evento,
    o rank fica grosseiro e a residualização linear nos controles sub-corrige, deixando em `x` e `y`
    uma dependência comum do controle. Medido sob H0 verdadeiro com 19 dias por evento (a mediana
    real): viés de +0,074 ranqueando dentro do evento contra +0,020 ranqueando globalmente, com a
    mesma capacidade de detecção. A centragem por grupo, feita depois, preserva o efeito fixo de evento.
    """
    from scipy import stats

    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return out
    out[finite] = stats.rankdata(values[finite]) / finite.sum()
    for g in np.unique(groups):
        index = np.flatnonzero((groups == g) & finite)
        if len(index) >= 2:
            out[index] -= out[index].mean()
        else:
            out[index] = np.nan
    return out


def control_diagnostics(control: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Quanto um controle varia **dentro** do evento — se não varia, o ajuste parcial é inerte.

    `centered_ranks` devolve 0 para uma covariável constante dentro do evento (empate total), e a
    residualização então não remove nada. Sem este diagnóstico, o relatório diria "parcial em X" sem
    que nenhum ajuste tivesse acontecido.
    """
    stds, varying, total = [], 0, 0
    for g in np.unique(groups):
        values = control[(groups == g) & np.isfinite(control)]
        if len(values) >= 2:
            total += 1
            stds.append(float(values.std()))
            varying += values.std() > 0
    share = varying / total if total else None
    return {
        "within_event_std_median": float(np.median(stds)) if stds else None,
        "share_events_with_variation": share,
        "n_events_evaluated": total,
        "effective": bool(share) and share > 0.5,
    }


def rank_design(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, controls: Sequence[np.ndarray] = ()
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Ranks centrados por evento de `x` e `y`, a matriz de desenho dos controles e os grupos.

    Os ranks são calculados **depois** da máscara conjunta de finitos: ranquear sobre um superconjunto
    deixaria a centragem por evento inexata e reintroduziria variação entre eventos.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    for control in controls:
        finite &= np.isfinite(control)
    if finite.sum() < 5:
        return None
    g = groups[finite]
    a, b = centered_ranks(x[finite], g), centered_ranks(y[finite], g)
    inner = np.isfinite(a) & np.isfinite(b)
    columns = []
    for control in controls:
        ranked = centered_ranks(control[finite], g)
        inner &= np.isfinite(ranked)
        columns.append(ranked)
    if inner.sum() < 5:
        return None
    design = np.column_stack([np.ones(inner.sum())] + [c[inner] for c in columns])
    return a[inner], b[inner], design, g[inner]


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def correlate(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def within_event_partial(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, controls: Sequence[np.ndarray] = ()
) -> float | None:
    """Correlação de ranks centrados por evento (Spearman dentro do evento), parcial nos controles."""
    prepared = rank_design(x, y, groups, controls)
    if prepared is None:
        return None
    a, b, design, _ = prepared
    return correlate(residualize(a, design), residualize(b, design))


def freedman_lane_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    controls: Sequence[np.ndarray],
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Nulo de Freedman–Lane: desloca o **resíduo** de `x` nos controles, não o `x` bruto.

    Permutar `x` bruto quebra também a relação entre `x` e os controles, então o nulo deixa de ser
    trocável para uma estatística parcial — justamente o caso de precipitação × nuvem. O deslocamento
    é circular dentro do evento, para preservar a autocorrelação temporal.

    O passo que não pode faltar é reconstruir `x* = ajustado + resíduo deslocado` e **residualizar de
    novo**: o resíduo deslocado ganha uma componente ao longo dos controles, e essa componente não
    contribui para a covariância com `y`. Sem a re-ortogonalização o nulo fica estreito demais e o
    erro tipo I sobe (medido: 0,22 a 5% nominal).
    """
    prepared = rank_design(x, y, groups, controls)
    if prepared is None:
        return None
    a, b, design, g = prepared
    residual_y = residualize(b, design)
    fitted_x = a - residualize(a, design)
    residual_x = a - fitted_x
    observed = correlate(residual_x, residual_y)
    if observed is None:
        return None
    index_by_group = [np.flatnonzero(g == value) for value in np.unique(g)]
    column = residual_x.reshape(-1, 1)
    null, degenerate = [], 0
    for _ in range(n_perm):
        shifted = residual_x.copy()
        for index in index_by_group:
            shifted[index] = circular_shift(column, index, rng).ravel()
        value = correlate(residualize(fitted_x + shifted, design), residual_y)
        if value is None:
            degenerate += 1
            continue
        null.append(value)
    if not null:
        return None
    return {
        "statistic": observed,
        "p_value": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)),
        "n_permutations": len(null),
        "degenerate_permutations": degenerate,
        "null": "Freedman-Lane: deslocamento circular do resíduo de x nos controles, dentro do evento, "
        "com re-residualização de x* nos controles",
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


# --------------------------------------------------------------------------- coleta


class CompositeTracker:
    """Composto NDVI vigente por dia do evento; devolve o composto "pré" (termina antes de t).

    Uma troca de composto é um dia em que mais da metade dos pixels válidos muda (filtra ruído pixel a
    pixel). Um composto iniciado no dia `s` cobre `[s, s+15]`, então só serve como "pré" para
    `day > s + 15` — é essa janela que garante que o valor não contém observações posteriores ao fogo.
    O histórico guarda no máximo `MAX_HISTORY` compostos: `pre()` só precisa do mais recente que já
    terminou, e cada entrada custa uma cópia `H×W` float32.
    """

    MAX_HISTORY = 3
    SWITCH_MIN_FRACTION = 0.5

    def __init__(self) -> None:
        self.history: list[tuple[int, np.ndarray]] = []
        self.switches = 0

    def update(self, day: int, ndvi: np.ndarray) -> None:
        if not self.history:
            self.history.append((day, ndvi.copy()))
            return
        previous = self.history[-1][1]
        both = np.isfinite(ndvi) & np.isfinite(previous)
        if both.any() and np.mean(ndvi[both] != previous[both]) > self.SWITCH_MIN_FRACTION:
            self.history.append((day, ndvi.copy()))
            self.switches += 1
            del self.history[: -self.MAX_HISTORY]

    def pre(self, day: int) -> np.ndarray | None:
        chosen = None
        for start, values in self.history:
            if start + COMPOSITE_SPAN_DAYS < day:
                chosen = values
        return chosen


def collect(events: Sequence[EventRef]) -> dict[str, Any]:
    from scipy import ndimage

    advection_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    fuel_rows: list[dict[str, Any]] = []
    front_rows: list[dict[str, Any]] = []
    landcover_ring = np.zeros((len(LANDCOVER_CLASSES) + 1, 2), dtype=np.int64)
    distances_px: list[np.ndarray] = []
    # (escala do evento, nº de pixels novos daquele evento-dia): expandido só na análise, para não
    # materializar uma lista Python com um float por pixel novo do treino inteiro.
    scale_runs: list[tuple[float, int]] = []
    all_nan_driver_days = 0
    for event_id, event in enumerate(events):
        geometry = grid_geometry(event.lnglat)
        gamma = geometry["gamma_deg"]
        dates = parse_dates(event.dates)
        composites = CompositeTracker()
        previous = None
        # Área já queimada no evento, acumulada em todo dia observado — inclusive nos que o laço descarta
        # por lacuna ou por ausência de fogo. Sem isso o anel de combustível pega a faixa logo atrás da
        # frente, que queimou dias antes: ali o NDVI já está deprimido e a taxa de ignição é baixa, o que
        # produz associação positiva sem nenhum efeito de combustível.
        burned = None
        for index, block in iter_event_days(event):
            frame = block[0]
            day = int((dates[index] - dates[0]).astype(int))
            consecutive = previous is not None and dates[index] - dates[index - 1] == np.timedelta64(1, "D")
            previous_frame, previous = previous, frame
            fire_here = fire_mask(frame).astype(bool)
            # `burned_through_t` é a união de todos os dias **anteriores** a este frame, o que inclui o dia t
            # e exclui t+1. Tem de ser capturado antes da atualização: descontar t+1 por diferença depois
            # re-incluiria no anel pixels que já haviam queimado em dias anteriores.
            burned_through_t = burned
            burned = fire_here.copy() if burned is None else (burned | fire_here)
            if not consecutive:
                composites.update(day, frame[NDVI_CHANNEL])
                continue
            fire_t, fire_next = fire_mask(previous_frame).astype(bool), fire_here
            if not fire_t.any() or burned_through_t is None:
                composites.update(day, frame[NDVI_CHANNEL])
                continue
            # EDT única por evento-dia: reaproveitada pela frente, pelo anel e pelo espalhamento.
            distance, nearest = distance_to_fire(fire_t)
            front_rows.append({"event": event_id, **front_statistics(fire_t, fire_next, distance)})
            area_t, area_next = int(fire_t.sum()), int(fire_next.sum())
            near_fire = ndimage.binary_dilation(fire_t, structure=disk(OBSERVABILITY_RADIUS_PX))
            observability = float(np.mean(~np.isfinite(frame[VIIRS_CHANNELS][:, near_fire])))
            drivers = {}
            with warnings.catch_warnings():
                # Um driver inteiramente NaN nos pixels de fogo vira NaN e é contado, não silenciado.
                warnings.simplefilter("ignore", RuntimeWarning)
                for name, channel in DRIVERS.items():
                    drivers[f"{name}_t"] = float(np.nanmean(previous_frame[channel][fire_t]))
                    drivers[f"{name}_t1"] = float(np.nanmean(frame[channel][fire_t]))
            if not all(np.isfinite(v) for v in drivers.values()):
                all_nan_driver_days += 1
            growth_rows.append({
                "event": event_id,
                "growth": float(np.log((area_next + 1) / (area_t + 1))),
                "log_area_t": float(np.log(area_t + 1)),
                "viirs_nan_share_t1_near_fire": observability,
                **drivers,
            })

            ring = (~burned_through_t) & (distance <= RING_PX)
            if ring.any():
                ignited = fire_next[ring]
                pre = composites.pre(day - 1)
                ndvi_pre = pre[ring] if pre is not None else np.array([np.nan])
                fuel_rows.append({
                    "event": event_id,
                    "ignition_rate": float(ignited.mean()),
                    "ndvi_pre_ring_mean": float(np.nanmean(ndvi_pre)) if np.isfinite(ndvi_pre).any() else np.nan,
                    "event_duration_days": float(event.n_days),
                    # Controles do bloco de combustível: o tamanho do fogo e o dia dentro do evento. Sem o
                    # segundo, uma tendência temporal comum (o anel amadurece, o NDVI cai, a taxa de ignição
                    # relativa cai) é lida como associação.
                    "log_area_t": float(np.log(area_t + 1)),
                    "day_in_event": float(day),
                    "ring_pixels": float(ring.sum()),
                })
                landcover = previous_frame[LANDCOVER_CHANNEL][ring]
                valid = np.isfinite(landcover) & (landcover >= 1) & (landcover <= len(LANDCOVER_CLASSES))
                np.add.at(landcover_ring, (landcover[valid].astype(np.int64), ignited[valid].astype(np.int64)), 1)
            composites.update(day, frame[NDVI_CHANNEL])

            spread = spread_vectors(fire_t, fire_next, precomputed=(distance, nearest))
            if spread is None:
                continue
            distances_px.append(spread["distances_px"])
            scale_runs.append((geometry["ground_pixel_m"], len(spread["distances_px"])))
            if spread["direction"] is None:
                continue
            winds = {}
            for suffix, source in (("t", previous_frame), ("t1", frame)):
                east, north = transport_vector(source[WIND_SPEED_CHANNEL], source[WIND_DIRECTION_CHANNEL])
                wind_col, wind_row = to_grid_components(east, north, gamma)
                winds[suffix] = mean_vector(wind_col, wind_row, fire_t)
            up_col, up_row = upslope_grid_from_elevation(previous_frame[ELEVATION_CHANNEL], geometry["ground_pixel_m"])
            slope = mean_vector(up_col, up_row, ring if ring.any() else fire_t)
            if winds["t"] is None or winds["t1"] is None or slope is None:
                continue
            advection_rows.append({
                "event": event_id,
                "spread": spread["direction"],
                "wind_t": winds["t"],
                "wind_t1": winds["t1"],
                "upslope": slope,
                "n_local_new_pixels": spread["n_local"],
            })
    return {
        "advection": advection_rows,
        "growth": growth_rows,
        "fuel": fuel_rows,
        "front": front_rows,
        "landcover_ring_counts": landcover_ring,
        "distances_px": np.concatenate(distances_px) if distances_px else np.zeros(0, dtype=np.float32),
        "ground_pixel_m": (
            np.repeat([s for s, _ in scale_runs], [n for _, n in scale_runs]) if scale_runs else np.zeros(0)
        ),
        "all_nan_driver_event_days": all_nan_driver_days,
    }


# --------------------------------------------------------------------------- análise


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in (10, 50, 90, 99)} | {"mean": float(values.mean()), "n": int(values.size)}


def analyze_advection(rows: list[dict[str, Any]], n_perm: int, n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    if len(rows) < 5:
        return {"n_event_days": len(rows)}
    spread_all = np.array([r["spread"] for r in rows])
    drivers_all = {name: np.array([r[name] for r in rows]) for name in ("wind_t", "wind_t1", "upslope")}
    groups_all = np.array([r["event"] for r in rows])
    # Filtro por teste: terreno plano zera só `upslope`, e não deve descartar o evento-dia dos testes
    # de vento, que têm dado válido. O filtro conjunto fica para o ajuste parcial, que usa os dois.
    tests: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for name, driver_all in drivers_all.items():
        keep = nonzero_rows(spread_all, driver_all)
        spread, driver, groups = spread_all[keep], driver_all[keep], groups_all[keep]
        if len(spread) < 5:
            tests[name] = {"n_event_days": int(len(spread)), "dropped_zero_norm": int((~keep).sum())}
            continue
        counts = np.array([r["n_local_new_pixels"] for r in rows], dtype=float)[keep]
        tests[name] = {
            "n_event_days": int(len(spread)),
            "n_events": int(len(np.unique(groups))),
            "dropped_zero_norm": int((~keep).sum()),
            "rotation_null": rotation_pvalue(spread, driver, groups, n_perm, rng),
            "within_event_circular_shift": shift_pvalue(spread, driver, groups, n_perm, rng),
            "weight_sensitivity": {
                "resultant_length (padrão)": weighted_mean_cosine(spread, driver),
                "uniforme": weighted_mean_cosine(spread, driver, np.ones(len(spread))),
                "sqrt(n_pixels_novos)": weighted_mean_cosine(spread, driver, np.sqrt(counts)),
            },
        }
        p_values[name] = tests[name]["rotation_null"]["p_value_two_sided"]
    for name, adjusted in holm(p_values).items():
        tests[name]["rotation_null"]["p_value_holm"] = adjusted

    joint = nonzero_rows(spread_all, drivers_all["wind_t"], drivers_all["upslope"])
    partial: dict[str, Any] = {"n_event_days": int(joint.sum()), "dropped_zero_norm": int((~joint).sum())}
    if joint.sum() >= 5:
        spread, wind, up = spread_all[joint], drivers_all["wind_t"][joint], drivers_all["upslope"][joint]
        coef = partial_fit(spread, wind, up)
        partial["coefficients"] = {"wind_t": float(coef[0]), "upslope": float(coef[1])}
        partial["ci95_bootstrap_events"] = bootstrap_by_group(
            lambda idx, _g: partial_fit(spread[idx], wind[idx], up[idx]), groups_all[joint], n_boot, rng
        )
        partial["wind_upslope_cosine_mean"] = float(
            np.nanmean([np.nan if c is None else c for c in (cosine(w, u) for w, u in zip(wind, up))])
        )
        # Fração da variância direcional explicada pelas duas forçantes. O espalhamento e as forçantes são
        # vetores unitários, então o R² é a norma² da parte ajustada: |a·w + b·u|² com w e u unitários.
        cos_wu = partial["wind_upslope_cosine_mean"]
        a, b = coef[0], coef[1]
        partial["r_squared_direction"] = float(a * a + b * b + 2 * a * b * cos_wu)
        # Ângulo cujo cosseno é o cosseno médio ponderado com o vento: a leitura intuitiva do tamanho de efeito.
        partial["angle_of_mean_cosine_with_wind_deg"] = float(
            np.degrees(np.arccos(np.clip(weighted_mean_cosine(spread, wind), -1.0, 1.0)))
        )
        partial["r_squared_note"] = (
            "fração da direção de espalhamento explicada por vento + subida. Um valor pequeno aqui é o que "
            "impede ler os coeficientes como efeito substancial: o termo entra, mas explica pouco."
        )
    coherence = np.linalg.norm(spread_all, axis=1)
    return {
        "unit": "evento-dia com >= 3 pixels novos a <= 10 px da frente; direção ponderada pelo comprimento resultante",
        "n_event_days_total": len(rows),
        "n_events_total": int(len(np.unique(groups_all))),
        "tests": tests,
        "holm_family": "p bilateral do nulo por rotação de wind_t, wind_t1 e upslope",
        "circular_shift_is_descriptive": "o nulo de deslocamento circular fica fora da família Holm",
        "partial_fit_wind_vs_upslope": partial,
        "directional_coherence": {
            "definition": "norma do vetor médio das direções unitárias de espalhamento no evento-dia; "
            "próximo de 1 = transporte direcional (termo v), próximo de 0 = expansão isotrópica (termo D)",
            **( _quantiles(coherence) or {}),
            # Sem a linha de base, "próximo de 0 = isotrópico" é inutilizável: o comprimento resultante tem
            # viés de amostra pequena, e sob isotropia pura E[R] ≈ 0,886/√n. Um R de 0,4 com n = 5 é
            # compatível com isotropia; com n = 100, não é.
            "isotropy_baseline": {
                "formula": "E[R] ≈ 0,886/√n sob isotropia; comparar com os quantis de n_local_new_pixels",
                "expected_at_n_pixels": {
                    str(n): round(0.886 / np.sqrt(n), 3) for n in (3, 5, 10, 25, 100, 400)
                },
            },
            "n_local_new_pixels": _quantiles(
                np.array([r["n_local_new_pixels"] for r in rows], dtype=float)
            ),
        },
        "note_t1": "forçantes de t+1 são diagnóstico; não estão disponíveis para um modelo preditivo",
    }


def analyze_growth(rows: list[dict[str, Any]], n_perm: int, n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    if len(rows) < 5:
        return {"n_event_days": len(rows)}
    groups = np.array([r["event"] for r in rows])
    target = np.array([r["growth"] for r in rows])
    log_area = np.array([r["log_area_t"] for r in rows])
    observability = np.array([r["viirs_nan_share_t1_near_fire"] for r in rows])
    names = [f"{d}_{s}" for d in DRIVERS for s in ("t", "t1")] + ["viirs_nan_share_t1_near_fire"]
    # Permutações dimensionadas pela família: com k testes e Holm, o menor p ajustado alcançável é
    # k/(n_perm+1). Sem isso, nenhum teste poderia ser significativo por construção.
    n_growth_perm = max(n_perm, 20 * len(names))
    controls_available = {"log_area_t": log_area, "viirs_nan_share_t1_near_fire": observability}
    diagnostics = {name: control_diagnostics(values, groups) for name, values in controls_available.items()}
    results: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for name in names:
        x = np.array([r[name] for r in rows])
        control_names = ["log_area_t"] if name == "viirs_nan_share_t1_near_fire" else list(controls_available)
        controls = [controls_available[c] for c in control_names]
        rho = within_event_partial(x, target, groups, controls)
        if rho is None:
            results[name] = {"partial_within_event_spearman": None, "controls": control_names}
            continue
        test = freedman_lane_pvalue(x, target, groups, controls, n_growth_perm, rng)
        p = test["p_value"] if test else None
        if p is not None:
            p_values[name] = p

        def statistic(idx: np.ndarray, sample_groups: np.ndarray, x: np.ndarray = x,
                      controls: list[np.ndarray] = controls) -> float | None:
            return within_event_partial(x[idx], target[idx], sample_groups, [c[idx] for c in controls])

        results[name] = {
            "partial_within_event_spearman": rho,
            "controls": control_names,
            "controls_effective": {c: diagnostics[c]["effective"] for c in control_names},
            "ci95_bootstrap_events": bootstrap_by_group(statistic, groups, n_boot, rng),
            "p_value_freedman_lane": p,
            "n_permutations": test["n_permutations"] if test else 0,
            "degenerate_permutations": test["degenerate_permutations"] if test else None,
        }
    for name, adjusted in holm(p_values).items():
        results[name]["p_value_holm"] = adjusted
    inert = [name for name, d in diagnostics.items() if not d["effective"]]
    return {
        "unit": "evento-dia com fogo em t",
        "response": "log((A_t+1 + 1)/(A_t + 1)), com A_t a área ativa em t",
        "estimator": "Spearman dentro do evento (ranks centrados por evento), parcial em log(A_t + 1) e na observabilidade",
        "null": "Freedman-Lane: deslocamento circular do resíduo de x nos controles, dentro do evento "
        "(preserva a autocorrelação temporal e a relação x-controles)",
        "n_event_days": len(rows),
        "n_events": int(len(np.unique(groups))),
        "control_diagnostics": diagnostics,
        "inert_controls": inert,
        "warning_inert_controls": (
            f"controles sem variação dentro do evento ({', '.join(inert)}): o ajuste parcial não removeu nada e "
            "nenhuma afirmação do tipo 'a associação sobrevive ao ajuste' pode ser feita"
        ) if inert else None,
        "drivers_at_fire_pixels": results,
        "note": "forçantes de t e t+1 são diagnóstico, não preditores; nuvem/omissão do VIIRS confundem extinção e "
        "um único proxy de NaN não identifica esse caminho causal",
    }



def analyze_fuel(rows: list[dict[str, Any]], counts: np.ndarray, n_perm: int, n_boot: int,
                 rng: np.random.Generator) -> dict[str, Any]:
    if len(rows) < 5:
        return {"n_event_days": len(rows)}
    groups = np.array([r["event"] for r in rows])
    rate = np.array([r["ignition_rate"] for r in rows])
    ndvi = np.array([r["ndvi_pre_ring_mean"] for r in rows])
    by_class = {
        name: {"ring_pixels": int(counts[i + 1].sum()), "ignition_rate": float(counts[i + 1, 1] / counts[i + 1].sum())}
        for i, name in enumerate(LANDCOVER_CLASSES)
        if counts[i + 1].sum() >= 100
    }
    retained = np.isfinite(ndvi)
    durations = np.array([r["event_duration_days"] for r in rows], dtype=float)
    # Controles da correção do anel queimado: o tamanho do fogo e o dia dentro do evento. O segundo é o que
    # separa um efeito de combustível de uma tendência temporal comum ao anel e à taxa de ignição.
    controls = {name: np.array([r[name] for r in rows], dtype=float)
                for name in ("log_area_t", "day_in_event") if name in rows[0]}
    control_values = list(controls.values())
    diagnostics = {name: control_diagnostics(values, groups) for name, values in controls.items()}
    inert = [name for name, d in diagnostics.items() if not d["effective"]]
    test = freedman_lane_pvalue(ndvi, rate, groups, control_values, n_perm, rng) if control_values else None
    return {
        "caveats": [
            "NDVI do composto 'pré' (termina antes de t) para evitar dias pós-fogo; evento-dias sem composto pré ficam de fora",
            "land cover MCD12Q1 do ano do fogo (classificação anual contém pós-fogo)",
            "o subconjunto retido é selecionado para eventos longos: um composto só vira 'pré' depois de 16 dias, "
            "então eventos curtos não contribuem. Comparar as durações abaixo antes de generalizar.",
            "o anel exclui a área já queimada no evento (máscara cumulativa), não apenas a detecção do dia: sem "
            "isso a faixa logo atrás da frente entrava no anel com NDVI já deprimido e taxa de ignição baixa, "
            "produzindo associação positiva sem efeito de combustível",
        ],
        "n_event_days": len(rows),
        "n_event_days_with_pre_composite": int(retained.sum()),
        "n_events": int(len(np.unique(groups))),
        "n_events_with_pre_composite": int(len(np.unique(groups[retained]))),
        "ring_pixels": _quantiles(np.array([r.get("ring_pixels", np.nan) for r in rows], dtype=float)),
        "event_duration_days": {
            "retidos": _quantiles(durations[retained]),
            "descartados": _quantiles(durations[~retained]),
        },
        "controls": list(controls),
        "control_diagnostics": diagnostics,
        "inert_controls": inert,
        "estimator": "Spearman parcial dentro do evento (ranks globais centrados por evento), ajustado em "
        + ", ".join(controls) if controls else "Spearman dentro do evento, sem controles",
        "within_event_spearman_ignition_rate_vs_ndvi_pre": within_event_partial(ndvi, rate, groups, control_values),
        "p_value_freedman_lane": test["p_value"] if test else None,
        "n_permutations": test["n_permutations"] if test else 0,
        "ci95_bootstrap_events": bootstrap_by_group(
            lambda idx, sample_groups: within_event_partial(
                ndvi[idx], rate[idx], sample_groups, [c[idx] for c in control_values]),
            groups, n_boot, rng
        ),
        "ignition_rate_by_landcover_in_ring": by_class,
        "landcover_caveat": "taxas agrupadas sobre todos os evento-dias, sem cluster por evento e sem IC: sujeitas a "
        "efeito Simpson (a composição de classes correlaciona com o evento). Não ranquear inflamabilidade com isto.",
    }


def analyze_scales(front_rows: list[dict[str, Any]], distances: np.ndarray, ground_pixel_m: np.ndarray) -> dict[str, Any]:
    speed = np.array([r["speed_px_per_day"] for r in front_rows])
    advancing = np.array([r["share_front_advancing"] for r in front_rows])
    front_pixels = np.array([r["front_pixels"] for r in front_rows], dtype=float)
    local = distances <= MAX_LOCAL_SPREAD_PX
    km = distances * ground_pixel_m / 1000.0 if distances.size else distances
    sensitivity = {
        f"<= {radius} px": {
            "share_of_all_new": float(np.mean(distances <= radius)) if distances.size else None,
            "distance_px": _quantiles(distances[distances <= radius]) if distances.size else None,
        }
        for radius in SENSITIVITY_RADII_PX
    }
    return {
        "radius_sensitivity": sensitivity,
        "front_speed_px_per_day_unconditional": _quantiles(speed),
        # A velocidade é um quociente por `front_pixels`, cujo p10 é da ordem de 5: numa frente minúscula a
        # cauda alta pode ser ruído de divisão, não fogo rápido. Esta versão condicionada é a que se deve ler
        # para dimensionar Courant.
        "front_speed_px_per_day_front_ge_20px": _quantiles(speed[front_pixels >= 20]),
        "front_pixels": _quantiles(front_pixels),
        "share_front_advancing": _quantiles(advancing),
        "share_event_days_without_advance": float(np.mean(speed == 0)) if speed.size else None,
        "new_pixel_distance_px_local": _quantiles(distances[local]) if distances.size else None,
        "share_new_local_more_than_one_cell": float(np.mean(distances[local] > ONE_CELL + 1e-9)) if local.any() else None,
        "share_new_local_among_all_new": float(local.mean()) if distances.size else None,
        "distance_km_local": _quantiles(km[local]) if distances.size else None,
        "courant_limiting_statistic": (
            "O que limita um esquema explícito é o deslocamento MÁXIMO por passo, não o médio: leia a cauda "
            "(p90/p99) de `new_pixel_distance_px_local` e de `front_speed_px_per_day_unconditional`, nunca a "
            "média ou a mediana."
        ),
        "interpretation": (
            "Número de Courant C = v·Δt/Δx com Δt = 1 dia e Δx = 1 pixel (~375 m). C > 1 na cauda torna instável um "
            "resíduo explícito de diferenças finitas sem subpassos; esquemas implícitos ou semi-lagrangianos não têm "
            "esse limite. A velocidade incondicional inclui dias sem avanço (zero); a distância só existe quando há "
            "avanço, e só conta pixels novos a <= 10 px da frente."
        ),
        "caveats": [
            "a velocidade é cota inferior: pixels novos a mais de 10 px da frente ficam fora do numerador",
            "a máscara de fogo ativo não é perímetro, então `front_pixels` pode se aproximar da área em máscaras esparsas",
        ],
    }



def analyze(collected: dict[str, Any], *, n_perm: int = 2000, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "advection": analyze_advection(collected["advection"], n_perm, n_boot, rng),
        "reaction_extinction": analyze_growth(collected["growth"], n_perm, n_boot, rng),
        "fuel": analyze_fuel(collected["fuel"], collected["landcover_ring_counts"], n_perm, n_boot, rng),
        "scales": analyze_scales(collected["front"], collected["distances_px"], collected["ground_pixel_m"]),
    }


def save_collected(collected: dict[str, Any], path: Path) -> Path:
    """Persiste a coleta para que reanalisar não exija reler todo o HDF5 do treino."""
    arrays: dict[str, np.ndarray] = {}
    for table in ("advection", "growth", "fuel", "front"):
        rows = collected[table]
        arrays[f"{table}__fields"] = np.array(sorted(rows[0]) if rows else [])
        for field in sorted(rows[0]) if rows else []:
            arrays[f"{table}__{field}"] = np.array([r[field] for r in rows])
    arrays["landcover_ring_counts"] = collected["landcover_ring_counts"]
    arrays["distances_px"] = collected["distances_px"]
    arrays["ground_pixel_m"] = collected["ground_pixel_m"]
    arrays["all_nan_driver_event_days"] = np.array(collected["all_nan_driver_event_days"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_collected(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        collected: dict[str, Any] = {}
        for table in ("advection", "growth", "fuel", "front"):
            fields = [str(f) for f in data[f"{table}__fields"]]
            columns = {field: data[f"{table}__{field}"] for field in fields}
            n = len(next(iter(columns.values()))) if columns else 0
            collected[table] = [
                {field: (tuple(values[i]) if values[i].ndim else values[i].item()) for field, values in columns.items()}
                for i in range(n)
            ]
        collected["landcover_ring_counts"] = data["landcover_ring_counts"]
        collected["distances_px"] = data["distances_px"]
        collected["ground_pixel_m"] = data["ground_pixel_m"]
        collected["all_nan_driver_event_days"] = int(data["all_nan_driver_event_days"])
    return collected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnósticos de candidatas a EDP no WSTS (somente treino).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--collected", type=Path, default=None,
                        help="Padrão: <data-root>/stats/pde_collected_fold<F>.npz")
    parser.add_argument("--from-collected", action="store_true",
                        help="Reanalisa a coleta salva, sem reler o HDF5.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Permite rodar com o script não commitado (resultado não publicável).")
    return parser.parse_args(argv)


def check_publishable(report_provenance: dict[str, Any]) -> list[str]:
    """Condições para o JSON valer como evidência: script commitado e manifesto do dataset registrado."""
    problems = []
    if report_provenance["repo"].get("dirty"):
        problems.append("árvore do repositório suja: o código executado não corresponde a um commit")
    if report_provenance["input_sha256"].get("hdf5_manifest") is None:
        problems.append(f"manifesto de conversão ausente ({HDF5_MANIFEST}): a identidade do dataset não fica registrada")
    return problems


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    started = time.perf_counter()
    check_resources(min_available_ram_gib=0.3, min_disk_free_gib=1)
    split = split_years(args.fold)
    collected_path = args.collected or args.data_root / "stats" / f"pde_collected_fold{args.fold}.npz"
    if args.from_collected:
        collected = load_collected(collected_path)
        events = []
    else:
        events = list_events(args.data_root / "hdf5", split["train"])
        if not events:
            raise SystemExit(f"Nenhum evento em {args.data_root / 'hdf5'} para os anos {split['train']}.")
        collected = collect(events)
        save_collected(collected, collected_path)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "fold": args.fold,
            "years_used": split["train"],
            "rule": "somente anos de treino; teste e validação não entram",
            "grid": "vento levado à grade EPSG:32610 com gamma por evento; subida pelo gradiente da elevação na grade",
            "local_spread_max_px": MAX_LOCAL_SPREAD_PX,
            "ring_px": RING_PX,
            "min_new_pixels": MIN_NEW_PIXELS,
            "observability_radius_px": OBSERVABILITY_RADIUS_PX,
            "composite_span_days": COMPOSITE_SPAN_DAYS,
            "drivers": {name: CHANNEL_NAMES[channel] for name, channel in DRIVERS.items()},
            "permutations": args.permutations,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "randomization_units": {
                "rotation_null": "evento (um ângulo por evento)",
                "within_event_null": "dia dentro do evento, por deslocamento circular",
                "bootstrap": "evento, com réplicas rotuladas separadamente",
            },
        },
        "events": [e.key for e in events],
        "n_events": len(events) if events else None,
        "collected_file": collected_path.as_posix(),
        "all_nan_driver_event_days": collected["all_nan_driver_event_days"],
        **analyze(collected, n_perm=args.permutations, n_boot=args.bootstrap, seed=args.seed),
        "provenance": provenance({"hdf5_manifest": HDF5_MANIFEST}),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "resources": resource_snapshot(),
    }
    problems = check_publishable(report["provenance"])
    report["publishable"] = not problems
    report["publishability_problems"] = problems
    write_json(report, args.report)
    if problems and not args.allow_dirty:
        print("ATENÇÃO — resultado NÃO publicável como evidência:")
        for problem in problems:
            print(f"  - {problem}")
    print(f"Eventos: {report['n_events']} | coleta: {collected_path} | relatório: {args.report}")
    return report


if __name__ == "__main__":
    main()
