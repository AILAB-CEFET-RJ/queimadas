"""Propriedades físicas da previsão do STConvS2S: métricas de violação e, depois, penalidades da perda (PIML).

Cada propriedade é uma função de (probabilidade prevista, entradas) que **não usa o rótulo**. A mesma função é
aplicada ao fogo observado em t+1, e é essa comparação que dá sentido ao número: um modelo só "viola" uma
propriedade na medida em que se afasta do que o próprio dado faz. Escolhas e evidência no
relatório local de candidatas a EDP (não exportado):

- `nonburnable`: fração da massa de fogo prevista em classes não combustíveis do MCD12Q1 (urbano, neve/gelo, solo
  exposto, água). Substitui o papel de combustível do NDVI, que não foi sustentado e tem *look-ahead*.
- `far`: fração da massa de fogo **novo** (fora de `fire_t`) a mais de `radius` px (distância de Chebyshev) do fogo
  em t. No treino, ~25% dos pixels novos estão além de 10 px: o observado não é zero, e a métrica compara.
- `wind` e `upslope`: alinhamento (cosseno médio ponderado pela massa) entre a normal externa da frente em t e o
  transporte do vento GRIDMET em t ou a subida do terreno, nos pixels novos até `radius` px da frente. O estimador
  não é o do `pde_diagnostics.py` (vetor até o pixel queimando mais próximo); por isso o observado é recalculado aqui
  com o mesmo estimador do modelo.
- `monotone`: sensibilidade da probabilidade a uma perturbação de `delta` desvios-padrão em ERC (+), temperatura
  máxima (+) e precipitação (−) em todos os dias de entrada; mede a fração de pixels perto do fogo em que a
  probabilidade **cai**. Não há equivalente observado: o diagnóstico de EDP mediu associação com o crescimento da
  área, não derivada por pixel.

Convenções de grade (ver `core.py`): os ângulos de entrada estão no norte verdadeiro e, com ou sem aumento de dados,
azimute_grade = θ + γ; colunas crescem para leste da grade e linhas para o sul. O padding fora da imagem é detectado
pelo one-hot de land cover, que soma 1 dentro da imagem e 0 fora.

Todas as funções são diferenciáveis na probabilidade prevista, para servirem também de penalidade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

NONBURNABLE_CLASSES = ("urban_and_built_up_lands", "permanent_snow_and_ice", "barren", "water_bodies")
DEFAULT_RADIUS_PX = 10
DEFAULT_DELTA_STD = 0.5
# (canal, sinal esperado de ∂p/∂canal): os três com associação sustentada no crescimento (relatório de candidatas a EDP, §5).
MONOTONE_DRIVERS = (("energy_release_component", +1.0), ("temperature_max", +1.0), ("precipitation", -1.0))
MONOTONE_TOLERANCE = 1e-4
MIN_VECTOR_NORM = 1e-6


@dataclass(frozen=True)
class PhysicsLayout:
    """Posições, na entrada do modelo, dos canais que as propriedades usam."""

    wind_speed: int
    wind_speed_mean: float
    wind_speed_std: float
    wind_sin: int
    wind_cos: int
    elevation: int
    landcover: tuple[int, ...]
    nonburnable: tuple[int, ...]
    monotone: tuple[tuple[str, int, float], ...]

    @classmethod
    def from_channels(cls, channel_names: Sequence[str], normalization: dict[str, Any]) -> "PhysicsLayout":
        position = {name: k for k, name in enumerate(channel_names)}

        def need(name: str) -> int:
            if name not in position:
                raise ValueError(f"Canal {name!r} fora da entrada do modelo; as propriedades físicas precisam dele.")
            return position[name]

        stats = normalization["channels"]["wind_speed"]
        return cls(
            wind_speed=need("wind_speed"),
            wind_speed_mean=float(stats["mean"]),
            wind_speed_std=float(stats["std"]),
            wind_sin=need("wind_direction_sin"),
            wind_cos=need("wind_direction_cos"),
            elevation=need("elevation"),
            landcover=tuple(k for name, k in position.items() if name.startswith("landcover_")),
            nonburnable=tuple(need(f"landcover_{name}") for name in NONBURNABLE_CLASSES),
            monotone=tuple((name, need(name), sign) for name, sign in MONOTONE_DRIVERS),
        )


# --------------------------------------------------------------------------- campos derivados das entradas


def _plane(tensor: Any) -> Any:
    """`[B, 1, 1, P, P]` ou `[B, 1, P, P]` -> `[B, P, P]` em float32."""
    while tensor.dim() > 3:
        tensor = tensor.squeeze(1)
    return tensor.float()


def inside_image(x: Any, layout: PhysicsLayout) -> Any:
    """1 nos pixels dentro da imagem vista (o one-hot de land cover soma 1), 0 no padding. `[B, P, P]`."""
    return (x[:, list(layout.landcover), -1].float().sum(dim=1) > 0.5).float()


def proximity(fire: Any, radius: int) -> Any:
    """`radius + 1 − d` para d ≤ radius (d = distância de Chebyshev ao fogo, 0 no fogo), 0 além. `[B, P, P]`.

    Soma das dilatações 3×3 sucessivas; o gradiente aponta para o fogo, então a normal externa é `−∇`.
    """
    import torch.nn.functional as F

    current = fire.unsqueeze(1)
    total = current.clone()
    for _ in range(radius):
        current = F.max_pool2d(current, kernel_size=3, stride=1, padding=1)
        total = total + current
    return total.squeeze(1)


def gradient(field: Any, valid: Any) -> tuple[Any, Any]:
    """Diferenças centrais `(d/dcoluna, d/dlinha)`, zeradas onde algum vizinho está fora de `valid`."""
    import torch

    d_col = torch.zeros_like(field)
    d_row = torch.zeros_like(field)
    d_col[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / 2.0
    d_row[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / 2.0
    ok_col = torch.zeros_like(valid)
    ok_row = torch.zeros_like(valid)
    ok_col[:, :, 1:-1] = valid[:, :, 2:] * valid[:, :, :-2] * valid[:, :, 1:-1]
    ok_row[:, 1:-1, :] = valid[:, 2:, :] * valid[:, :-2, :] * valid[:, 1:-1, :]
    return d_col * ok_col, d_row * ok_row


def wind_transport_grid(x: Any, layout: PhysicsLayout, gamma_deg: Any) -> tuple[Any, Any]:
    """Transporte do vento no último dia de entrada, `(d_coluna, d_linha)` em m/s. `gamma_deg`: `[B]`.

    Vento "de onde vem" θ -> transporte (leste, norte) = −v·(sen θ, cos θ); depois gira para a grade por γ e inverte
    o eixo das linhas, como `core.to_grid_components`. Sem dado de vento, sen = cos = 0 e o vetor é nulo.
    """
    speed = (x[:, layout.wind_speed, -1].float() * layout.wind_speed_std + layout.wind_speed_mean).clamp(min=0.0)
    sin_t, cos_t = x[:, layout.wind_sin, -1].float(), x[:, layout.wind_cos, -1].float()
    east, north = -speed * sin_t, -speed * cos_t
    g = gamma_deg.float().view(-1, 1, 1) * (math.pi / 180.0)
    grid_east = east * g.cos() + north * g.sin()
    grid_north = -east * g.sin() + north * g.cos()
    return grid_east, -grid_north


def upslope_grid(x: Any, layout: PhysicsLayout, inside: Any) -> tuple[Any, Any]:
    """Direção de subida na grade, pelo gradiente da elevação (z-score; só a direção importa)."""
    return gradient(x[:, layout.elevation, -1].float(), inside)


# --------------------------------------------------------------------------- propriedades


def _alignment(weight: Any, normal: tuple[Any, Any], driver: tuple[Any, Any]) -> tuple[Any, Any]:
    """Σ peso·cos(normal, driver) e Σ peso nos pixels em que os dois vetores existem. `[B]` cada."""
    n_norm = (normal[0] ** 2 + normal[1] ** 2).sqrt()
    d_norm = (driver[0] ** 2 + driver[1] ** 2).sqrt()
    valid = ((n_norm > MIN_VECTOR_NORM) & (d_norm > MIN_VECTOR_NORM)).float()
    cosine = (normal[0] * driver[0] + normal[1] * driver[1]) / (n_norm * d_norm).clamp(min=MIN_VECTOR_NORM)
    w = weight * valid
    return (w * cosine).sum(dim=(1, 2)), w.sum(dim=(1, 2))


def property_sums(
    prob: Any, batch: dict[str, Any], x: Any, layout: PhysicsLayout, gamma_deg: Any, radius: int = DEFAULT_RADIUS_PX,
) -> dict[str, tuple[Any, Any]]:
    """Numerador e denominador `[B]` de cada propriedade para um campo de fogo `prob` em t+1 (previsto ou observado).

    `batch` precisa de `mask` e `fire_t` (mesmo dispositivo de `prob`). Razões são agregadas fora daqui (por evento),
    para que tiles com pouca massa não pesem como tiles com muita.
    """
    mask = _plane(batch["mask"])
    fire_t = (_plane(batch["fire_t"]) > 0.5).float()
    p = _plane(prob) * mask
    inside = inside_image(x, layout)
    nonburnable = x[:, list(layout.nonburnable), -1].float().sum(dim=1).clamp(max=1.0)
    prox = proximity(fire_t, radius)
    near = (prox > 0).float()
    new = p * (1.0 - fire_t)
    normal = tuple(-component for component in gradient(prox, inside))
    near_new = new * near
    out = {
        "nonburnable": ((p * nonburnable).sum(dim=(1, 2)), p.sum(dim=(1, 2))),
        "far": ((new * (1.0 - near)).sum(dim=(1, 2)), new.sum(dim=(1, 2))),
        "wind": _alignment(near_new, normal, wind_transport_grid(x, layout, gamma_deg)),
        "upslope": _alignment(near_new, normal, upslope_grid(x, layout, inside)),
    }
    return out


def perturb(x: Any, layout: PhysicsLayout, drivers: Sequence[str], delta: float) -> Any:
    """Cópia de `x` com cada driver deslocado de `sinal × delta` (em desvios-padrão) em todos os dias, só na imagem."""
    shifted = x.clone()
    inside = inside_image(x, layout).unsqueeze(1).to(x.dtype)
    for name, channel, sign in layout.monotone:
        if name in drivers:
            shifted[:, channel] = shifted[:, channel] + sign * delta * inside
    return shifted


def monotone_sums(
    prob: Any, prob_shifted: Any, batch: dict[str, Any], radius: int = DEFAULT_RADIUS_PX,
    tolerance: float = MONOTONE_TOLERANCE,
) -> dict[str, tuple[Any, Any]]:
    """Fração de pixels avaliados perto do fogo em que a probabilidade cai sob a perturbação "mais propícia ao fogo",
    e a variação média da probabilidade nesses pixels. `[B]` cada.

    "Perto do fogo" = até `radius` px de `fire_t`, onde a previsão tem massa; longe dele a probabilidade é ~0 e a
    derivada não informa nada.
    """
    mask = _plane(batch["mask"])
    fire_t = (_plane(batch["fire_t"]) > 0.5).float()
    region = (proximity(fire_t, radius) > 0).float() * mask
    change = _plane(prob_shifted) - _plane(prob)
    return {
        "violation": ((change < -tolerance).float().mul(region).sum(dim=(1, 2)), region.sum(dim=(1, 2))),
        "mean_change": ((change * region).sum(dim=(1, 2)), region.sum(dim=(1, 2))),
    }


# --------------------------------------------------------------------------- agregação por evento


class EventSums:
    """Acumula numeradores e denominadores por (métrica, evento) e resume com IC por bootstrap de eventos."""

    def __init__(self) -> None:
        self.sums: dict[str, dict[int, list[float]]] = {}

    def add(self, metric: str, events: Sequence[int], numerator: Sequence[float], denominator: Sequence[float]) -> None:
        table = self.sums.setdefault(metric, {})
        for event, num, den in zip(events, numerator, denominator):
            entry = table.setdefault(int(event), [0.0, 0.0])
            entry[0] += float(num)
            entry[1] += float(den)

    def arrays(self, metric: str, events: Sequence[int] | None = None) -> tuple[list[int], Any, Any]:
        import numpy as np

        table = self.sums.get(metric, {})
        keys = sorted(table) if events is None else list(events)
        num = np.array([table.get(k, [0.0, 0.0])[0] for k in keys], dtype=np.float64)
        den = np.array([table.get(k, [0.0, 0.0])[1] for k in keys], dtype=np.float64)
        return keys, num, den

    def summary(self, metric: str, *, n_boot: int = 1000, seed: int = 0, reference: str | None = None) -> dict[str, Any]:
        """Razão agregada Σnum/Σden, com IC 95% por bootstrap de eventos; com `reference`, também a diferença pareada."""
        import numpy as np

        keys, num, den = self.arrays(metric)
        if reference is not None:
            _, ref_num, ref_den = self.arrays(reference, keys)
        has = den > 0
        result: dict[str, Any] = {"events": int(len(keys)), "events_with_mass": int(has.sum()),
                                  "value": float(num.sum() / den.sum()) if den.sum() > 0 else None}
        if reference is not None:
            result["reference"] = reference
            result["reference_value"] = float(ref_num.sum() / ref_den.sum()) if ref_den.sum() > 0 else None
            if result["value"] is not None and result["reference_value"] is not None:
                result["difference"] = result["value"] - result["reference_value"]
        if len(keys) < 2 or n_boot <= 0:
            return result
        rng = np.random.default_rng(seed)
        values, differences = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(keys), len(keys))
            d = den[idx].sum()
            if d <= 0:
                continue
            value = num[idx].sum() / d
            values.append(value)
            if reference is not None:
                rd = ref_den[idx].sum()
                if rd > 0:
                    differences.append(value - ref_num[idx].sum() / rd)
        if values:
            result["ci95"] = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        if differences:
            result["difference_ci95"] = [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]
        return result
