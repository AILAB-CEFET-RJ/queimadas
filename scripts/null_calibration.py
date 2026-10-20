"""Mede e persiste o erro tipo I dos nulos usados nos diagnósticos de EDP, em `reports/null_calibration.json`.

Por que isto existe: a conclusão de EDP sobre advecção se apoia em "o nulo está calibrado", e esse número
estava **digitado à mão** num documento que afirma não digitar nenhum. O teste versionado
afirmava apenas `taxa < 0,15` com 120 simulações, o que passaria com 14% de erro tipo I e não media o regime da
execução real. Aqui a taxa é medida, com a semente e os tamanhos registrados, e o documento a lê deste artefato.

Quatro cenários, todos sob H0 verdadeiro (nenhum alinhamento real entre espalhamento e forçante):

1. `rotation_per_event` — o nulo em uso. Um ângulo por evento.
2. `rotation_per_day` — a versão anterior, com um ângulo por dia. Fica aqui para que a comparação que
   justificou a troca seja reproduzível, e não uma lembrança.
3. `circular_shift_ar1` — o nulo de variação diária, com forçante autocorrelacionada dentro do evento.
4. `circular_shift_confounded` — o mesmo nulo quando as duas séries seguem uma **terceira** série latente,
   sem elo entre si. O parecer científico aponta que o deslocamento circular não separa co-variação de causa
   comum, e este cenário mede o quanto isso custa. É a razão para tratar esse nulo como descritivo.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import write_json
from scripts.env_report import provenance
from scripts.pde_diagnostics import rotate, rotation_pvalue, shift_pvalue, weighted_mean_cosine

DEFAULT_REPORT = REPO_ROOT / "reports" / "null_calibration.json"
# Regime da execução real: 3384 evento-dias sobre 335 eventos, ou seja ~10 dias por evento.
N_EVENTS = 335
N_DAYS = 10
ALPHA = 0.05


def unit(angles: np.ndarray) -> np.ndarray:
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _weights_and_unit(spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peso (comprimento resultante) e direção unitária do espalhamento, como em `weighted_mean_cosine`."""
    norms = np.linalg.norm(spread, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    return norms, spread / safe[:, None]


def _mean_cosine(unit_spread: np.ndarray, weights: np.ndarray, driver: np.ndarray) -> float:
    """Cosseno médio ponderado, já com o espalhamento normalizado — o laço quente da calibração."""
    cosines = np.einsum("ij,ij->i", unit_spread, driver)
    total = weights.sum()
    return float((weights * cosines).sum() / total) if total > 0 else float("nan")


def rotation_pvalue_fast(spread: np.ndarray, driver: np.ndarray, groups: np.ndarray,
                         n_perm: int, rng: np.random.Generator, *, per_day: bool = False) -> dict[str, Any]:
    """Equivalente vetorizado de `rotation_pvalue`, com a opção de rotacionar por dia.

    A produção percorre eventos em Python a cada permutação, o que é irrelevante para uma chamada e
    proibitivo para centenas de milhares. `test_null_calibration.py` confere que esta versão reproduz a
    estatística observada e um p compatível com a da produção. Com `per_day=True` reproduz a versão
    anterior do nulo, que é a comparação que justificou a troca.
    """
    weights, unit_spread = _weights_and_unit(spread)
    observed = _mean_cosine(unit_spread, weights, driver)
    if per_day:
        angle_index = np.arange(len(driver))
        n_angles = len(driver)
    else:
        _, angle_index = np.unique(groups, return_inverse=True)
        n_angles = angle_index.max() + 1
    greater = smaller = 0
    for _ in range(n_perm):
        angles = rng.uniform(0, 2 * np.pi, n_angles)[angle_index]
        cos, sin = np.cos(angles), np.sin(angles)
        rotated = np.column_stack([driver[:, 0] * cos - driver[:, 1] * sin,
                                   driver[:, 0] * sin + driver[:, 1] * cos])
        value = _mean_cosine(unit_spread, weights, rotated)
        greater += value >= observed
        smaller += value <= observed
    return {
        "mean_cosine_weighted": observed,
        "p_value_two_sided": min(1.0, 2 * min(greater + 1, smaller + 1) / (n_perm + 1)),
    }


def rotation_pvalue_per_day(spread: np.ndarray, driver: np.ndarray, groups: np.ndarray,
                            n_perm: int, rng: np.random.Generator) -> dict[str, Any]:
    return rotation_pvalue_fast(spread, driver, groups, n_perm, rng, per_day=True)


def shift_pvalue_fast(spread: np.ndarray, driver: np.ndarray, groups: np.ndarray,
                      n_perm: int, rng: np.random.Generator) -> dict[str, Any]:
    """Equivalente vetorizado de `shift_pvalue`: um deslocamento circular independente por evento."""
    weights, unit_spread = _weights_and_unit(spread)
    observed = _mean_cosine(unit_spread, weights, driver)
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    sizes = np.array([len(b) for b in blocks])
    flat = np.concatenate(blocks)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    offsets = np.arange(len(flat)) - np.repeat(starts, sizes)
    exceed = 0
    for _ in range(n_perm):
        rolls = rng.integers(0, np.maximum(sizes, 1))
        # Índice deslocado dentro de cada bloco, sem laço Python por evento.
        shifted = flat[np.repeat(starts, sizes) + (offsets + np.repeat(rolls, sizes)) % np.repeat(sizes, sizes)]
        rolled = np.empty_like(driver)
        rolled[flat] = driver[shifted]
        exceed += _mean_cosine(unit_spread, weights, rolled) >= observed
    return {"mean_cosine_weighted": observed, "p_value": (exceed + 1) / (n_perm + 1)}


def ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """Série AR(1) de ângulos, para dar ao driver autocorrelação temporal dentro do evento."""
    out = np.empty(n)
    out[0] = rng.normal()
    for i in range(1, n):
        out[i] = rho * out[i - 1] + np.sqrt(1 - rho * rho) * rng.normal()
    return out


def sample_event_direction(rng: np.random.Generator, n_events: int, n_days: int, spread_noise: float = 0.5):
    """H0 realista: **as duas** séries têm direção predominante por evento, sorteadas de forma independente.

    A persistência no espalhamento é o que importa aqui. Se o espalhamento fosse isotrópico dia a dia, a
    estatística observada não teria estrutura de evento para o nulo destruir, e a versão por dia pareceria
    calibrada — o que não é o caso nos dados, onde um incêndio avança na mesma direção por vários dias.
    """
    groups = np.repeat(np.arange(n_events), n_days)
    driver = unit(np.repeat(rng.uniform(0, 2 * np.pi, n_events), n_days))
    spread_base = np.repeat(rng.uniform(0, 2 * np.pi, n_events), n_days)
    return unit(spread_base + rng.normal(0, spread_noise, groups.size)), driver, groups


def sample_ar1(rng: np.random.Generator, n_events: int, n_days: int, rho: float = 0.8):
    """H0 com forçante autocorrelacionada dentro do evento, espalhamento independente."""
    groups = np.repeat(np.arange(n_events), n_days)
    base = rng.uniform(0, 2 * np.pi, n_events)
    angles = np.concatenate([base[e] + ar1(rng, n_days, rho) for e in range(n_events)])
    spread = unit(rng.uniform(0, 2 * np.pi, groups.size))
    return spread, unit(angles), groups


def sample_confounded(rng: np.random.Generator, n_events: int, n_days: int, rho: float = 0.8,
                      noise: float = 0.4):
    """Espalhamento e forçante seguem uma **terceira** série latente, sem nenhum elo entre si.

    É o cenário que importa para a EDP: no anel da frente, o terreno faz o espalhamento migrar e o vento
    local acompanha a mesma topografia, de modo que os dois co-variam dentro do evento por causa comum. Não há
    "o vento move o fogo" aqui — e o deslocamento circular, que quebra o pareamento dia a dia mas preserva a
    co-variação, ainda assim rejeita. É por isso que esse nulo descreve e não confirma.
    """
    groups = np.repeat(np.arange(n_events), n_days)
    base = rng.uniform(0, 2 * np.pi, n_events)
    latent = np.concatenate([base[e] + ar1(rng, n_days, rho) for e in range(n_events)])
    spread = unit(latent + rng.normal(0, noise, groups.size))
    driver = unit(latent + rng.normal(0, noise, groups.size))
    return spread, driver, groups


def measure(name: str, sampler: Callable[..., Any], test: Callable[..., dict[str, Any]], key: str,
            *, n_sims: int, n_perm: int, n_events: int, n_days: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    rejections = 0
    for _ in range(n_sims):
        spread, driver, groups = sampler(rng, n_events, n_days)
        rejections += test(spread, driver, groups, n_perm, rng)[key] < ALPHA
    rate = rejections / n_sims
    # Erro padrão **sob H0 nominal**: é o yardstick certo para testar "a taxa é alpha?". Usar o SE da taxa
    # observada degenera a zero quando não houve rejeição, e marcaria como descalibrado um nulo conservador.
    null_error = float(np.sqrt(ALPHA * (1 - ALPHA) / n_sims))
    observed_error = float(np.sqrt(max(rate * (1 - rate), 1e-12) / n_sims))
    return {
        "scenario": name,
        "type_i_error_at_5pct": rate,
        "standard_error": round(observed_error, 4),
        "standard_error_under_alpha": round(null_error, 4),
        "ci95": [round(max(0.0, rate - 1.96 * observed_error), 4),
                 round(min(1.0, rate + 1.96 * observed_error), 4)],
        "calibrated": bool(rate <= ALPHA + 1.96 * null_error),
        "conservative": bool(rate < ALPHA - 1.96 * null_error),
        "n_simulations": n_sims,
        "n_permutations": n_perm,
        "n_events": n_events,
        "n_days_per_event": n_days,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mede o erro tipo I dos nulos dos diagnósticos de EDP.")
    parser.add_argument("--simulations", type=int, default=400, help="Mais simulações, menos ruído na taxa.")
    parser.add_argument("--permutations", type=int, default=400)
    parser.add_argument("--n-events", type=int, default=N_EVENTS)
    parser.add_argument("--n-days", type=int, default=N_DAYS)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    started = time.perf_counter()
    common = {"n_sims": args.simulations, "n_perm": args.permutations,
              "n_events": args.n_events, "n_days": args.n_days}
    scenarios = {
        "rotation_per_event": measure(
            "nulo em uso: um ângulo por evento, H0 com direção predominante por evento",
            sample_event_direction, rotation_pvalue_fast, "p_value_two_sided", seed=args.seed, **common),
        "rotation_per_day": measure(
            "versão anterior: um ângulo por dia, mesmo H0 — a comparação que justificou a troca",
            sample_event_direction, rotation_pvalue_per_day, "p_value_two_sided", seed=args.seed + 1, **common),
        "circular_shift_ar1": measure(
            "deslocamento circular com forçante AR(1) dentro do evento",
            sample_ar1, shift_pvalue_fast, "p_value", seed=args.seed + 2, **common),
        "circular_shift_confounded": measure(
            "deslocamento circular quando as duas séries seguem uma terceira latente — confundimento por "
            "causa comum, que este nulo não separa",
            sample_confounded, shift_pvalue_fast, "p_value", seed=args.seed + 3, **common),
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alpha": ALPHA,
        "note": (
            "Todos os cenários são sob H0 verdadeiro: nenhum alinhamento real entre espalhamento e forçante. "
            "Uma taxa acima de alpha significa que o teste rejeita mais do que deveria, e qualquer p daquele "
            "nulo está otimista na mesma proporção."
        ),
        "scenarios": scenarios,
        "conclusions": {
            "rotation_null_is_the_one_to_trust": scenarios["rotation_per_event"]["calibrated"],
            "circular_shift_is_descriptive_only": not scenarios["circular_shift_confounded"]["calibrated"],
            "why": (
                "O nulo por rotação está calibrado no regime da execução e é o que carrega a inferência. O "
                "deslocamento circular quebra o pareamento dia a dia mas preserva a co-variação dentro do "
                "evento, então rejeita também quando as duas séries seguem uma terceira causa comum: nesse "
                "cenário ele não é o nulo certo. Por isso fica fora da família Holm e vale como descrição, "
                "não como segunda confirmação. O que nenhum dos dois exclui é co-orientação regional entre "
                "vento, terreno e espalhamento."
            ),
        },
        "provenance": provenance({}),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    write_json(report, args.report)
    for key, block in scenarios.items():
        print(f"{key:30s} {block['type_i_error_at_5pct']:.3f} ± {block['standard_error']:.3f} "
              f"{'ok' if block['calibrated'] else 'NAO CALIBRADO'}")
    print(f"\nRelatório: {args.report}")
    return report


if __name__ == "__main__":
    main()
