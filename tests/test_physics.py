"""Testes das propriedades físicas (`scripts/physics.py`) em campos sintéticos com resposta conhecida."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from scripts.core import input_channel_names
from scripts.physics import (
    EventSums,
    PhysicsLayout,
    monotone_sums,
    perturb,
    property_sums,
    proximity,
    upslope_grid,
    wind_transport_grid,
    inside_image,
)

P = 21
NAMES = input_channel_names(include_forecast_precip=True)
NORMALIZATION = {"channels": {"wind_speed": {"mean": 4.0, "std": 2.0}}}


def make_x(wind_from_deg: float | None = None, speed: float = 4.0) -> "torch.Tensor":
    """Entrada `[1, C, 2, P, P]` com grassland em toda a imagem e vento uniforme opcional."""
    x = torch.zeros(1, len(NAMES), 2, P, P)
    x[:, NAMES.index("landcover_grasslands")] = 1.0
    if wind_from_deg is not None:
        theta = math.radians(wind_from_deg)
        x[:, NAMES.index("wind_speed")] = (speed - 4.0) / 2.0
        x[:, NAMES.index("wind_direction_sin")] = math.sin(theta)
        x[:, NAMES.index("wind_direction_cos")] = math.cos(theta)
    return x


def fields(fire_t: np.ndarray) -> dict[str, "torch.Tensor"]:
    return {
        "mask": torch.ones(1, 1, 1, P, P),
        "fire_t": torch.from_numpy(fire_t.astype(np.float32)).view(1, 1, 1, P, P),
    }


def plane(points: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((P, P), dtype=np.float32)
    for r, c in points:
        out[r, c] = 1.0
    return out


@unittest.skipIf(torch is None, "torch ausente")
class PhysicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = PhysicsLayout.from_channels(NAMES, NORMALIZATION)
        self.gamma = torch.zeros(1)

    def ratio(self, prob: np.ndarray, fire_t: np.ndarray, x: "torch.Tensor", name: str, radius: int = 3) -> float:
        prob_t = torch.from_numpy(prob).view(1, 1, 1, P, P)
        num, den = property_sums(prob_t, fields(fire_t), x, self.layout, self.gamma, radius)[name]
        return float(num[0] / den[0])

    def test_proximity_is_chebyshev(self) -> None:
        fire = torch.from_numpy(plane([(10, 10)])).unsqueeze(0)
        prox = proximity(fire, 2)[0]
        self.assertEqual(float(prox[10, 10]), 3.0)
        self.assertEqual(float(prox[11, 11]), 2.0)
        self.assertEqual(float(prox[12, 8]), 1.0)
        self.assertEqual(float(prox[13, 10]), 0.0)

    def test_nonburnable_share(self) -> None:
        x = make_x()
        water = NAMES.index("landcover_water_bodies")
        x[:, NAMES.index("landcover_grasslands"), :, 0, 0] = 0.0
        x[:, water, :, 0, 0] = 1.0
        prob = plane([(0, 0), (5, 5), (6, 6), (7, 7)])
        self.assertAlmostEqual(self.ratio(prob, plane([(5, 5)]), x, "nonburnable"), 0.25)

    def test_far_counts_only_new_mass_beyond_radius(self) -> None:
        fire = plane([(10, 10)])
        near = plane([(10, 10), (11, 11)])  # o pixel em fogo em t não é "novo"
        far = plane([(10, 10), (10, 18)])
        self.assertAlmostEqual(self.ratio(near, fire, make_x(), "far"), 0.0)
        self.assertAlmostEqual(self.ratio(far, fire, make_x(), "far"), 1.0)

    def test_wind_alignment_sign(self) -> None:
        fire = np.zeros((P, P), dtype=np.float32)
        fire[:, 8:10] = 1.0
        downwind_east = np.zeros((P, P), dtype=np.float32)
        downwind_east[3:18, 10] = 1.0
        # Vento de oeste (270°) transporta para leste: espalhamento a leste alinha (+1); vento de leste, −1.
        self.assertAlmostEqual(self.ratio(downwind_east, fire, make_x(270.0), "wind"), 1.0, places=4)
        self.assertAlmostEqual(self.ratio(downwind_east, fire, make_x(90.0), "wind"), -1.0, places=4)
        # Vento de norte (0°) é perpendicular a uma frente que avança para leste.
        self.assertAlmostEqual(self.ratio(downwind_east, fire, make_x(0.0), "wind"), 0.0, places=4)

    def test_wind_rotates_with_gamma(self) -> None:
        # Transporte para leste verdadeiro com γ = 90°: azimute na grade 180°, ou seja, para o sul (+linha).
        d_col, d_row = wind_transport_grid(make_x(270.0, speed=6.0), self.layout, torch.tensor([90.0]))
        self.assertAlmostEqual(float(d_col[0, 5, 5]), 0.0, places=5)
        self.assertAlmostEqual(float(d_row[0, 5, 5]), 6.0, places=5)

    def test_no_wind_data_is_excluded(self) -> None:
        fire = np.zeros((P, P), dtype=np.float32)
        fire[:, 8:10] = 1.0
        prob = plane([(5, 10)])
        num, den = property_sums(torch.from_numpy(prob).view(1, 1, 1, P, P), fields(fire), make_x(None),
                                 self.layout, self.gamma, 3)["wind"]
        self.assertEqual(float(den[0]), 0.0)

    def test_upslope_points_uphill_and_ignores_padding(self) -> None:
        x = make_x()
        x[:, NAMES.index("elevation")] = torch.arange(P, dtype=torch.float32).view(1, 1, 1, P)
        x[:, NAMES.index("landcover_grasslands"), :, :, 15:] = 0.0  # padding à direita
        inside = inside_image(x, self.layout)
        d_col, d_row = upslope_grid(x, self.layout, inside)
        self.assertAlmostEqual(float(d_col[0, 5, 5]), 1.0)
        self.assertAlmostEqual(float(d_row[0, 5, 5]), 0.0)
        self.assertEqual(float(d_col[0, 5, 14]), 0.0)  # vizinho fora da imagem

    def test_perturb_moves_drivers_with_expected_sign_inside_image(self) -> None:
        x = make_x()
        x[:, NAMES.index("landcover_grasslands"), :, :, 15:] = 0.0
        shifted = perturb(x, self.layout, ("energy_release_component", "precipitation"), 0.5)
        erc, pr, tmax = (NAMES.index(n) for n in ("energy_release_component", "precipitation", "temperature_max"))
        self.assertAlmostEqual(float(shifted[0, erc, 0, 3, 3]), 0.5)
        self.assertAlmostEqual(float(shifted[0, pr, 1, 3, 3]), -0.5)
        self.assertEqual(float(shifted[0, tmax, 0, 3, 3]), 0.0)
        self.assertEqual(float(shifted[0, erc, 0, 3, 17]), 0.0)
        self.assertEqual(float(x[0, erc, 0, 3, 3]), 0.0)  # não altera a original

    def test_monotone_violation_counts_drops_near_fire(self) -> None:
        fire = plane([(10, 10)])
        prob = torch.full((1, 1, 1, P, P), 0.5)
        shifted = prob.clone()
        shifted[..., 10, 11] = 0.2  # cai perto do fogo: violação
        shifted[..., 0, 0] = 0.1    # cai longe: fora da região
        out = monotone_sums(prob, shifted, fields(fire), radius=1)
        num, den = out["violation"]
        self.assertEqual(float(den[0]), 9.0)
        self.assertEqual(float(num[0]), 1.0)

    def test_event_sums_pool_and_pair(self) -> None:
        sums = EventSums()
        sums.add("a.model", [0, 1, 1], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        sums.add("a.observed", [0, 1], [0.0, 1.0], [2.0, 4.0])
        result = sums.summary("a.model", n_boot=50, reference="a.observed")
        self.assertAlmostEqual(result["value"], 0.5)
        self.assertAlmostEqual(result["reference_value"], 1.0 / 6.0)
        self.assertAlmostEqual(result["difference"], 0.5 - 1.0 / 6.0)
        self.assertIn("difference_ci95", result)

    def test_properties_are_differentiable(self) -> None:
        fire = plane([(10, 10)])
        logits = torch.zeros(1, 1, 1, P, P, requires_grad=True)
        num, den = property_sums(torch.sigmoid(logits), fields(fire), make_x(270.0), self.layout, self.gamma, 3)["far"]
        (num / den).sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
