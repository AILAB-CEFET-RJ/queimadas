"""Testes das penalidades físicas do `stconv_train_piml.py` em campos sintéticos."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from scripts.core import input_channel_names

P = 21
NAMES = input_channel_names(include_forecast_precip=True)
NORMALIZATION = {"channels": {"wind_speed": {"mean": 4.0, "std": 2.0}}}


def make_x(wind_from_deg: float) -> "torch.Tensor":
    x = torch.zeros(1, len(NAMES), 2, P, P)
    x[:, NAMES.index("landcover_grasslands")] = 1.0
    theta = math.radians(wind_from_deg)
    x[:, NAMES.index("wind_direction_sin")] = math.sin(theta)
    x[:, NAMES.index("wind_direction_cos")] = math.cos(theta)
    return x


def front_batch() -> dict[str, "torch.Tensor"]:
    fire = np.zeros((P, P), dtype=np.float32)
    fire[:, 8:10] = 1.0
    return {"mask": torch.ones(1, 1, 1, P, P), "fire_t": torch.from_numpy(fire).view(1, 1, 1, P, P)}


def east_spread() -> "torch.Tensor":
    prob = torch.zeros(1, 1, 1, P, P)
    prob[..., 3:18, 10] = 1.0
    return prob


@unittest.skipIf(torch is None, "torch ausente")
class PimlPenaltyTest(unittest.TestCase):
    def setUp(self) -> None:
        from scripts.physics import PhysicsLayout
        from scripts import stconv_train_piml as piml

        self.piml = piml
        self.layout = PhysicsLayout.from_channels(NAMES, NORMALIZATION)

    def terms(self, prob: "torch.Tensor", x: "torch.Tensor", shifted: "torch.Tensor | None" = None) -> dict:
        return self.piml.penalty_terms(prob, shifted, front_batch(), x, self.layout, torch.zeros(1), 3)

    def test_wind_penalty_charges_only_upwind_spread(self) -> None:
        self.assertAlmostEqual(float(self.terms(east_spread(), make_x(270.0))["wind"]), 0.0, places=5)  # a favor
        self.assertAlmostEqual(float(self.terms(east_spread(), make_x(90.0))["wind"]), 1.0, places=4)   # contra
        self.assertAlmostEqual(float(self.terms(east_spread(), make_x(0.0))["wind"]), 0.0, places=4)    # de lado

    def test_nonburnable_penalty(self) -> None:
        x = make_x(270.0)
        x[:, NAMES.index("landcover_grasslands"), :, 0, :] = 0.0
        x[:, NAMES.index("landcover_water_bodies"), :, 0, :] = 1.0
        prob = torch.zeros(1, 1, 1, P, P)
        prob[..., 0, :] = 1.0
        self.assertAlmostEqual(float(self.terms(prob, x)["nonburnable"]), P / (P * P), places=6)

    def test_monotone_penalty_only_when_probability_drops(self) -> None:
        prob = torch.full((1, 1, 1, P, P), 0.5)
        logits_up = torch.full((1, 1, 1, P, P), 1.0)    # σ = 0,73 > 0,5: sem violação
        logits_down = torch.full((1, 1, 1, P, P), -1.0)  # σ = 0,27 < 0,5: violação em toda a região
        self.assertAlmostEqual(float(self.terms(prob, make_x(270.0), logits_up)["monotone"]), 0.0)
        self.assertAlmostEqual(float(self.terms(prob, make_x(270.0), logits_down)["monotone"]),
                               0.5 - 1 / (1 + math.e), places=5)

    def test_penalties_are_differentiable(self) -> None:
        logits = torch.zeros(1, 1, 1, P, P, requires_grad=True)
        terms = self.terms(torch.sigmoid(logits), make_x(90.0), logits - 1.0)
        sum(terms.values()).backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_piml_arguments_are_split_from_baseline_arguments(self) -> None:
        settings, rest = self.piml.split_argv(["--recipe", "paper_es", "--lambda-wind", "0.5", "--fold", "2"])
        self.assertEqual(settings.lambda_wind, 0.5)
        self.assertEqual(settings.lambda_monotone, 0.0)
        self.assertEqual(rest, ["--recipe", "paper_es", "--fold", "2"])

    def test_run_name_must_be_prefixed(self) -> None:
        with self.assertRaises(SystemExit):
            self.piml.main(["--recipe", "paper_es", "--run-name", "baseline_es_f02", "--lambda-wind", "1"])


if __name__ == "__main__":
    unittest.main()
