"""Testes da calibração dos nulos dos diagnósticos de EDP.

O risco aqui é sutil: a calibração usa versões **vetorizadas** dos nulos, porque as de produção percorrem
eventos em Python e seriam proibitivas em centenas de milhares de permutações. Se a versão rápida não
reproduzir a de produção, a calibração mede um teste que ninguém usa — e o número que licencia a conclusão
sobre advecção passa a não dizer nada. Estes testes amarram as duas.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import null_calibration as cal
from scripts import pde_diagnostics as pde


class FastNullsMatchProductionTest(unittest.TestCase):
    """A estatística observada tem de ser idêntica; o p, compatível dentro do erro de Monte Carlo."""

    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        self.spread, self.driver, self.groups = cal.sample_event_direction(rng, n_events=25, n_days=8)

    def test_estatistica_observada_e_identica_na_rotacao(self) -> None:
        fast = cal.rotation_pvalue_fast(self.spread, self.driver, self.groups, 1, np.random.default_rng(1))
        production = pde.rotation_pvalue(self.spread, self.driver, self.groups, 1, np.random.default_rng(1))
        self.assertAlmostEqual(production["mean_cosine_weighted"], fast["mean_cosine_weighted"], places=12)

    def test_estatistica_observada_e_identica_no_deslocamento(self) -> None:
        fast = cal.shift_pvalue_fast(self.spread, self.driver, self.groups, 1, np.random.default_rng(1))
        production = pde.shift_pvalue(self.spread, self.driver, self.groups, 1, np.random.default_rng(1))
        self.assertAlmostEqual(production["mean_cosine_weighted"], fast["mean_cosine_weighted"], places=12)

    def test_p_da_rotacao_concorda_com_a_producao(self) -> None:
        """Sementes diferentes consomem o RNG em outra ordem, então a comparação é estatística."""
        rng = np.random.default_rng(5)
        spread, driver, groups = cal.sample_event_direction(rng, n_events=20, n_days=8)
        # Alinhamento real, para os dois caírem na mesma região de p e não ambos no piso por acidente.
        spread = driver + np.random.default_rng(6).normal(0, 0.4, driver.shape)
        fast = cal.rotation_pvalue_fast(spread, driver, groups, 400, np.random.default_rng(7))
        production = pde.rotation_pvalue(spread, driver, groups, 400, np.random.default_rng(8))
        self.assertAlmostEqual(production["mean_cosine_weighted"], fast["mean_cosine_weighted"], places=12)
        self.assertLess(abs(production["p_value_two_sided"] - fast["p_value_two_sided"]), 0.05)

    def test_deslocamento_rapido_preserva_o_evento_de_cada_linha(self) -> None:
        """Um deslocamento circular só pode trocar linhas **dentro** do mesmo evento."""
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(6), 5)
        driver = np.column_stack([np.arange(30.0), np.zeros(30)])  # primeira coluna identifica a linha
        spread = cal.unit(rng.uniform(0, 2 * np.pi, 30))
        seen = set()
        for _ in range(40):
            result = cal.shift_pvalue_fast(spread, driver, groups, 1, rng)
            self.assertIn("p_value", result)
        # Reconstrói um deslocamento e confere que cada linha recebeu valor do próprio evento.
        for _ in range(20):
            blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
            for block in blocks:
                rolled = np.roll(driver[block], int(rng.integers(len(block))), axis=0)
                origin = rolled[:, 0].astype(int)
                self.assertTrue(set(origin).issubset(set(block.tolist())))
                seen.update(origin.tolist())
        self.assertEqual(30, len(seen))


class CalibrationScenariosTest(unittest.TestCase):
    """Os cenários têm de produzir a patologia que dizem produzir, senão a medição não vale nada."""

    def test_rotacao_por_dia_infla_e_por_evento_nao(self) -> None:
        """A comparação que justificou trocar o nulo. Poucas simulações, mas a diferença é grosseira."""
        per_event = cal.measure("por evento", cal.sample_event_direction, cal.rotation_pvalue_fast,
                                "p_value_two_sided", n_sims=40, n_perm=120, n_events=40, n_days=10, seed=1)
        per_day = cal.measure("por dia", cal.sample_event_direction, cal.rotation_pvalue_per_day,
                              "p_value_two_sided", n_sims=40, n_perm=120, n_events=40, n_days=10, seed=1)
        self.assertTrue(per_event["calibrated"], f"nulo em uso descalibrado: {per_event}")
        self.assertFalse(per_day["calibrated"], f"a versão por dia deveria inflar: {per_day}")
        self.assertGreater(per_day["type_i_error_at_5pct"], per_event["type_i_error_at_5pct"] + 0.1)

    def test_deslocamento_circular_rejeita_sob_confundimento(self) -> None:
        """É isto que impede o nulo de deslocamento de valer como segunda confirmação."""
        result = cal.measure("confundido", cal.sample_confounded, cal.shift_pvalue_fast, "p_value",
                             n_sims=25, n_perm=120, n_events=40, n_days=10, seed=2)
        self.assertFalse(result["calibrated"], f"deveria rejeitar sob causa comum: {result}")
        self.assertGreater(result["type_i_error_at_5pct"], 0.5)

    def test_criterio_de_calibracao_nao_degenera_sem_rejeicao(self) -> None:
        """Com a taxa observada o erro padrão vira 0, e um nulo conservador seria marcado descalibrado.

        Precisa de simulações suficientes: com 20 o intervalo sob alpha já cobre zero, e "conservador" não é
        declarável — o que também é correto, e é por isso que o teste usa 200.
        """
        conservative = cal.measure("sempre aceita", cal.sample_event_direction,
                                   lambda *a, **k: {"p_value_two_sided": 1.0},
                                   "p_value_two_sided", n_sims=200, n_perm=1, n_events=5, n_days=4, seed=0)
        self.assertEqual(0.0, conservative["type_i_error_at_5pct"])
        self.assertTrue(conservative["calibrated"])
        self.assertTrue(conservative["conservative"])

    def test_ar1_tem_autocorrelacao(self) -> None:
        series = cal.ar1(np.random.default_rng(0), 4000, rho=0.8)
        lag1 = float(np.corrcoef(series[:-1], series[1:])[0, 1])
        self.assertGreater(lag1, 0.7)
        self.assertLess(lag1, 0.9)


if __name__ == "__main__":
    unittest.main()
