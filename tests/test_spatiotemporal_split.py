"""Testes do split espaço-temporal.

O protocolo cruza as duas separações: blocos geográficos × anos de um fold oficial. As propriedades abaixo são
o que o torna uma estimativa conservadora em vez de uma mistura inauditável — se qualquer uma cair, o número do
teste deixa de significar "lugar novo e período novo" sem que nada avise:

1. **Toda célula é uma partição**: cada evento mantido cai em exatamente uma das nove células, e nenhum aparece
   em duas. Sem isso, um evento de treino poderia ser contado como teste.
2. **O teste não toca pegada de treino** (independência espacial, herdada do corredor de exclusão) **e** começa
   depois do último dia de treino (independência temporal). As duas ao mesmo tempo são a razão de ser do split.
3. **Os dois cenários isolam um deslocamento cada**: `future_only` está no ano de teste mas na geografia de
   treino (sobreposição de pegada alta, por construção); `space_only` está no período de treino mas em blocos
   sem treino (sobreposição zero). Se um deles deslocasse as duas coisas, a decomposição da perda seria falsa.
4. **As células seladas ficam fora**: nada que esteja em (blocos de validação × ano de teste) ou em (blocos de
   teste × ano de validação) entra em treino, seleção ou avaliação.
5. **É determinístico e independente da ordem** dos eventos, porque um experimento pré-registrado não pode ter
   o conjunto de treino mudando entre execuções.
6. **O gate de viabilidade reprova células pequenas** e a tabela de folds mostra quais folds sobrevivem ao
   cruzamento — o fold do protocolo temporal não é um deles.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import core
from tests.test_spatial_split import grid_of_events

FOLD = core.SPATIOTEMPORAL_FOLD


def long_event(key: str, year: int, lng: float, lat: float, n_days: int) -> core.EventRef:
    """Evento sintético longo: datas válidas atravessando meses, para exercitar os gates de volume."""
    start = np.datetime64(f"{year}-04-01", "D")
    dates = tuple(str(start + offset) for offset in range(n_days))
    return core.EventRef(year=year, fire_name=key, path=Path(f"/fake/{year}/{key}.hdf5"),
                         n_days=n_days, height=128, width=128, dates=dates, lnglat=(lng, lat))


def blocks_of(events: list[core.EventRef], keys: list[str]) -> set[str]:
    """Rótulo de bloco de cada chave, para distinguir "mesma geografia" de "geografia nova"."""
    labels, _ = core.spatial_blocks(events)
    by_key = {event.key: label for event, label in zip(events, labels)}
    return {by_key[key] for key in keys}


class SpatiotemporalPartitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = grid_of_events()
        self.result = core.spatiotemporal_split(self.events, fold=FOLD)
        self.years = core.split_years(FOLD)

    def test_cada_evento_mantido_cai_em_exatamente_uma_celula(self) -> None:
        cells = list(self.result["assignment"].values()) + list(self.result["sealed_cells"].values())
        flat = [key for keys in cells for key in keys]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(sorted(flat + self.result["dropped"]), sorted(e.key for e in self.events))

    def test_nenhuma_celula_compartilha_evento_com_outra(self) -> None:
        for name, keys in self.result["assignment"].items():
            for other, other_keys in self.result["assignment"].items():
                if name != other:
                    self.assertFalse(set(keys) & set(other_keys), f"{name} e {other} compartilham eventos")

    def test_janelas_somam_o_total(self) -> None:
        windows = self.result["windows"]
        named = sum(windows[name] for name in self.result["assignment"])
        self.assertEqual(windows["total"], named + windows["sealed"] + windows["dropped"])

    def test_celulas_tem_os_anos_do_seu_grupo(self) -> None:
        by_key = {e.key: e for e in self.events}
        for cell, (_, year_group) in core.SPATIOTEMPORAL_CELLS.items():
            years = {by_key[k].year for k in self.result["assignment"][cell]}
            self.assertTrue(years <= set(self.years[year_group]),
                            f"{cell} tem anos {sorted(years)} fora de {self.years[year_group]}")

    def test_celulas_seladas_sao_as_quatro_combinacoes_sem_papel(self) -> None:
        self.assertEqual(4, len(self.result["sealed_cells"]))
        used = set(core.SPATIOTEMPORAL_CELLS.values())
        self.assertEqual(9, len(used) + len(self.result["sealed_cells"]))
        usados = {key for keys in self.result["assignment"].values() for key in keys}
        selados = {key for keys in self.result["sealed_cells"].values() for key in keys}
        self.assertFalse(usados & selados)


class SpatiotemporalIndependenceTest(unittest.TestCase):
    """Propriedade 2 e 3: é o que o protocolo promete medir."""

    def setUp(self) -> None:
        self.events = grid_of_events()
        self.result = core.spatiotemporal_split(self.events, fold=FOLD)
        self.by_key = {e.key: e for e in self.events}

    def _boxes(self, cell: str) -> list[tuple[float, float, float, float]]:
        return [core.event_footprint(self.by_key[k]) for k in self.result["assignment"][cell]]

    def test_teste_nao_toca_pegada_de_treino(self) -> None:
        train = self._boxes("train")
        buffer_m = self.result["parameters"]["buffer_m"]
        for box in self._boxes("test"):
            self.assertFalse(any(core.boxes_intersect(box, other, buffer_m) for other in train))
        self.assertEqual(0.0, self.result["verification"]["test"]["share_with_any_overlap_of_train"])

    def test_teste_comeca_depois_do_ultimo_dia_de_treino(self) -> None:
        test = self.result["verification"]["test"]
        self.assertTrue(test["fully_after_train"])
        self.assertGreater(test["days_from_last_train_day_to_first_day"], 0)

    def test_so_futuro_esta_no_ano_de_teste_e_na_geografia_de_treino(self) -> None:
        cell = self.result["verification"]["future_only"]
        self.assertEqual(core.split_years(FOLD)["test"], cell["years"])
        self.assertTrue(cell["fully_after_train"])
        # Nos blocos de treino: é a geografia familiar que faz este cenário isolar **tempo**. No conjunto
        # completo isso aparece como 98% de sobreposição de pegada; na malha sintética, esparsa, o que se
        # verifica é a pertinência ao mesmo grupo de blocos.
        assignment = self.result["assignment"]
        train_blocks = blocks_of(self.events, assignment["train"])
        self.assertTrue(blocks_of(self.events, assignment["future_only"]) <= train_blocks)
        self.assertFalse(blocks_of(self.events, assignment["test"]) & train_blocks)

    def test_so_locais_novos_esta_no_periodo_de_treino_e_em_blocos_sem_treino(self) -> None:
        cell = self.result["verification"]["space_only"]
        self.assertEqual(core.split_years(FOLD)["train"], cell["years"])
        self.assertFalse(cell["fully_after_train"])
        self.assertEqual(0.0, cell["share_with_any_overlap_of_train"])
        self.assertGreater(cell["min_gap_to_train_km"], 0.0)

    def test_validacao_tambem_e_independente_do_treino(self) -> None:
        validation = self.result["verification"]["validation"]
        self.assertEqual(0.0, validation["share_with_any_overlap_of_train"])
        self.assertEqual(core.split_years(FOLD)["validation"], validation["years"])


class SpatiotemporalStabilityTest(unittest.TestCase):
    def test_e_deterministico(self) -> None:
        events = grid_of_events()
        first = core.spatiotemporal_split(events, fold=FOLD)
        second = core.spatiotemporal_split(events, fold=FOLD)
        self.assertEqual(first["assignment"], second["assignment"])

    def test_ordem_dos_eventos_nao_muda_as_celulas(self) -> None:
        events = grid_of_events()
        direct = core.spatiotemporal_split(events, fold=FOLD)["assignment"]
        reversed_order = core.spatiotemporal_split(list(reversed(events)), fold=FOLD)["assignment"]
        self.assertEqual(direct, reversed_order)

    def test_fold_diferente_move_as_celulas_mas_mantem_os_blocos(self) -> None:
        events = grid_of_events()
        a = core.spatiotemporal_split(events, fold=1)   # teste em 2020
        b = core.spatiotemporal_split(events, fold=FOLD)  # teste em 2021
        self.assertEqual(a["blocks"]["split_by_block"], b["blocks"]["split_by_block"])
        self.assertNotEqual(a["assignment"]["test"], b["assignment"]["test"])


class ViabilityGateTest(unittest.TestCase):
    """Propriedade 6: uma célula pequena demais tem de ser reprovada, não silenciosamente aceita."""

    def test_gate_reprova_validacao_pequena(self) -> None:
        events = grid_of_events()
        result = core.spatiotemporal_split(events, fold=FOLD)
        # A malha sintética tem poucas janelas por evento, então nenhum conjunto alcança o gate real.
        self.assertFalse(result["viability"]["validation"]["passes"])
        self.assertEqual(core.SPATIOTEMPORAL_GATES["validation"], result["viability"]["validation"]["gate"])

    def test_gate_aprova_quando_a_celula_e_grande(self) -> None:
        # Malha densa de eventos longos, com os anos alternados posição a posição para que cada grupo de blocos
        # receba os quatro anos: é a situação em que as células têm eventos e janelas de sobra.
        events = [
            long_event(f"fire_{i}", core.ALL_YEARS[i % len(core.ALL_YEARS)],
                       -124.0 + 2.0 * (i % 16), 32.0 + 2.0 * (i // 16), 50)
            for i in range(256)
        ]
        result = core.spatiotemporal_split(
            events, fold=FOLD, shares={"train": 0.3, "validation": 0.3, "test": 0.4})
        passes = {name: cell["passes"] for name, cell in result["viability"].items()}
        self.assertEqual({"validation": True, "test": True}, passes)
        for name, gate in core.SPATIOTEMPORAL_GATES.items():
            self.assertGreaterEqual(result["viability"][name]["events"], gate["events"])
            self.assertGreaterEqual(result["viability"][name]["windows"], gate["windows"])

    def test_tabela_de_folds_cobre_os_doze_e_marca_os_viaveis(self) -> None:
        table = core.spatiotemporal_fold_table(grid_of_events())
        self.assertEqual(12, len(table["folds"]))
        for entry in table["folds"].values():
            self.assertEqual(set(core.SPATIOTEMPORAL_CELLS), set(entry["cells"]))
            self.assertIsInstance(entry["passes_gates"], bool)
        self.assertEqual(core.SPATIOTEMPORAL_FOLD, table["chosen"])
        self.assertEqual([f for f, e in table["folds"].items() if e["passes_gates"]],
                         [str(f) for f in table["viable_folds"]])

    def test_tabela_de_folds_concorda_com_o_split_celula_por_celula(self) -> None:
        events = grid_of_events()
        table = core.spatiotemporal_fold_table(events)
        result = core.spatiotemporal_split(events, fold=FOLD)
        for name, cell in table["folds"][str(FOLD)]["cells"].items():
            self.assertEqual(result["cells"][name]["windows"], cell["windows"], name)
            self.assertEqual(result["cells"][name]["events"], cell["events"], name)


class ResolveSpatiotemporalTest(unittest.TestCase):
    """`resolve_split` com eventos sintéticos: o caminho que o treino e a EDA usam."""

    def setUp(self) -> None:
        self.events = grid_of_events()
        self.patch = mock.patch.object(core, "list_events", return_value=self.events)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_fold_padrao_e_o_do_protocolo(self) -> None:
        spec = core.resolve_split("/fake", "spatiotemporal")
        self.assertEqual(core.SPATIOTEMPORAL_FOLD, spec.fold)
        self.assertEqual(set(core.SPATIOTEMPORAL_CELLS), set(spec.events))

    def test_definicao_divergente_e_erro(self) -> None:
        spec = core.resolve_split("/fake", "spatiotemporal")
        definition = {"assignment": {name: [e.key for e in items][:-1]
                                     for name, items in spec.events.items()}}
        with self.assertRaises(ValueError) as caught:
            core.resolve_split("/fake", "spatiotemporal", definition=definition)
        self.assertIn("spatiotemporal", str(caught.exception))

    def test_definicao_igual_passa(self) -> None:
        spec = core.resolve_split("/fake", "spatiotemporal")
        definition = {"assignment": {name: sorted(e.key for e in items)
                                     for name, items in spec.events.items()}}
        core.resolve_split("/fake", "spatiotemporal", definition=definition)  # não levanta

    def test_normalizacao_registra_as_chaves_da_celula_de_treino(self) -> None:
        spec = core.resolve_split("/fake", "spatiotemporal")
        payload = core.split_normalization_payload(core.StreamingChannelStats(), spec)
        self.assertEqual("spatiotemporal", payload["split_kind"])
        self.assertEqual(core.SPATIOTEMPORAL_FOLD, payload["fold"])
        self.assertEqual(core.split_years(FOLD)["train"], payload["fit_years"])
        self.assertEqual(sorted(e.key for e in spec.events["train"]), payload["train_event_keys"])
        core.check_normalization(payload, spec)  # não levanta

    def test_normalizacao_do_fold_temporal_nao_serve_ao_espaco_temporal(self) -> None:
        spec = core.resolve_split("/fake", "spatiotemporal")
        with self.assertRaises(ValueError):
            core.check_normalization({"split_kind": "temporal", "train_event_keys": []}, spec)


if __name__ == "__main__":
    unittest.main()
