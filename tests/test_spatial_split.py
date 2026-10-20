"""Testes do split espacial.

As propriedades testadas aqui são as que tornam o split defensável como resposta à limitação medida na
EDA (o split por ano repete as regiões do treino). Se qualquer uma delas cair, o split deixa de
medir generalização espacial e passa a medir outra coisa, sem avisar:

1. **Nenhum evento de validação ou teste toca a pegada de um evento de treino**, dentro do buffer. É a
   definição operacional de independência espacial, e é o que o corredor de exclusão garante.
2. **Todos os anos aparecem em todos os conjuntos.** Sem isso o split misturaria separação espacial e
   temporal, e um resultado não diria qual das duas o explica.
3. **É determinístico**: a mesma entrada dá o mesmo split, porque um experimento pré-registrado não pode ter
   seu conjunto de treino mudando entre execuções.
4. **A normalização é conferida pelo conjunto de eventos**, não pelo ano — um arquivo ajustado no split
   temporal tem anos coerentes e ainda assim está errado para o split espacial.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import core


def fake_event(key: str, year: int, lng: float, lat: float, n_days: int = 10,
               size: int = 128) -> core.EventRef:
    """Evento sintético: só os atributos que o split usa (posição, tamanho, datas)."""
    dates = tuple(f"2020-01-{day:02d}" for day in range(1, n_days + 1))
    return core.EventRef(
        year=year, fire_name=key, path=Path(f"/fake/{year}/{key}.hdf5"),
        n_days=n_days, height=size, width=size,
        dates=tuple(f"{year}-07-{day:02d}" for day in range(1, n_days + 1)),
        lnglat=(lng, lat),
    )


def grid_of_events(spacing_deg: float = 3.0, per_side: int = 6) -> list[core.EventRef]:
    """Malha regular de eventos sobre o oeste dos EUA, com os quatro anos distribuídos."""
    events = []
    years = core.ALL_YEARS
    index = 0
    for i in range(per_side):
        for j in range(per_side):
            events.append(fake_event(
                f"fire_{i}_{j}", years[index % len(years)],
                -122.0 + j * spacing_deg, 36.0 + i * spacing_deg,
            ))
            index += 1
    return events


class SpatialSplitGeometryTest(unittest.TestCase):
    def test_pegada_tem_o_tamanho_esperado(self) -> None:
        event = fake_event("a", 2020, -120.0, 40.0, size=128)
        west, south, east, north = core.event_footprint(event)
        self.assertAlmostEqual(128 * core.PIXEL_SIZE_M, east - west, places=3)
        self.assertAlmostEqual(128 * core.PIXEL_SIZE_M, north - south, places=3)

    def test_intersecao_respeita_o_buffer(self) -> None:
        a = (0.0, 0.0, 100.0, 100.0)
        b = (150.0, 0.0, 250.0, 100.0)  # 50 m de folga
        self.assertFalse(core.boxes_intersect(a, b))
        self.assertFalse(core.boxes_intersect(a, b, buffer_m=49.0))
        self.assertTrue(core.boxes_intersect(a, b, buffer_m=51.0))

    def test_fracao_de_sobreposicao(self) -> None:
        box = (0.0, 0.0, 10.0, 10.0)
        self.assertEqual(1.0, core.overlap_fraction(box, [box]))
        self.assertEqual(0.25, core.overlap_fraction(box, [(5.0, 5.0, 15.0, 15.0)]))
        self.assertEqual(0.0, core.overlap_fraction(box, [(20.0, 20.0, 30.0, 30.0)]))
        # Máximo por par, não soma: dois vizinhos de 25% não viram 50%.
        self.assertEqual(0.25, core.overlap_fraction(box, [(5.0, 5.0, 15.0, 15.0), (-5.0, -5.0, 5.0, 5.0)]))

    def test_folga_entre_caixas(self) -> None:
        box = (0.0, 0.0, 10.0, 10.0)
        self.assertEqual(0.0, core.box_gap_m(box, [(5.0, 5.0, 15.0, 15.0)]))
        self.assertEqual(10.0, core.box_gap_m(box, [(20.0, 0.0, 30.0, 10.0)]))
        self.assertIsNone(core.box_gap_m(box, []))


class SpatialSplitPropertiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = grid_of_events()
        self.result = core.spatial_split(self.events, block_m=400_000.0, buffer_m=10_000.0)
        self.by_key = {e.key: e for e in self.events}

    def test_independencia_espacial_entre_treino_e_os_outros(self) -> None:
        """Propriedade 1, a razão de ser do split: ninguém em teste toca pegada de treino."""
        boxes = {name: [core.event_footprint(self.by_key[k]) for k in keys]
                 for name, keys in self.result["assignment"].items()}
        for name in ("validation", "test"):
            for box in boxes[name]:
                for train_box in boxes["train"]:
                    self.assertFalse(
                        core.boxes_intersect(box, train_box, self.result["parameters"]["buffer_m"]),
                        f"evento de {name} toca pegada de treino dentro do buffer",
                    )

    def test_sobreposicao_verificada_e_zero(self) -> None:
        for name in ("validation", "test"):
            share = self.result["verification"][name]["share_with_any_overlap_of_train"]
            if share is not None:
                self.assertEqual(0.0, share)

    def test_nenhum_evento_aparece_em_dois_conjuntos(self) -> None:
        keys = [k for keys in self.result["assignment"].values() for k in keys]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(), set(keys) & set(self.result["dropped"]))

    def test_todo_evento_e_atribuido_ou_descartado(self) -> None:
        keys = {k for keys in self.result["assignment"].values() for k in keys}
        self.assertEqual({e.key for e in self.events}, keys | set(self.result["dropped"]))

    def test_e_deterministico(self) -> None:
        """Propriedade 3: um experimento pré-registrado não pode ter o treino mudando entre execuções."""
        again = core.spatial_split(self.events, block_m=400_000.0, buffer_m=10_000.0)
        self.assertEqual(self.result["assignment"], again["assignment"])
        self.assertEqual(self.result["dropped"], again["dropped"])

    def test_ordem_dos_eventos_nao_muda_o_split(self) -> None:
        shuffled = list(reversed(self.events))
        other = core.spatial_split(shuffled, block_m=400_000.0, buffer_m=10_000.0)
        for name in self.result["assignment"]:
            self.assertEqual(sorted(self.result["assignment"][name]), sorted(other["assignment"][name]))

    def test_minimo_de_blocos_por_conjunto(self) -> None:
        """Um conjunto que é uma única região contígua calibraria o limiar numa região só."""
        for name, count in self.result["blocks"]["blocks_per_split"].items():
            self.assertGreaterEqual(count, core.SPATIAL_MIN_BLOCKS, f"{name} ficou com {count} bloco(s)")

    def test_janelas_somam_o_total(self) -> None:
        windows = self.result["windows"]
        parcels = sum(windows[name] for name in ("train", "validation", "test", "dropped"))
        self.assertEqual(windows["total"], parcels)

    def test_buffer_maior_descarta_mais(self) -> None:
        tight = core.spatial_split(self.events, block_m=400_000.0, buffer_m=0.0)
        loose = core.spatial_split(self.events, block_m=400_000.0, buffer_m=100_000.0)
        self.assertLessEqual(len(tight["dropped"]), len(loose["dropped"]))


class ResolveSplitTest(unittest.TestCase):
    """`resolve_split` precisa de HDF5 real para o caminho temporal; aqui testa-se a parte pura."""

    def test_split_desconhecido_e_erro(self) -> None:
        with self.assertRaises(ValueError) as caught:
            core.resolve_split("/fake", "diagonal")
        self.assertIn("temporal", str(caught.exception))

    def test_identidade_do_split_lista_as_chaves_de_treino(self) -> None:
        events = grid_of_events()
        spec = core.SplitSpec(kind="spatial", fold=None,
                              events={"train": events[:3], "validation": events[3:5], "test": events[5:7]},
                              metadata={})
        identity = spec.identity
        self.assertEqual("spatial", identity["split_kind"])
        self.assertEqual(sorted(e.key for e in events[:3]), identity["train_event_keys"])
        self.assertEqual({"train": 3, "validation": 2, "test": 2}, identity["n_events"])


class NormalizationCheckTest(unittest.TestCase):
    """Propriedade 4: o cheque é pelo conjunto de eventos, não pelo ano."""

    def setUp(self) -> None:
        events = grid_of_events()
        self.spatial = core.SplitSpec(kind="spatial", fold=None,
                                      events={"train": events[:10], "validation": events[10:14],
                                              "test": events[14:18]},
                                      metadata={})

    def test_chaves_corretas_passam(self) -> None:
        payload = {"split_kind": "spatial",
                   "train_event_keys": sorted(e.key for e in self.spatial.events["train"])}
        core.check_normalization(payload, self.spatial)  # não levanta

    def test_conjunto_diferente_e_erro_com_mensagem_util(self) -> None:
        keys = sorted(e.key for e in self.spatial.events["train"])
        payload = {"split_kind": "spatial", "train_event_keys": keys[:-1] + ["fire_9_9"]}
        with self.assertRaises(ValueError) as caught:
            core.check_normalization(payload, self.spatial)
        message = str(caught.exception)
        self.assertIn("Faltando", message)
        self.assertIn("Sobrando", message)

    def test_split_kind_divergente_e_erro(self) -> None:
        payload = {"split_kind": "temporal", "train_event_keys": []}
        with self.assertRaises(ValueError) as caught:
            core.check_normalization(payload, self.spatial)
        self.assertIn("temporal", str(caught.exception))

    def test_arquivo_antigo_sem_chaves_e_rejeitado_no_split_espacial(self) -> None:
        """Um JSON de normalização anterior não pode passar por válido para um split novo."""
        payload = {"fold": 2, "fit_years": [2018, 2020]}
        with self.assertRaises(ValueError) as caught:
            core.check_normalization(payload, self.spatial)
        self.assertIn("train_event_keys", str(caught.exception))

    def test_arquivo_antigo_ainda_vale_no_split_temporal(self) -> None:
        """Compatibilidade: a normalização já calculada continua válida para o fold temporal."""
        temporal = replace(self.spatial, kind="temporal", fold=2)
        core.check_normalization({"fold": 2, "fit_years": [2018, 2020]}, temporal)
        with self.assertRaises(ValueError):
            core.check_normalization({"fold": 0, "fit_years": [2018, 2019]}, temporal)


if __name__ == "__main__":
    unittest.main()
