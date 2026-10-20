"""Gera e versiona as definições de split em `config/splits/{spatial,spatiotemporal}_split.json`.

Por que são artefatos e não cálculos implícitos: o split é determinístico dado o conjunto de eventos e os
parâmetros, mas se um dos dois mudar — um evento novo convertido, um buffer ajustado — o split muda em
silêncio, e métricas de execuções diferentes deixam de ser comparáveis sem que nada avise. Com a definição
commitada, `resolve_split` confere o recomputado contra ela e **falha** em caso de divergência.

Dois protocolos são gerados por este script:

- `spatial`: blocos geográficos sobre todos os anos. Isola **espaço**. Serve de diagnóstico da sobreposição e
  de contraparte de volume; não é o protocolo de avaliação.
- `spatiotemporal`: cruza os blocos com os anos de um fold oficial (dois de treino, um de validação, um de
  teste). É o protocolo de avaliação: o teste é lugar novo **e** período novo, e as células `future_only` e
  `space_only` decompõem de onde vem a perda.

Rodar isto de novo depois de um experimento é uma mudança metodológica: registre-a como desvio, com o motivo.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.io_utils import write_json
from scripts.core import (
    ALL_YEARS,
    DEFAULT_FOLD,
    SPATIAL_BLOCK_M,
    SPATIAL_BUFFER_M,
    SPATIAL_MIN_BLOCKS,
    SPATIAL_SHARES,
    SPATIOTEMPORAL_FOLD,
    event_footprint,
    event_window_counts,
    list_events,
    overlap_fraction,
    resolve_split,
    spatial_split,
    spatiotemporal_fold_table,
    spatiotemporal_split,
    split_years,
)
from scripts.env_report import provenance

DEFAULT_ROOT = REPO_ROOT / "dados_geoespaciais" / "wildfirespreadts" / "hdf5"
SPLITS_DIR = REPO_ROOT / "config" / "splits"


def temporal_comparison(hdf5_root: Path, fold: int) -> dict[str, Any]:
    """Contrapartida temporal, para que a comparação de volume e de sobreposição fique no mesmo arquivo.

    É o que permite dizer se uma diferença de desempenho entre protocolos pode ser atribuída a volume de treino
    em vez de generalização.
    """
    split = resolve_split(hdf5_root, "temporal", fold)
    windows = {name: int(event_window_counts(items).sum()) for name, items in split.events.items()}
    train_boxes = [event_footprint(e) for e in split.events["train"]]
    overlap = {}
    for name in ("validation", "test"):
        fractions = [overlap_fraction(event_footprint(e), train_boxes) for e in split.events[name]]
        overlap[name] = {
            "share_with_any_overlap_of_train": round(sum(f > 0 for f in fractions) / len(fractions), 4)
            if fractions else None,
            "share_overlap_gt_50pct": round(sum(f > 0.5 for f in fractions) / len(fractions), 4)
            if fractions else None,
        }
    return {
        "fold": fold,
        "years": split.metadata["years"],
        "n_events": {name: len(items) for name, items in split.events.items()},
        "windows": windows,
        "overlap_with_train": overlap,
        "note": "o split por ano separa tempo; a sobreposição acima mostra que não separa regiões",
    }


def years_by_split(events: Any, assignment: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    by_key = {e.key: e for e in events}
    result: dict[str, dict[str, int]] = {}
    for name, keys in assignment.items():
        counts: dict[str, int] = {}
        for key in keys:
            year = str(by_key[key].year)
            counts[year] = counts.get(year, 0) + 1
        result[name] = dict(sorted(counts.items()))
    return result


def build_spatial(events: Any, args: argparse.Namespace) -> dict[str, Any]:
    result = spatial_split(
        events,
        block_m=args.block_km * 1000,
        buffer_m=args.buffer_km * 1000,
        min_blocks=args.min_blocks,
    )
    return {
        "rationale": (
            "O split por ano não dá independência espacial: na EDA, 96,2% dos eventos de teste caem sobre a "
            "pegada de algum evento de treino e 66,7% com mais de metade da área. Este split separa regiões "
            "por blocos geográficos e descarta quem atravessa o corredor entre conjuntos, de modo que a "
            "independência espacial é verificada e não suposta."
        ),
        "why_blocks_and_not_clusters": (
            "A pegada mediana de um evento é de 93 km. Agrupar por sobreposição transitiva junta 98,4% das "
            "janelas num único aglomerado mesmo com buffer zero, o que não produz split algum."
        ),
        "isolates": "espaço (todos os anos aparecem em todos os conjuntos)",
        "role": (
            "diagnóstico: mede a sobreposição e serve de contraparte de volume ao protocolo espaço-temporal. "
            "Não é o protocolo de avaliação — ele treina com todos os anos, inclusive o de teste."
        ),
        "years_by_split": years_by_split(events, result["assignment"]),
        "all_years": list(ALL_YEARS),
        "shares_target": SPATIAL_SHARES,
        **result,
    }


def build_spatiotemporal(events: Any, args: argparse.Namespace) -> dict[str, Any]:
    result = spatiotemporal_split(
        events,
        fold=args.fold if args.fold is not None else SPATIOTEMPORAL_FOLD,
        block_m=args.block_km * 1000,
        buffer_m=args.buffer_km * 1000,
        min_blocks=args.min_blocks,
    )
    table = spatiotemporal_fold_table(
        events,
        block_m=args.block_km * 1000,
        buffer_m=args.buffer_km * 1000,
        min_blocks=args.min_blocks,
    )
    return {
        "rationale": (
            "Nem o split por ano nem o por bloco medem o que a aplicação enfrenta. O temporal repete regiões "
            "(96,2% dos eventos de teste sobre pegada de treino); o espacial repete anos, então o modelo já viu "
            "a estação de fogo do ano em que é avaliado. Aqui as duas separações são cruzadas: o teste é lugar "
            "novo e período novo ao mesmo tempo. É a leitura mais conservadora e a que reflete a incerteza real."
        ),
        "why_report_the_scenarios": (
            "Sozinho, o número do teste não diz de onde vem a perda. `future_only` (blocos de treino, ano de "
            "teste) mede o deslocamento temporal com a geografia familiar; `space_only` (blocos de teste, anos "
            "de treino) mede o deslocamento espacial dentro do período de treino. Os dois são avaliados com o "
            "mesmo modelo treinado na célula de treino, então a diferença entre a queda do teste e a soma das "
            "duas quedas é a interação entre os deslocamentos."
        ),
        "isolates": "espaço e tempo ao mesmo tempo (bloco × ano)",
        "fold_choice": table,
        "years_by_split": years_by_split(events, result["assignment"]),
        "all_years": list(ALL_YEARS),
        "shares_target": SPATIAL_SHARES,
        **result,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera as definições versionadas de split.")
    parser.add_argument("--kind", choices=("spatial", "spatiotemporal"), default="spatiotemporal")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=None,
                        help="Padrão: config/splits/<kind>_split.json")
    parser.add_argument("--block-km", type=float, default=SPATIAL_BLOCK_M / 1000)
    parser.add_argument("--buffer-km", type=float, default=SPATIAL_BUFFER_M / 1000)
    parser.add_argument("--min-blocks", type=int, default=SPATIAL_MIN_BLOCKS)
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold oficial do protocolo espaço-temporal (padrão: o escolhido em core).")
    parser.add_argument("--temporal-fold", type=int, default=DEFAULT_FOLD,
                        help="Fold temporal usado como comparação de volume.")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescreve uma definição existente, ou aceita um fold que não passa os gates. "
                             "Use só quando a mudança for deliberada.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report_path = args.report or SPLITS_DIR / f"{args.kind}_split.json"
    if report_path.exists() and not args.force:
        raise SystemExit(
            f"{report_path} já existe. Sobrescrever muda o split de experimentos já rodados: use --force e "
            "registre o desvio e o motivo."
        )
    events = list_events(args.data_root, ALL_YEARS)
    body = (build_spatial if args.kind == "spatial" else build_spatiotemporal)(events, args)

    if args.kind == "spatiotemporal":
        failed = [name for name, gate in body["viability"].items() if not gate["passes"]]
        if failed and not args.force:
            table = body["fold_choice"]
            raise SystemExit(
                f"O fold {body['fold']} não passa o gate de viabilidade em {failed}: "
                + "; ".join(f"{name} com {body['viability'][name]['events']} eventos e "
                            f"{body['viability'][name]['windows']} janelas" for name in failed)
                + f". Folds viáveis: {table['viable_folds']}. Use --fold ou --force com a justificativa registrada."
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **body,
        "temporal_for_comparison": temporal_comparison(args.data_root, args.temporal_fold),
        "provenance": provenance({}),
    }
    write_json(report, report_path)

    windows, verification = report["windows"], report["verification"]
    if args.kind == "spatial":
        print(f"Split espacial: treino {windows['train']} | validação {windows['validation']} | "
              f"teste {windows['test']} janelas; {report['n_dropped_events']} eventos descartados "
              f"({windows['dropped']} janelas)")
        print(f"Sobreposição teste×treino: {verification['test']['share_with_any_overlap_of_train']:.0%} "
              f"(temporal: "
              f"{report['temporal_for_comparison']['overlap_with_train']['test']['share_with_any_overlap_of_train']:.0%})")
        print(f"Folga mínima teste→treino: {verification['test']['min_gap_to_train_km']} km")
    else:
        years = report["years"]
        print(f"Split espaço-temporal, fold {report['fold']}: treino {years['train']} | "
              f"validação {years['validation']} | teste {years['test']}")
        for cell in ("train", "validation", "test", "future_only", "space_only"):
            print(f"  {cell:<12} {report['n_events'][cell]:>4} eventos  {windows[cell]:>5} janelas")
        print(f"  {'seladas':<12} {'':>4}          {windows['sealed']:>5} janelas (não usadas)")
        print(f"  {'descartadas':<12} {report['n_dropped_events']:>4} eventos  {windows['dropped']:>5} janelas "
              "(corredor de exclusão)")
        test = verification["test"]
        print(f"Teste: {test['share_with_any_overlap_of_train']:.0%} de sobreposição com treino, "
              f"folga mínima {test['min_gap_to_train_km']} km, "
              f"{test['days_from_last_train_day_to_first_day']} dias após o último dia de treino")
        print(f"Folds viáveis: {report['fold_choice']['viable_folds']}")
    print(f"Relatório: {report_path}")
    return report


if __name__ == "__main__":
    main()
