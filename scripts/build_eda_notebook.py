"""Gera `notebooks/04_analise_exploratoria.ipynb`.

O notebook é somente leitura: carrega `reports/eda_summary.json` e as figuras
produzidas por `scripts/eda.py`, sem reprocessar o dataset (roda offline e com pouca RAM).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "04_analise_exploratoria.ipynb"


def md(cell_id: str, source: str):
    cell = new_markdown_cell(dedent(source).strip())
    cell["id"] = cell_id
    return cell


def code(cell_id: str, source: str):
    cell = new_code_cell(dedent(source).strip())
    cell["id"] = cell_id
    return cell


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "wsts-eda-title",
            """
            # WildfireSpreadTS — análise exploratória

            Resumo da EDA em streaming (`scripts/eda.py`). **Protocolo:** o fold é fixado antes da EDA;
            tudo o que envolve o alvo ou orienta decisões usa **somente os anos de treino**; validação e teste aparecem
            apenas no inventário (o teste sem o canal de fogo). Decisões e limitações: `README.md`.
            """,
        ),
        code(
            "wsts-eda-setup",
            """
            import json
            from pathlib import Path

            import pandas as pd
            from IPython.display import Image, Markdown, display

            def find_root():
                for candidate in (Path.cwd(), *Path.cwd().parents):
                    if (candidate / 'reports' / 'eda_summary.json').exists():
                        return candidate
                raise FileNotFoundError('Rode scripts/eda.py antes de abrir este notebook.')

            ROOT = find_root()
            summary = json.loads((ROOT / 'reports' / 'eda_summary.json').read_text(encoding='utf-8'))
            protocol = summary['protocol']
            figures = summary.get('figures', {})

            def show_figure(name):
                path = figures.get(name)
                if path and (ROOT / path).exists():
                    display(Image(ROOT / path))
                else:
                    display(Markdown(f'_Figura `{name}` não gerada nesta execução._'))

            display(Markdown(
                f"**Fold {protocol['fold']}** — treino {protocol['train_years']}, validação {protocol['validation_years']}, "
                f"teste {protocol['test_years']} · T_in = {protocol['t_in']} · gerado em {summary['generated_at_utc']} · "
                f"commit `{(summary['provenance']['repo']['commit'] or 'desconhecido')[:7]}`"
            ))
            """,
        ),
        md("wsts-eda-inventory-md", "## 1. Inventário (todos os anos, sem alvo)"),
        code(
            "wsts-eda-inventory",
            """
            inventory = pd.DataFrame(summary['inventory']['per_year']).T
            display(inventory[['role', 'events', 'images', 'events_with_date_gaps', 'windows_contiguous', 'windows_official']])
            print('Menor imagem (H, W):', summary['inventory']['min_height'], summary['inventory']['min_width'],
                  '| maior imagem (pixels):', summary['inventory']['max_pixels'])
            show_figure('wsts_inventario_por_ano')
            show_figure('wsts_sazonalidade')
            show_figure('wsts_tamanho_eventos')
            """,
        ),
        md("wsts-eda-missing-md", "## 2. Dados faltantes, faixas físicas e médias por ano (sem alvo)"),
        code(
            "wsts-eda-missing",
            """
            rows = []
            for year, channels in summary['channels_by_year'].items():
                for name, entry in channels.items():
                    rows.append({'ano': year, 'canal': name, 'NaN (%)': 100 * (entry['missing_rate'] or 0),
                                 'média': entry.get('mean'), 'fora da faixa': entry['out_of_range']})
            channels = pd.DataFrame(rows)
            display(channels.pivot(index='canal', columns='ano', values='NaN (%)').round(2))
            display(channels.pivot(index='canal', columns='ano', values='média').round(4))
            display(channels[channels['fora da faixa'] > 0])
            show_figure('wsts_faltantes_por_canal')
            show_figure('wsts_histogramas_canais')
            """,
        ),
        md(
            "wsts-eda-spatial-md",
            """
            ## 3. Distribuição espacial

            Os três mapas abaixo respondem a perguntas diferentes:

            - **Pegada dos eventos** — onde cada incêndio aconteceu e a que conjunto ele pertence. A sobreposição
              entre cores é a evidência visual de que o split por ano **não** separa regiões: 96% dos eventos de
              teste caem sobre a pegada de algum evento de treino.
            - **Atividade de fogo** — quanto cada evento queimou. A contagem de pixels de fogo é o alvo, então o
              mapa exclui **toda célula de avaliação de qualquer protocolo** (o ano de teste do fold, o teste
              espacial e as três células do espaço-temporal, inclusive `só locais novos`, que é do período de
              treino); esses eventos aparecem só como localização no primeiro mapa.
            - **Forçantes no espaço** — a média por evento de cada variável, que expõe o gradiente geográfico
              (elevação, secura, vento) por trás dos incêndios.
            """,
        ),
        code(
            "wsts-eda-spatial",
            """
            show_figure('wsts_mapa_eventos')
            show_figure('wsts_mapa_atividade_fogo')
            show_figure('wsts_mapa_variaveis')

            aggregates = summary.get('event_aggregates', {})
            inventory_rows = summary['inventory']['events']
            if aggregates and inventory_rows and 'lon' in inventory_rows[0]:
                frame = pd.DataFrame([
                    {'evento': r['key'], 'ano': r['year'], 'papel': protocol['role_by_year'][str(r['year'])],
                     'lon': round(r['lon'], 3), 'lat': round(r['lat'], 3), 'dias': r['n_days'],
                     **{k: v for k, v in aggregates.get(r['key'], {}).items() if k != 'n_days'}}
                    for r in inventory_rows
                ])
                display(Markdown('**Extremos geográficos e os 5 eventos que mais queimaram**'))
                display(frame[['lon', 'lat']].agg(['min', 'max']).round(2))
                if frame['fire_pixels'].notna().any():
                    display(frame.nlargest(5, 'fire_pixels')[
                        ['evento', 'papel', 'lon', 'lat', 'dias', 'fire_pixels']])
            """,
        ),
        md(
            "wsts-eda-split-md",
            """
            ## 4. Protocolo de split: espaço, tempo e os dois cenários

            Três protocolos convivem no resumo, e cada um responde a uma pergunta diferente:

            - **temporal** (fold 2, por ano) — "generaliza para um ano novo?". É o único comparável com o
              benchmark publicado, e o que **não** separa regiões: 96% dos eventos de teste caem sobre a pegada
              de algum evento de treino.
            - **espacial** (blocos de 600 km, todos os anos) — diagnóstico da sobreposição e contraparte de
              volume. Não é protocolo de avaliação, porque treina com o ano de teste.
            - **espaço-temporal** (bloco × ano) — o protocolo de avaliação. O teste é **lugar novo e período
              novo** ao mesmo tempo: blocos sem nenhum evento de treino, no ano mais recente.

            A grade 3×3 abaixo mostra as nove células do cruzamento. Além de treino, validação e teste, duas
            células são avaliadas com o **mesmo** modelo e existem para dizer de onde vem a perda:
            **só futuro** (blocos de treino, ano de teste) isola o deslocamento temporal com a geografia
            familiar, e **só locais novos** (blocos de teste, anos de treino) isola o deslocamento espacial
            dentro do período de treino. O que a queda do teste tiver além da soma das duas é a interação entre
            os deslocamentos. As quatro células restantes ficam **seladas**: usar (blocos de validação × ano de
            teste) para escolher época ou limiar calibraria o modelo no período de teste.
            """,
        ),
        code(
            "wsts-eda-split",
            """
            comparison = summary['split_comparison']
            st = summary['spatiotemporal_split']
            display(Markdown(
                f"Protocolo espaço-temporal: **fold {st['fold']}** — treino {st['years']['train']}, "
                f"validação {st['years']['validation']}, teste {st['years']['test']}; blocos de "
                f"{st['parameters']['block_m'] / 1000:.0f} km com corredor de "
                f"{st['parameters']['buffer_m'] / 1000:.0f} km"
            ))
            show_figure('wsts_mapa_splits')
            show_figure('wsts_grade_espacotemporal')
            show_figure('wsts_mapa_blocos_espaciais')
            show_figure('wsts_janelas_por_split')

            rotulos = {'train': 'treino', 'validation': 'validação', 'test': 'teste (lugar e período novos)',
                       'future_only': 'só futuro', 'space_only': 'só locais novos'}
            cells = pd.DataFrame([
                {'célula': rotulos[name],
                 'blocos': spec['blocks'], 'anos': spec['years'],
                 'eventos': spec['events'], 'janelas': spec['windows'],
                 'sobreposição com treino': (st['verification'].get(name) or {}).get(
                     'share_with_any_overlap_of_train'),
                 'folga mínima (km)': (st['verification'].get(name) or {}).get('min_gap_to_train_km'),
                 'dias após o treino': (st['verification'].get(name) or {}).get(
                     'days_from_last_train_day_to_first_day')}
                for name, spec in st['cells'].items()
            ])
            display(Markdown('**As cinco células com papel, e o que separa cada uma do treino**'))
            display(cells.set_index('célula'))

            display(Markdown('**Volume de treino por protocolo** (uma queda de desempenho com menos dados não '
                             'é evidência de pior generalização)'))
            display(pd.DataFrame({
                kind: {'janelas de treino': comparison[kind]['train']['windows_contiguous'],
                       'anos de treino': comparison[kind]['train']['years'],
                       'sobreposição teste×treino': comparison['test_overlap_with_train'][kind],
                       'isola': comparison['isolates'][kind]}
                for kind in ('temporal', 'spatial', 'spatiotemporal')
            }))

            viability = st['viability']
            display(Markdown('**Gate de viabilidade das células** (fixado antes de olhar qualquer rótulo)'))
            display(pd.DataFrame(viability).T[['events', 'windows', 'passes']])
            folds = st.get('fold_choice') or {}
            if folds:
                print('Folds viáveis sob o cruzamento:', folds.get('viable_folds'),
                      '· escolhido:', folds.get('chosen'))
            print('Selado (sem papel):', st['sealed']['windows_contiguous'], 'janelas ·',
                  'descartado no corredor:', st['dropped']['windows_contiguous'], 'janelas')
            """,
        ),
        md(
            "wsts-eda-target-md",
            """
            ## 5. Alvo, transições e frente (somente treino)

            Prevalência de fogo em t+1, persistência (todos os pares consecutivos; não comparável à do STConv) e distância
            das ignições novas ao pixel ativo mais próximo — em t e na união t−4..t (omissões do VIIRS).
            """,
        ),
        code(
            "wsts-eda-target",
            """
            target = summary['train_target']
            display(Markdown(
                f"Prevalência t+1: **{target['prevalence_next']:.5f}** · pixels novos / positivos: "
                f"**{target['new_fraction_of_positives']:.3f}** · dias-evento com fogo em t+1: "
                f"{target['event_days_with_fire_next']} de {target['event_days']}"
            ))
            display(pd.Series(target['persistence_baseline_all_pairs']).to_frame('persistência (treino)'))
            display(pd.DataFrame(target['transitions_per_event_day']).T.round(1))
            distance = target['new_pixel_distance']
            display(pd.DataFrame({'ao fogo em t': distance['to_fire_t'],
                                  'à união da janela': distance['to_window_union']}))
            display(pd.Series(distance['km_to_fire_t']).to_frame('km até o fogo em t'))
            show_figure('wsts_distancia_ignicoes_frente')
            """,
        ),
        md(
            "wsts-eda-ring-md",
            """
            ## 6. Anel da frente (somente treino)

            SMD entre pixels do anel (≤ 2 px do fogo em t) que **inflamam** em t+1 e os que não inflamam, calculada por
            evento-dia e agregada por evento (IC por bootstrap de eventos). Associação descritiva, não causal; NDVI/EVI2 com
            a ressalva de look-ahead do composto (seção 7). Os cossenos comparam a direção frente→candidato com o transporte
            do vento e com a subida (gradiente da elevação na grade).
            """,
        ),
        code(
            "wsts-eda-ring",
            """
            ring = target['front_ring']
            print('Fração dos pixels novos dentro do anel:', ring['share_of_new_pixels_in_ring'])
            smd = pd.DataFrame(ring['smd_ignite_vs_not_by_event_day']).T
            display(smd.sort_values('smd_mean_of_event_means'))
            display(pd.DataFrame(ring['cosine_front_direction_ignite_minus_not']).T)
            show_figure('wsts_smd_anel_frente')
            """,
        ),
        md("wsts-eda-checks-md", "## 7. Checagens de convenção e de semântica temporal (somente treino)"),
        code(
            "wsts-eda-checks",
            """
            aspect = summary['aspect_grid_rotation_check']
            display(Markdown('**Rotação da grade EPSG:32610 (aspect × gradiente):**'))
            display(pd.Series(aspect.get('regression_offset_on_minus_gamma', {})).to_frame('offset ~ −γ'))
            display(pd.Series(summary['grid_gamma_deg_train']).to_frame('γ (graus), treino').round(2))
            display(Markdown('**Look-ahead do composto VNP13A1 (diferença-em-diferenças por evento):**'))
            composite = summary['composite_look_ahead_check']
            for name in ('ndvi', 'evi2'):
                block = composite[name]
                display(Markdown(f"*{name}* — cobertura {block['coverage_of_burned_pixels']}; "
                                 f"dose-resposta {block['dose_response_spearman_days_after_d0_vs_did']['value']}"))
                display(pd.DataFrame({'DiD': block['did'], 'placebo': block['placebo_did']}))
            display(Markdown('**Horizonte efetivo da previsão GFS:**'))
            for name in ('temperature', 'specific_humidity'):
                block = summary['forecast_horizon_check'].get(name, {})
                if 'corr_observed_by_lag' in block:
                    display(Markdown(f"*{name}* — mais perto de H1 em {block['bootstrap_share_closer_to_H1']:.0%} dos bootstraps; "
                                     f"razão de desvios GFS/GRIDMET = {block['std_ratio_gfs_over_gridmet']:.2f}"))
                    display(pd.DataFrame({k: block[k] for k in ('corr_observed_by_lag', 'profile_H0', 'profile_H1')}).round(3))
            show_figure('wsts_horizonte_previsao')
            wind = summary['wind_axis_check']
            display(Markdown(f"**Eixo do vento previsto × GRIDMET:** média de cos 2Δ = {wind['mean_cos_2delta']}"))
            show_figure('wsts_convencao_vento')
            hours = pd.Series(summary['active_fire_hour_histogram_utc_train'], name='detecções')
            display(hours[hours > 0].to_frame().T)
            """,
        ),
        md("wsts-eda-official-md", "## 8. Estatísticas de normalização × tabela oficial e sobreposição espacial"),
        code(
            "wsts-eda-official",
            """
            official = summary.get('official_statistics_comparison')
            if official:
                display(pd.DataFrame(official).T[['mean_ours', 'mean_official', 'mean_rel_diff', 'std_rel_diff']].round(4))
                show_figure('wsts_estatisticas_vs_oficial')
            display(pd.DataFrame(summary['spatial_overlap_with_train']).T)
            print('Detecções hhmm < 100 apagadas na conversão:', summary.get('lost_detections_hhmm_lt_100'))
            """,
        ),
    ]
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    nbformat.validate(notebook)
    return notebook


def main() -> None:
    notebook = build()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook: {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
