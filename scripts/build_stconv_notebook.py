"""Gera `notebooks/05_stconv_integracao.ipynb`.

O notebook é somente leitura: carrega os relatórios de `reports/stconv_runs/*.json` produzidos por
`scripts/stconv_train.py` e não retreina nem reavalia nada. Mesmo padrão do builder da EDA.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "05_stconv_integracao.ipynb"


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
            "stconv-title",
            """
            # WildfireSpreadTS × STConvS2S — integração

            Resultados dos runs de `scripts/stconv_train.py`. **Protocolo:** fold fixado antes de qualquer treino;
            checkpoint escolhido pela maior AP de validação; threshold pelo maior F2 de validação; o ano de teste é
            avaliado **uma única vez por run pré-registrado**, a partir do checkpoint com SHA conferido, e cada
            avaliação fica registrada em `reports/test_ledger.jsonl`.

            O pré-registro (configuração, seeds, métricas e regra de decisão dos ablations) foi fixado antes do
            primeiro run completo; os parâmetros que decidem a leitura estão em `scripts/core.py`
            (`PREREGISTERED_*`, `ABLATION_*`).
            """,
        ),
        code(
            "stconv-setup",
            """
            import json
            from pathlib import Path

            import pandas as pd
            from IPython.display import Markdown, display

            def find_root():
                for candidate in (Path.cwd(), *Path.cwd().parents):
                    reports = candidate / 'reports'
                    if (reports / 'stconv_runs').is_dir() or (reports / 'demo_runs').is_dir():
                        return candidate
                raise FileNotFoundError('Rode scripts/stconv_train.py antes de abrir este notebook.')

            ROOT = find_root()
            RUNS_DIR = ROOT / 'reports' / 'stconv_runs'
            DEMO_DIR = ROOT / 'reports' / 'demo_runs'
            runs, kind = {}, 'full'
            if RUNS_DIR.is_dir():
                for path in sorted(RUNS_DIR.glob('*.json')):
                    report = json.loads(path.read_text(encoding='utf-8'))
                    if report.get('kind') == 'full':
                        runs[report['run_name']] = report
            if not runs and DEMO_DIR.is_dir():
                kind = 'demo'
                for path in sorted(DEMO_DIR.glob('*.json')):
                    report = json.loads(path.read_text(encoding='utf-8'))
                    if report.get('kind') == 'demo':
                        runs[report['run_name']] = report
            if not runs:
                raise FileNotFoundError(f'Nenhum run em {RUNS_DIR} nem em {DEMO_DIR}.')
            if kind == 'demo':
                display(Markdown(
                    '> ⚠️ **Demonstração, não evidência.** Estes runs usaram uma amostra de '
                    f"{next(iter(runs.values()))['config']['max_events']} eventos por ano e o conjunto de teste "
                    'não foi avaliado. Servem para mostrar que o fluxo roda ponta a ponta; '
                    'nenhum número aqui sustenta conclusão sobre desempenho.'
                ))

            def label(report):
                dropped = report['protocol']['drop_groups']
                base = 'completo' if not dropped else 'sem ' + '+'.join(dropped)
                return f"{base} (seed {report['config']['seed']})"

            reference = next(iter(runs.values()))
            display(Markdown(
                f"**{len(runs)} run(s) ({kind})** · fold {reference['protocol']['fold']} · "
                f"T_in = {reference['protocol']['t_in']} · alvo: {reference['protocol']['target']}"
            ))
            display(pd.DataFrame([{
                'run': name, 'configuração': label(r), 'épocas': len(r['history']),
                'melhor época': r['best_epoch'], 'teste avaliado': 'test' in r,
                     'tipo': r['kind'],
                'commit': (r['provenance']['repo']['commit'] or '?')[:7],
            } for name, r in runs.items()]).set_index('run'))
            """,
        ),
        md(
            "stconv-contract-md",
            """
            ## 1. Contrato de dados

            Cada amostra é uma janela de `T_in` dias consecutivos de um mesmo evento. O alvo é a máscara binária de fogo
            ativo do dia seguinte. A máscara `M` vale 1 em todo pixel avaliado — **inclusive os negativos** — e 0 apenas
            em padding e no halo dos tiles, de modo que a classe majoritária continua entrando na loss e nas métricas.
            """,
        ),
        code(
            "stconv-contract",
            """
            protocol = reference['protocol']
            display(pd.Series({
                'fold': protocol['fold'],
                'treino': protocol['split']['train'],
                'validação': protocol['split']['validation'],
                'teste': protocol['split']['test'],
                'dias de entrada (T_in)': protocol['t_in'],
                'leitura da saída': protocol['readout'],
                'amostragem de treino': protocol['train_sampling'],
                'tiles na avaliação': protocol['eval_tiling'],
                'nº de canais': len(protocol['channels']),
            }).to_frame('contrato'))
            display(Markdown('**Vistas avaliadas**'))
            display(pd.Series(protocol['views']).to_frame('definição'))
            display(Markdown('**Baselines de persistência**'))
            display(pd.Series(protocol['baselines']).to_frame('campo usado como predição'))
            print(protocol['baseline_notes'])
            print()
            print('Canais:', ', '.join(protocol['channels']))
            """,
        ),
        md(
            "stconv-training-md",
            """
            ## 2. Treino e seleção

            `pos_weight` é estimado sob a mesma distribuição de recortes usada no treino. O checkpoint guardado é o da
            época com maior AP de validação; nenhuma métrica de teste participa da seleção.
            """,
        ),
        code(
            "stconv-training",
            """
            for name, report in runs.items():
                display(Markdown(f"### {name} — {label(report)}"))
                pw = report['pos_weight']
                display(Markdown(
                    f"`pos_weight` = **{pw['pos_weight']:.1f}** ({pw['positive']} positivos / {pw['negative']} negativos "
                    f"em {pw['samples']} recortes) · janelas: {report['n_train_windows']} treino, "
                    f"{report['n_validation_windows']} validação · pico de VRAM: {report['peak_vram_mib']} MiB"
                ))
                history = pd.DataFrame(report['history']).set_index('epoch')
                display(history.round(5))
                if len(history) > 1:
                    ax = history[['val_ap', 'val_ap_new_vs_t']].plot(
                        figsize=(6.4, 3), marker='o', markersize=4, linewidth=2,
                        color=['#2a78d6', '#eb6834'], title=f'AP de validação por época — {name}')
                    ax.set_xlabel('época'); ax.grid(axis='y', color='#e4e3df', linewidth=0.6); ax.set_axisbelow(True)
                    for side in ('top', 'right'):
                        ax.spines[side].set_visible(False)
            """,
        ),
        md(
            "stconv-metrics-md",
            """
            ## 3. Métricas por vista

            A métrica primária é a AP (*average precision*), adequada a classe rara. O IC 95% vem de bootstrap
            reamostrando **eventos inteiros**, não pixels: pixels do mesmo incêndio são fortemente dependentes.

            - `all` — todos os pixels avaliados
            - `crop32` — recorte central múltiplo de 32, vista comparável à do artigo
            - `new_vs_t` — só pixels sem fogo em t (onde a persistência não ajuda)
            - `new_vs_window` — só pixels sem fogo em nenhum dos dias de entrada
            """,
        ),
        code(
            "stconv-metrics",
            """
            def metric_rows(block, origin, run):
                rows = []
                for view, entry in block.items():
                    model, ci = entry['model'], entry.get('ap_bootstrap_events')
                    at = entry.get('model_at_validation_threshold', {})
                    rows.append({
                        'run': run, 'origem': origin, 'vista': view,
                        'AP': model['average_precision'],
                        'IC 95% AP': None if not ci else f"[{ci['ci95'][0]:.4f}; {ci['ci95'][1]:.4f}]",
                        'AP prevalência': entry['prevalence_baseline_ap'],
                        'F2': at.get('f2'), 'precisão': at.get('precision'), 'recall': at.get('recall'),
                        'positivos': model['n_positive'],
                    })
                return rows

            rows = []
            for name, report in runs.items():
                rows += metric_rows(report['validation'], 'validação', name)
                if 'test' in report:
                    rows += metric_rows(report['test']['metrics'], 'teste', name)
            display(pd.DataFrame(rows).set_index(['run', 'origem', 'vista']).round(5))
            """,
        ),
        md(
            "stconv-baselines-md",
            """
            ## 4. Comparação com persistência

            O baseline a bater é a persistência: repetir o mapa de fogo do dia anterior. O IC pareado reamostra os
            mesmos eventos para o modelo e para a persistência, então mede a diferença dentro de cada evento.
            """,
        ),
        code(
            "stconv-baselines",
            """
            rows = []
            for name, report in runs.items():
                for origin, block in [('validação', report['validation'])] + (
                        [('teste', report['test']['metrics'])] if 'test' in report else []):
                    entry = block['all']
                    for baseline, metrics in entry['baselines'].items():
                        rows.append({'run': name, 'origem': origin, 'baseline': baseline,
                                     'F2': metrics['f2'], 'precisão': metrics['precision'], 'recall': metrics['recall']})
                    at = entry.get('model_at_validation_threshold', {})
                    rows.append({'run': name, 'origem': origin, 'baseline': '** modelo **',
                                 'F2': at.get('f2'), 'precisão': at.get('precision'), 'recall': at.get('recall')})
            display(pd.DataFrame(rows).set_index(['run', 'origem', 'baseline']).round(5))

            display(Markdown('**Teste pareado: F2(modelo) − F2(persistência de todas as detecções de t)**'))
            paired = []
            for name, report in runs.items():
                for origin, block in [('validação', report['validation'])] + (
                        [('teste', report['test']['metrics'])] if 'test' in report else []):
                    p = block['all'].get('f2_vs_persistence_paired_bootstrap')
                    if p:
                        paired.append({'run': name, 'origem': origin, 'diferença': p['point'],
                                       'IC 95%': f"[{p['ci95'][0]:.4f}; {p['ci95'][1]:.4f}]",
                                       'supera persistência': p['model_beats_persistence'], 'eventos': p['n_events']})
            display(pd.DataFrame(paired).set_index(['run', 'origem']).round(5) if paired
                    else Markdown('_IC pareado indisponível (menos de 3 eventos)._'))
            """,
        ),
        md(
            "stconv-ablations-md",
            """
            ## 5. Ablations

            Dois grupos de canais são removidos em runs separados, conforme o pré-registro:

            - **sem NDVI/EVI2** — esses canais vêm de um composto de 16 dias que pode conter observações posteriores ao
              dia alvo (*look-ahead* estrutural). Uma queda grande é ambígua, porque mistura sinal legítimo com
              vazamento; uma queda desprezível indica que o look-ahead não é relevante para o modelo.
            - **sem canais GFS** — a previsão meteorológica tem horizonte efetivo provavelmente longo (vários dias),
              então seu papel preditivo precisa ser medido, não assumido.
            """,
        ),
        code(
            "stconv-ablations",
            """
            import itertools
            import numpy as np

            by_config = {}
            for report in runs.values():
                by_config.setdefault(tuple(report['protocol']['drop_groups']), []).append(report)

            summary = []
            for dropped, group in by_config.items():
                aps = [r['validation']['all']['model']['average_precision'] for r in group]
                summary.append({'configuração': 'completo' if not dropped else 'sem ' + '+'.join(dropped),
                                'seeds': len(aps), 'AP média (validação)': np.mean(aps),
                                'desvio entre seeds': np.std(aps, ddof=1) if len(aps) > 1 else np.nan})
            display(pd.DataFrame(summary).set_index('configuração').round(5))

            baseline_key = ()
            if baseline_key in by_config and len(by_config) > 1:
                base = [r['validation']['all']['model']['average_precision'] for r in by_config[baseline_key]]
                rows = []
                for dropped, group in by_config.items():
                    if dropped == baseline_key:
                        continue
                    aps = [r['validation']['all']['model']['average_precision'] for r in group]
                    pooled = np.sqrt((np.var(base, ddof=1) + np.var(aps, ddof=1)) / 2) if min(len(base), len(aps)) > 1 else np.nan
                    effect = np.mean(base) - np.mean(aps)
                    rows.append({'ablation': 'sem ' + '+'.join(dropped), 'efeito (ΔAP)': effect,
                                 'desvio agrupado': pooled,
                                 'relevante': bool(abs(effect) > 2 * pooled and abs(effect) > 0.01) if pooled == pooled else None})
                display(pd.DataFrame(rows).set_index('ablation').round(5))
                display(Markdown(
                    '_Regra pré-registrada: o efeito é relevante se |ΔAP| > 2 × desvio agrupado **e** > 0,01 **e** o IC '
                    'pareado por eventos exclui 0._'))
            else:
                display(Markdown('_Ablations ainda não executados._'))
            """,
        ),
        md(
            "stconv-provenance-md",
            """
            ## 6. Proveniência e protocolo de teste

            Cada relatório registra o commit do código, o SHA-256 dos scripts executados, o commit do clone do
            STConvS2S, a versão do PyTorch/CUDA e o SHA-256 do arquivo de normalização e do checkpoint avaliado. O
            livro-razão abaixo lista toda avaliação de teste já realizada — por construção, ele só cresce.
            """,
        ),
        code(
            "stconv-provenance",
            """
            rows = []
            for name, report in runs.items():
                prov = report['provenance']
                rows.append({
                    'run': name,
                    'commit': (prov['repo']['commit'] or '?')[:7],
                    'árvore suja': prov['repo']['dirty'],
                    'stconvs2s': (prov['stconvs2s'] or {}).get('commit', '?')[:7],
                    'torch': prov['torch']['version'],
                    'checkpoint sha256': report['checkpoint_sha256'][:12],
                })
            display(pd.DataFrame(rows).set_index('run'))
            display(Markdown('**Determinismo**'))
            display(pd.Series(reference['protocol']['determinism']).to_frame('configuração'))

            ledger = ROOT / 'reports' / 'test_ledger.jsonl'
            if ledger.exists():
                entries = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines() if line.strip()]
                display(Markdown(f'**Livro-razão de avaliações do teste** ({len(entries)} registro(s))'))
                display(pd.DataFrame(entries))
            else:
                display(Markdown('_Nenhuma avaliação de teste registrada._'))
            """,
        ),
        md(
            "stconv-limits-md",
            """
            ## 7. Limitações

            - **Fold único.** O artigo original agrega 12 folds; aqui há um só, então não há estimativa de variância
              entre folds e os números não são diretamente comparáveis às tabelas do artigo.
            - **Ruído de rótulo.** Pixels marcados como negativos incluem fogo que o VIIRS não detectou (nuvem, omissão),
              então a classe negativa está contaminada.
            - **Fogo ativo ≠ perímetro.** O alvo é detecção de fogo ativo no dia, não a área queimada acumulada.
            - **`pos_weight` diferente do oficial**, calculado sob a distribuição de recortes em vez de 1/taxa global.
            - **Ângulos no norte verdadeiro.** As convoluções operam na grade UTM, que está rotacionada em relação ao
              norte verdadeiro em até ~14° nos eventos mais a leste.
            - **Sem augmentation geométrica**, que exigiria transformar também os ângulos e vetores.
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
