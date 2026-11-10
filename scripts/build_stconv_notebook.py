"""Gera `notebooks/05_stconv_integracao.ipynb`.

O notebook é somente leitura: carrega os relatórios de `reports/stconv_runs/*.json` produzidos por
`scripts/stconv_train.py`, os resumos do baseline (`run_baseline.py`) e a comparação pareada entre as receitas
(`compare_recipes.py`), e não retreina nem reavalia nada. Mesmo padrão do builder da EDA.

Parte A: o baseline nos 12 folds oficiais (receitas `paper` e `paper_es`). Parte B: os runs pré-registrados do
fold 2 (configuração principal e ablations); os relatórios do baseline ficam fora dela.
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


# Parte A: baseline nos 12 folds oficiais. Só lê os resumos e a comparação; cada célula tolera a ausência deles.
BASELINE_CELLS = [
    md(
        "baseline-md",
        """
        ## Parte A — baseline nos 12 folds oficiais

        Protocolo dos modelos multitemporais do artigo do dataset (Gerard et al., *WildfireSpreadTS*, NeurIPS 2023
        Datasets and Benchmarks): cada fold treina com dois anos, valida com um e testa com o quarto; AdamW (lr 0,01,
        weight decay 0,01), checkpoint pela menor `val_loss` em recortes aumentados da validação, `pos_weight` =
        1/taxa de pixels com fogo no treino, teste uma única vez por run na região `center_crop_x32`, todas as
        fontes de features, semente 0. Duas receitas, com runs, logs e resumos separados:

        - **`paper`** — 10.000 passos com lr constante, a duração do artigo. Rodou só em parte dos folds (tabela
          abaixo), por custo computacional na GPU disponível (6 GB).
        - **`paper_es`** — a mesma, com o lr reduzido no platô da `val_loss` e parada antecipada (valores na tabela
          abaixo). Rodou nos 12 folds e é o resultado principal.

        A troca de receita foi decidida com resultados de teste da `paper` já conhecidos; por isso a regra da
        `paper_es` e o resultado principal foram fixados num commit antes de qualquer run dela, e as duas receitas são
        sempre reportadas juntas. Os folds da `paper` servem só para a comparação pareada; nunca completam a
        `paper_es`.

        **Desvios em relação ao artigo, nas duas receitas:** lote 32 em 2 micro-lotes de 16 (o BatchNorm vê 16
        amostras); AMP fp16 em vez de fp32; 48 canais em vez de 40 (seno e cosseno dos ângulos, flags de ausência,
        sem o canal de hora de detecção); ângulos corrigidos no aumento de dados por rotação.
        """,
    ),
    code(
        "baseline-protocol",
        """
        KEYS = ['batch_size', 'accumulate', 'max_steps', 'lr', 'weight_decay', 'lr_schedule', 'lr_factor',
                'lr_patience', 'min_lr', 'early_stop_patience', 'checkpoint_metric', 'val_sampling', 'eval_region',
                'pos_weight_source', 'augment', 'seed']
        if not baseline:
            display(Markdown('_Baseline ainda não executado._'))
        else:
            config = {}
            for recipe in BASELINE_RECIPES:
                report = next((r for r in baseline_reports.values() if r['protocol']['recipe'] == recipe), None)
                if report:
                    optimization = report['protocol']['optimization']
                    config[recipe] = {**{key: report['config'].get(key) for key in KEYS},
                                      'micro_batch': optimization.get('micro_batch'),
                                      'precision': optimization.get('precision'),
                                      'channels': len(report['protocol']['channels'])}
            display(pd.DataFrame(config))
            display(Markdown('**Folds testados:** ' + '; '.join(
                f"`{recipe}` {s['aggregate']['n_folds_tested']} de {s['aggregate']['n_folds_expected']} "
                f"({', '.join(str(f['fold']) for f in sorted(s['folds'], key=lambda f: f['fold']))})"
                for recipe, s in baseline.items())))
        """,
    ),
    md(
        "baseline-results-md",
        """
        ### A1. AP de teste por fold

        AP na região `center_crop_x32`, com IC 95% por bootstrap de eventos inteiros. F2 no limiar escolhido na
        validação, contra a persistência de todas as detecções do dia `t`, com IC pareado por eventos. O desvio entre
        folds mede sobretudo a variação entre anos de teste e não é erro-padrão; a prevalência, que é o piso da AP,
        muda muito entre os anos.
        """,
    ),
    code(
        "baseline-results",
        """
        def fmt(value, digits=3):
            return '—' if value is None else f'{value:.{digits}f}'

        def f2_ci(recipe, fold):
            report = baseline_reports.get(f"{'baseline_es_f' if recipe == 'paper_es' else 'baseline_paper_f'}{fold:02d}")
            block = (report or {}).get('test', {}).get('metrics', {}).get('all', {})
            return (block.get('f2_vs_persistence_paired_bootstrap') or {}).get('ci95')

        def fold_table(recipe, summary):
            rows = []
            for f in sorted(summary['folds'], key=lambda f: f['fold']):
                ci = f2_ci(recipe, f['fold'])
                rows.append({
                    'fold': f['fold'], 'teste': f['years']['test'][0], 'AP': f['test_ap'],
                    'IC 95% (eventos)': f"[{f['test_ap_ci95'][0]:.3f}; {f['test_ap_ci95'][1]:.3f}]",
                    'F2 modelo': f['test_f2_at_val_threshold'], 'F2 persistência': f['test_persistence_f2'],
                    'IC 95% ΔF2': '—' if not ci else f'[{ci[0]:+.3f}; {ci[1]:+.3f}]',
                    # o resumo da `paper` é anterior ao campo `epochs_done`
                    'melhor época': f['best_epoch'], 'épocas': f.get('epochs_done'),
                    'prevalência': f['test_prevalence'],
                })
            return pd.DataFrame(rows).set_index('fold')

        for recipe, summary in baseline.items():
            agg = summary['aggregate']
            display(Markdown(
                f"#### `{recipe}` — AP de teste **{fmt(agg['test_ap_mean'])} ± {fmt(agg['test_ap_std'])}** em "
                f"{agg['n_folds_tested']} folds" + ('' if agg['complete'] else ' (não são os 12)')))
            table = fold_table(recipe, summary)
            display(table.round(4))
            prevalence = table.groupby('teste')['prevalência'].mean()
            display(pd.DataFrame(agg['by_test_year']).T.rename(
                columns={'n': 'folds', 'mean': 'AP média', 'std': 'desvio'}).assign(
                prevalência=prevalence.rename(index=str)).round(4))
            cis = [f2_ci(recipe, f['fold']) for f in summary['folds']]
            above = sum(1 for ci in cis if ci and ci[0] > 0)
            points = sum(f['test_f2_at_val_threshold'] > f['test_persistence_f2'] for f in summary['folds'])
            held_out = agg.get('folds_with_test_2021') or {}
            display(Markdown(
                f"F2 do modelo acima do da persistência com IC pareado acima de zero em {above} de "
                f"{len(summary['folds'])} folds (na estimativa pontual, {points})."
                + (f" Os folds com teste em 2021 são os únicos cujo ano de teste não participou da EDA nem das "
                   f"decisões de codificação: AP {fmt(held_out.get('mean'))} ± {fmt(held_out.get('std'))} em "
                   f"{held_out['n']} folds." if held_out.get('n') else '')))
        """,
    ),
    code(
        "baseline-figure",
        """
        if 'paper_es' in baseline:
            import matplotlib.pyplot as plt

            es = {f['fold']: f for f in baseline['paper_es']['folds']}
            order = sorted(es, key=lambda k: (es[k]['years']['test'][0], k))
            x = list(range(len(order)))
            ap = [es[k]['test_ap'] for k in order]
            low = [es[k]['test_ap'] - es[k]['test_ap_ci95'][0] for k in order]
            high = [es[k]['test_ap_ci95'][1] - es[k]['test_ap'] for k in order]
            fig, ax = plt.subplots(figsize=(7.5, 3.2))
            ax.errorbar(x, ap, yerr=[low, high], fmt='o', color='#2a78d6', ecolor='#9cc0ea', capsize=3,
                        label='paper_es (IC 95% por eventos)')
            if 'paper' in baseline:
                paper = {f['fold']: f['test_ap'] for f in baseline['paper']['folds']}
                shown = [i for i, k in enumerate(order) if k in paper]
                ax.scatter(shown, [paper[order[i]] for i in shown], marker='D', facecolors='none',
                           edgecolors='#eb6834', s=40, label='paper', zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"f{k} ({es[k]['years']['test'][0]})" for k in order], rotation=45, ha='right')
            ax.set_ylabel('AP de teste')
            ax.set_title('AP de teste por fold, ordenado pelo ano de teste')
            ax.grid(axis='y', color='#e4e3df', linewidth=0.6); ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
            plt.show()
        """,
    ),
    md(
        "baseline-paired-md",
        """
        ### A2. Comparação pareada `paper` × `paper_es`

        Nos folds que rodaram nas duas receitas, Δ = AP(`paper_es`) − AP(`paper`), com IC 95% por bootstrap pareado:
        cada reamostragem sorteia os eventos de teste com reposição e calcula o AP das duas receitas nos mesmos
        eventos. Mesma semente, mesma ordem dos lotes e mesmos recortes de validação: até o primeiro corte do lr as
        duas receitas devem ter a mesma `val_loss` por época, o que também testa o determinismo do treino na GPU.
        """,
    ),
    code(
        "baseline-paired",
        """
        if comparison:
            entries = comparison['folds']
            rows = [{
                'fold': e['fold'], 'teste': e['test_year'],
                'paper': e['test_ap']['paper'], 'paper_es': e['test_ap']['paper_es'], 'Δ': e['delta'],
                'IC 95% do Δ': f"[{e['bootstrap']['ci95'][0]:+.3f}; {e['bootstrap']['ci95'][1]:+.3f}]",
                'melhor época': f"{e['best_epoch']['paper']} → {e['best_epoch']['paper_es']}",
                'mesmos pesos': e['same_weights'],
                'val_loss idêntica até o corte': (f"sim ({e['prefix']['epochs_compared']} épocas)"
                                                  if e['prefix']['identical'] else 'não'),
            } for e in entries]
            display(pd.DataFrame(rows).set_index('fold').round(3))
            s = comparison['summary']
            by_year = {}
            for e in entries:
                by_year.setdefault(e['test_year'], []).append(str(e['fold']))
            shared = '; '.join(f"os folds {' e '.join(f)} testam os mesmos eventos de {year}"
                               for year, f in sorted(by_year.items()) if len(f) > 1)
            truncation = all(e['best_epoch']['paper_es'] <= e['prefix']['up_to_epoch'] for e in entries)
            text = (f"Média nos {s['n_folds']} folds: `paper` {s['mean_test_ap']['paper']:.3f} → `paper_es` "
                    f"{s['mean_test_ap']['paper_es']:.3f} (Δ médio {s['mean_delta']:+.3f}, descritivo). "
                    + (f"Os folds não são independentes ({shared}). " if shared else '')
                    + ('A melhor época da `paper_es` caiu dentro do trecho idêntico às duas receitas, então o Δ '
                       'compara um checkpoint mais cedo e um mais tarde da mesma trajetória: mede o efeito de parar '
                       'antes, não o do lr reduzido. ' if truncation else ''))
            if all(e['delta'] <= 0 for e in entries) and s['mean_delta'] < 0:
                text += ('Nesta semente, Δ ≤ 0 em todos os folds pareados, e a direção sugere que o resultado da '
                         '`paper_es` nos 12 folds subestima a receita do artigo; a magnitude nos 12 folds não foi '
                         'estimada. ')
            display(Markdown(text + s['reading']))
        else:
            display(Markdown('_Comparação pareada ainda não gerada (`compare_recipes.py`)._'))

        es_reports = [r for r in baseline_reports.values() if r['protocol']['recipe'] == 'paper_es']
        if es_reports:
            later = sorted(r['protocol']['fold'] for r in es_reports
                           if r['protocol']['optimization'].get('lr_reductions')
                           and r['best_epoch'] > r['protocol']['optimization']['lr_reductions'][0]['after_epoch'])
            display(Markdown(
                f"Na `paper_es`, a melhor época veio depois do primeiro corte do lr em {len(later)} de "
                f"{len(es_reports)} folds ({', '.join(map(str, later)) or 'nenhum'}); nos demais, o checkpoint é de "
                'antes do corte, com lr 0,01.'))
        """,
    ),
    md(
        "baseline-reference-md",
        """
        ### A3. Referência: tabela 5 do artigo

        Gerard et al. (2023), p. 9, tabela 5, coluna "All": AP de teste, média ± desvio nos 12 folds, com todas as
        features. O STConvS2S não está no artigo, a `paper_es` não é a receita dele e a codificação e a precisão
        diferem (desvios acima): a tabela é referência, não reprodução. A AP da persistência não é calculada aqui (só
        o F2), e o repositório oficial avisa que a correção de um erro do dataset, posterior ao artigo, elevou
        levemente os resultados. A linha da `paper` cobre só parte dos folds e anos de teste e não é comparável às
        médias de 12 folds; a linha seguinte dá a `paper_es` nos mesmos folds.
        """,
    ),
    code(
        "baseline-reference",
        """
        TABLE5 = [
            ('Persistência (1 dia)', 0.193, 0.065), ('Regressão logística (1 dia)', 0.286, 0.092),
            ('ConvLSTM (5 dias)', 0.292, 0.094), ('UTAE (5 dias)', 0.321, 0.135),
            ('Res18 U-Net (5 dias)', 0.325, 0.108), ('Res18 U-Net (1 dia)', 0.341, 0.086),
        ]
        rows = [{'modelo': name, 'fonte': 'artigo, 12 folds', 'AP': mean, 'desvio': std} for name, mean, std in TABLE5]
        for recipe, summary in baseline.items():
            agg = summary['aggregate']
            rows.append({'modelo': f'STConvS2S ({recipe})', 'fonte': f"este trabalho, {agg['n_folds_tested']} folds",
                         'AP': agg['test_ap_mean'], 'desvio': agg['test_ap_std']})
        if comparison:
            s = comparison['summary']
            rows.append({'modelo': 'STConvS2S (paper_es), folds da paper',
                         'fonte': f"este trabalho, {s['n_folds']} folds", 'AP': s['mean_test_ap']['paper_es'],
                         'desvio': None})
        display(pd.DataFrame(rows).set_index('modelo').round(3))
        """,
    ),
    md(
        "baseline-limits-md",
        """
        ### A4. Limitações do baseline

        - **Receita alterada.** O resultado principal usa parada antecipada, não os 10.000 passos do artigo; o efeito
          medido está na comparação pareada (A2) e só nos folds em que as duas receitas rodaram.
        - **Desvios de codificação e de precisão.** Micro-lote 16, AMP fp16 e 48 canais separam o baseline da
          receita do artigo mesmo onde a duração é a mesma.
        - **Uma semente.** Os IC medem a amostra de eventos de teste, não a variação do treino.
        - **Comparação pareada limitada.** Poucos folds, não independentes e desbalanceados por ano de teste (A2);
          as médias são descritivas, e um IC que cruzasse zero não seria equivalência.
        - **`val_loss` ruidosa.** Medida em recortes ressorteados a cada época e com `pos_weight` de várias centenas
          (1/taxa de fogo do treino de cada fold), ela escolhe o checkpoint e decide os cortes e a parada.
        - **NDVI/EVI2 com look-ahead.** Os compostos de 16 dias podem conter observações posteriores ao dia alvo;
          os canais entram no baseline como no artigo.
        """,
    ),
]


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "stconv-title",
            """
            # WildfireSpreadTS × STConvS2S — integração

            Resultados dos runs de `scripts/stconv_train.py`, em duas partes:

            - **Parte A — baseline nos 12 folds oficiais**, com o protocolo de treino do artigo do dataset e uma
              variante dele com parada antecipada;
            - **Parte B — runs pré-registrados do fold 2** (configuração principal, seeds e ablations).

            Nas duas, o ano de teste é avaliado **uma única vez por run**, a partir do checkpoint com SHA conferido, e
            cada avaliação fica registrada em `reports/test_ledger.jsonl`. O pré-registro da Parte B
            (configuração, seeds, métricas e regra de decisão dos ablations) foi fixado antes do primeiro run completo;
            os parâmetros que decidem a leitura estão em `scripts/core.py` (`PREREGISTERED_*`, `ABLATION_*`).
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
            REPORTS = ROOT / 'reports'
            RUNS_DIR = REPORTS / 'stconv_runs'
            DEMO_DIR = REPORTS / 'demo_runs'
            BASELINE_RECIPES = ('paper', 'paper_es')

            def read_json(path):
                return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

            # Parte A: resumos de `run_baseline.py` e comparação de `compare_recipes.py`
            baseline = {recipe: read_json(REPORTS / name) for recipe, name in
                        (('paper', 'baseline_paper_summary.json'), ('paper_es', 'baseline_es_summary.json'))}
            baseline = {recipe: summary for recipe, summary in baseline.items() if summary}
            comparison = read_json(REPORTS / 'baseline_recipe_comparison.json')
            baseline_reports = {}

            # Parte B: runs completos que não são do baseline
            runs, kind = {}, 'full'
            if RUNS_DIR.is_dir():
                for path in sorted(RUNS_DIR.glob('*.json')):
                    report = json.loads(path.read_text(encoding='utf-8'))
                    if report.get('kind') != 'full':
                        continue
                    if report.get('protocol', {}).get('recipe') in BASELINE_RECIPES:
                        baseline_reports[report['run_name']] = report
                    else:
                        runs[report['run_name']] = report
            if not runs and DEMO_DIR.is_dir():
                kind = 'demo'
                for path in sorted(DEMO_DIR.glob('*.json')):
                    report = json.loads(path.read_text(encoding='utf-8'))
                    if report.get('kind') == 'demo':
                        runs[report['run_name']] = report
            if not runs and not baseline:
                raise FileNotFoundError(f'Nenhum run em {RUNS_DIR} nem em {DEMO_DIR}.')
            """,
        ),
        *BASELINE_CELLS,
        md(
            "stconv-part-b-md",
            """
            ## Parte B — runs pré-registrados do fold 2

            Configuração principal em 3 seeds e os ablations sem NDVI/EVI2 e sem canais GFS, com checkpoint pela maior
            AP de validação e threshold pelo maior F2 de validação. Enquanto os runs completos não existem, esta parte
            mostra a demonstração em amostra, que não é evidência.
            """,
        ),
        code(
            "stconv-runs",
            """
            if not runs:
                raise FileNotFoundError(f'Nenhum run pré-registrado em {RUNS_DIR} nem em {DEMO_DIR}.')
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
            ### B1. Contrato de dados

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
            ### B2. Treino e seleção

            `pos_weight` é estimado sob a mesma distribuição de recortes usada no treino. O checkpoint guardado é o da
            época com maior AP de validação; nenhuma métrica de teste participa da seleção.
            """,
        ),
        code(
            "stconv-training",
            """
            for name, report in runs.items():
                display(Markdown(f"#### {name} — {label(report)}"))
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
            ### B3. Métricas por vista

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
            ### B4. Comparação com persistência

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
            ### B5. Ablations

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
            ### B6. Proveniência e protocolo de teste

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
            ### B7. Limitações

            - **Fold único.** O artigo original agrega 12 folds; aqui há um só, então não há estimativa de variância
              entre folds e os números não são diretamente comparáveis às tabelas do artigo (para isso, ver a Parte A).
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
