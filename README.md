# WildfireSpreadTS — EDA, STConvS2S e candidatas a EDP

Pipeline reprodutível para analisar o **WildfireSpreadTS** (Gerard et al., NeurIPS 2023 Datasets & Benchmarks;
[Zenodo 8006177](https://zenodo.org/records/8006177)): 607 incêndios nos EUA entre 2018 e 2021, em grade de
375 m e passo diário, com fogo ativo do VIIRS, meteorologia, topografia, vegetação e previsões do GFS.

O pipeline cobre três frentes:

1. **Análise exploratória**: inventário, dados faltantes, alvo e frente de fogo, distribuição espacial e
   checagens da semântica temporal de cada canal.
2. **Integração com o STConvS2S**: previsão do fogo ativo em t+1 a partir de 5 dias de entrada, com teste único,
   baselines de persistência e ablations pré-registrados.
3. **Diagnósticos de EDP**: testes de que papéis físicos de uma EDP de reação–advecção–difusão o dado sustenta
   (advecção por vento e encosta, reação/extinção, combustível, escalas e Courant).

Os resultados (JSON, figuras e relatórios) ficam fora do Git, em `reports/`, e são regenerados pelo pipeline. Os
notebooks são a exceção: vão versionados com as saídas da execução de referência, para que a análise possa ser
lida sem baixar os dados.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| [`scripts/`](scripts/) | Pipeline: download, conversão, splits, EDA, STConvS2S, diagnósticos de EDP |
| [`tests/`](tests/) | Testes (`python -m unittest discover -s tests -t .`) |
| [`notebooks/`](notebooks/) | Notebooks somente leitura, gerados pelos builders a partir de `reports/` |
| [`config/splits/`](config/splits/) | Definições de split versionadas; o treino confere o split recomputado contra elas |
| `reports/` | Saídas geradas localmente (fora do Git) |

## Instalação

Python 3.12, a versão em que a suíte foi validada (`numpy==2.2.6` e `scipy==1.15.3` não têm wheels para 3.14).

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt # ou requirements.txt, só com as dependências diretas
pip install -r requirements-cuda.txt # GPU NVIDIA (driver >= 560); sem GPU: requirements-cpu.txt
```

### Repositórios de terceiros

Dois clones, fixados em commit, são esperados na raiz do repositório (fora do Git):

```bash
# Modelo STConvS2S (o treino importa só model.stconvs2s.STConvS2S_C, sem modificar o clone)
# (core.longpaths: o repositório tem saídas com nomes longos que o Windows recusa sem ele)
git clone -c core.longpaths=true -b noemi https://github.com/noemicho/stconvs2s .stconvs2s
git -C .stconvs2s checkout c355e9e023a84625f8e31afd386e47fea98708d6

# Repositório oficial do dataset (a EDA compara a normalização com a tabela oficial dos autores)
git clone https://github.com/SebastianGer/WildfireSpreadTS .wildfirespreadts
git -C .wildfirespreadts checkout f62dc49f2c2a6d249c071533ae5340c0d01bfb95
```

A semântica dos canais foi verificada no repositório de criação do dataset,
[`SebastianGer/WildfireSpreadTSCreateDataset`](https://github.com/SebastianGer/WildfireSpreadTSCreateDataset)
@ `f6c5167`, que não precisa ser clonado.

## Execução

Os comandos rodam a partir da raiz do repositório. Os dados ficam em `dados_geoespaciais/wildfirespreadts/`
(fora do Git); a raiz é configurável por `--data-root` ou pela variável `WSTS_DATA_ROOT`.

```bash
# 0. Ambiente registrado (versões, GPU, commits dos clones)
python scripts/env_report.py

# 1. Dados (uma vez): ~45 GiB do Zenodo com MD5 conferido, depois ~24 GiB em HDF5
python scripts/download.py
python scripts/convert_hdf5.py

# 2. Splits: as definições versionadas em config/splits/ já são as do experimento, e o treino e a EDA
#    conferem o split recomputado contra elas. make_splits.py se recusa a sobrescrevê-las; regenerar
#    (--force) só se o conjunto de eventos mudar, e isso é um desvio metodológico a registrar.
# python scripts/make_splits.py --kind spatiotemporal --force

# 3. EDA (fixa a normalização de cada protocolo) e notebook 04
python scripts/eda.py
python scripts/build_eda_notebook.py

# 4. STConvS2S: medir o custo por época ANTES de ver qualquer métrica
python scripts/stconv_train.py --timing-only --run-name tempo
#    os 9 runs pré-registrados (retomável; avalia o teste uma vez ao fim de cada run, salvo com --skip-test)
python scripts/run_preregistered.py
#    ou um run isolado, e a avaliação única do teste a partir do checkpoint
python scripts/stconv_train.py --run-name principal_s42 --seed 42
python scripts/stconv_train.py --evaluate-test --run-name principal_s42
#    protocolo espaço-temporal (o teste mede também os cenários "só futuro" e "só locais novos")
python scripts/stconv_train.py --split-kind spatiotemporal --run-name principal_s42_st --seed 42
python scripts/build_stconv_notebook.py

# 5. Diagnósticos de EDP (somente anos de treino) e calibração dos nulos
python scripts/null_calibration.py
python scripts/pde_diagnostics.py
```

`stconv_train.py --max-events N` roda uma **demonstração** com os N primeiros eventos de cada ano: o relatório
recebe `kind: "demo"`, vai para `reports/demo_runs/` e nunca pode avaliar o teste.

## Protocolos de split

Dois protocolos convivem de propósito.

- **Temporal (fold 2 oficial)**: treino 2018 e 2020, validação 2019, teste 2021. É a divisão ilustrada na
  documentação do dataset e o único protocolo comparável com o benchmark publicado. Foi fixada antes da EDA.
  O split por ano repete regiões: os mesmos incêndios recorrem nos mesmos lugares, e a EDA mede essa
  sobreposição.
- **Espaço-temporal (fold 6 × blocos de 600 km, corredor de exclusão de 10 km)**: cruza blocos geográficos com
  os anos do fold, e cada evento cai em uma de nove células. O teste é **lugar novo e período novo**. Duas
  células intermediárias são avaliadas com o mesmo modelo: **só futuro** (blocos de treino no ano de teste) e
  **só locais novos** (blocos de teste nos anos de treino). Elas separam o custo do deslocamento temporal do
  custo do espacial (`shift_decomposition` no relatório do run). As outras quatro células ficam **seladas**.
  Os gates de viabilidade (validação ≥ 10 eventos e 300 janelas; teste ≥ 20 eventos e 300 janelas) e o
  critério de escolha do fold foram fixados antes de olhar qualquer rótulo. `make_splits.py` recusa escrever
  um fold que não passe neles.
- **Espacial (só blocos)**: diagnóstico da sobreposição e contraparte de volume, não protocolo de avaliação.

## Pré-registro do experimento STConvS2S

Os parâmetros que decidem a leitura ficam numa definição única em `scripts/core.py` (`PREREGISTERED_EPOCHS`,
`PREREGISTERED_RUNS`, `ABLATION_SIGMA_MULTIPLE`, `ABLATION_MIN_ABS_AP`, `ablation_is_relevant`), para que nenhum
consumidor possa afrouxar o critério, ou esticar o orçamento, depois de ver os números.

- **Dados**: `T_in = 5`, alvo = fogo ativo binário em t+1.
- **Configuração principal**: `STConvS2S_C`, `num_layers = 3`, `hidden_dim = 32`, `dropout = 0`, patch 128,
  halo 8; AdamW (`lr = 1e-3`, `weight_decay = 1e-4`), batch 8, AMP; `pos_weight` = negativos/positivos em 2.000
  recortes de treino; canal 17 (precipitação prevista) fora.
- **Orçamento**: 20 épocas, fixado só a partir do custo medido com `--timing-only`, sem busca de
  hiperparâmetros.
- **Configurações e sementes**: principal, sem NDVI/EVI2 (`--drop-groups vegetation_index`) e sem GFS
  (`--drop-groups gfs`), cada uma com as sementes 42, 43 e 44, num total de 9 runs.
- **Seleção**: checkpoint pela maior AP de validação; limiar pelo maior F2 de validação.
- **Métricas**: primária = AP em toda a imagem; secundárias = AP nos pixels novos (`new_vs_t`, `new_vs_window`)
  e no recorte central (`crop32`), F2, precisão e recall no limiar de validação, baselines de prevalência e
  de três persistências. IC 95% por bootstrap de eventos.
- **Ablation relevante**: |efeito| > 2 × desvio-padrão agrupado entre sementes **e** > 0,01 em AP absoluta
  **e** IC 95% do bootstrap pareado de eventos de ΔAP excluindo 0. Uma única semente devolve "não estimável",
  que é diferente de "não relevante".
- **Modelo supera persistência**: IC 95% pareado por eventos de F2(modelo) − F2(persistência) acima de 0.
- **Teste**: uma avaliação por run, registrada no livro-razão `reports/test_ledger.jsonl`. Nenhuma decisão de
  modelagem usa o teste.

O código aceita os dois protocolos (`--split-kind temporal | spatiotemporal`). Ainda não está fixado qual deles
produz o número reportado.

## Semântica dos canais e limitações conhecidas

Verificadas no código de criação do dataset (`DataPreparation/satellites/FirePred.py`) e conferidas pela EDA:

- **Previsões do GFS (canais 17–21)**: rodada 00Z do próprio dia t, então não vazam o alvo t+1. O filtro de
  horizontes compara *strings* e, com IDs de 3 dígitos, seleciona F010–F239. O canal é uma média de horizonte
  longo emitida em t, não a previsão do dia t. A EDA testa as duas hipóteses (`forecast_horizon_check`).
- **Direção do vento prevista (canal 19)**: no Zenodo v1.0 foi calculada com `atan(v/u)`, então perde o
  quadrante. É tratada como dado axial (`sin 2θ`, `cos 2θ`). O vetor de vento usado nos diagnósticos vem do
  GRIDMET.
- **Precipitação prevista (canal 17)**: muda de definição em 2019-11-07 (soma horária → último acumulado). Fica
  fora do modelo principal.
- **NDVI/EVI2 (canais 3–4)**: vêm de um composto VNP13A1 de 16 dias iniciado em [t−7, t], logo contêm
  observações posteriores a t (*look-ahead* estrutural). Por isso o ablation sem esses canais é obrigatório. A
  EDA testa o efeito com diferença-em-diferenças, placebo e dose-resposta (`composite_look_ahead_check`).
- **Grade rotacionada**: todos os eventos estão em EPSG:32610 (UTM 10N), inclusive os distantes do meridiano
  −123°. A grade fica girada pela convergência meridiana γ, e o pixel no terreno mede 375/k m. Vetores e
  ângulos são convertidos com `grid_geometry` e `to_grid_components` (`scripts/core.py`).
- **Antecipações menores**: a cobertura do solo MCD12Q1 é a do ano do fogo, e o PDSI é o do pentad que contém t.
- **Alvo**: o NaN no canal de fogo significa "sem detecção" e vira 0. Pixels não observados (nuvem, omissão do
  VIIRS) entram como negativos, o que é ruído de rótulo. Fogo ativo não é perímetro.
- **Imputação**: média de treino (= 0 após z-score), com flags de ausência por grupo de canais.

Desvios deliberados em relação ao protocolo oficial, que limitam a comparação com as tabelas do artigo:
validação e teste em imagem inteira (oficial: recortes); checkpoint por AP de validação (oficial: `val_loss`);
ângulos com `sin` e `cos` (oficial: só `sin`); AP por histograma em streaming; sem augmentation geométrica;
canal de hora do fogo descartado; `pos_weight` medido nos recortes; um fold (o artigo usa a média de 12).

## Regras que o código faz cumprir

Estão implementadas e cobertas por teste:

- **O ano de teste não é tocado durante o treino.** A avaliação do teste é um modo separado
  (`--evaluate-test`), que parte de um checkpoint com SHA-256 conferido e do limiar já escolhido na validação.
- **Cada avaliação do teste fica registrada** no livro-razão, que só cresce. Um run cujo teste já foi avaliado
  não pode ser retreinado, nem com `--force`.
- **Estatísticas são ajustadas apenas no treino do protocolo**, conferidas em tempo de execução pelas chaves dos
  eventos.
- **Um resultado só se declara publicável** se o código estiver commitado e o manifesto do dataset registrado.
  Caso contrário o JSON traz `publishable: false` e o motivo.
- **A unidade de inferência é o evento**: os nulos randomizam por evento e os intervalos de confiança vêm de
  bootstrap de eventos. `scripts/null_calibration.py` mede o erro tipo I dos nulos em vez de supô-lo.

## Notebooks

| Notebook | Gerado por | Lê |
|---|---|---|
| `04_analise_exploratoria.ipynb` | `scripts/build_eda_notebook.py` | `reports/eda_summary.json` e `reports/figures/` |
| `05_stconv_integracao.ipynb` | `scripts/build_stconv_notebook.py` | `reports/stconv_runs/` (ou `reports/demo_runs/`) e o livro-razão |

Os notebooks não reprocessam o dataset: os builders os geram e executam a partir dos artefatos. As saídas
versionadas são as da execução de referência. Enquanto os runs completos não existirem, o notebook 05 mostra a
demonstração com amostra, sob o aviso de que ela não é evidência.

## Créditos

- Dataset: Gerard, S., Zhao, Y. e Sullivan, J. *WildfireSpreadTS: A dataset of multi-modal time series for
  wildfire spread prediction*. NeurIPS 2023 Datasets and Benchmarks.
- Modelo: STConvS2S (Castro et al., 2021), na branch `noemi` do fork `noemicho/stconvs2s`.
- AILAB — CEFET/RJ.

Este repositório ainda não tem licença definida.
