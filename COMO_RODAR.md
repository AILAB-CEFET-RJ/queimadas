# Como rodar tudo com um comando

`scripts/run_pipeline.py` faz o caminho inteiro sem supervisão: confere o ambiente e a GPU, clona os repositórios
de terceiros no commit fixado, baixa o WildfireSpreadTS do Zenodo, converte para HDF5, treina os folds pedidos e
empacota os resultados num zip. Cada etapa é retomável. **Depois de qualquer queda, rode o mesmo comando de novo**:
o que terminou é pulado e o que parou continua de onde estava.

## Requisitos

| Item | Mínimo | Observação |
|---|---|---|
| GPU NVIDIA | 6 GB de VRAM | O treino exige CUDA e para se a GPU sumir. Usa AMP fp16. |
| Driver NVIDIA | >= 580 para RTX 50xx; >= 560 para as demais | `nvidia-smi` precisa estar no PATH. |
| Python | 3.12 | O lock foi validado no 3.12 (`numpy 2.2.6` não tem wheel para 3.14). |
| git | qualquer | Para os clones fixados. |
| Disco | **~80 GiB livres** no disco dos dados | Zip de ~45 GiB + HDF5 de ~24 GiB + folga. Com `--delete-zip`, sobram ~25 GiB depois da conversão. |
| RAM | 16 GiB | Com menos, use `--num-workers 2` (não muda o resultado). |
| Internet | ~45 GiB do Zenodo | O download retoma do ponto em que parou e confere o MD5. |

**RTX 50xx (5090, 5080…)**: estas placas (arquitetura sm_120) precisam do PyTorch com CUDA >= 12.8. O
`requirements-cuda.txt` traz o build cu126, que não tem kernels para elas. Use `requirements-cuda130.txt` (mesma
versão do torch, build CUDA 13.0). O pipeline confere isso antes de baixar qualquer coisa e diz o que instalar.

## Passo a passo (Linux)

```bash
# 1. Código e ambiente (uma vez)
git clone https://github.com/AILAB-CEFET-RJ/queimadas
cd queimadas
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -r requirements-cuda130.txt   # RTX 50xx; outras placas: requirements-cuda.txt

# 2. Conferência rápida (~1 min): Python, pacotes, GPU com AMP, disco e clones. Não baixa nada.
python scripts/run_pipeline.py --check-only --data-root /dados/wildfirespreadts

# 3. O comando único, dentro de um tmux para sobreviver ao fechamento do terminal
tmux new -s wsts
source .venv/bin/activate
systemd-inhibit --what=sleep:idle --why="treino WSTS" \
  python scripts/run_pipeline.py --train tin1_es_rtx --data-root /dados/wildfirespreadts --delete-zip
# Ctrl-b d desanexa; tmux attach -t wsts volta
```

`/dados/wildfirespreadts` é só um exemplo: use uma pasta num disco com o espaço da tabela. Sem `--data-root`, os
dados vão para `dados_geoespaciais/wildfirespreadts/` dentro do clone. Numa máquina que não suspende, o
`systemd-inhibit` pode sair.

Para escolher a GPU numa máquina com mais de uma, acrescente `--gpu 0` (índice ou UUID do `nvidia-smi -L`).

## Passo a passo (Windows, PowerShell)

```powershell
git clone https://github.com/AILAB-CEFET-RJ/queimadas
cd queimadas
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -r requirements-cuda130.txt
python scripts\run_pipeline.py --check-only --data-root D:\dados\wildfirespreadts
python scripts\run_pipeline.py --train tin1_es_rtx --data-root D:\dados\wildfirespreadts --delete-zip
```

No Windows o executor pede ao sistema para não suspender enquanto roda. Deixe o computador na tomada.

## O que acontece, em ordem

1. **Ambiente** (segundos): Python, pacotes, `nvidia-smi`, um teste do torch na GPU com uma convolução em fp16,
   espaço em disco e RAM. Qualquer problema para aqui, com a correção na mensagem. Grava `reports/ambiente.json` e
   `reports/pipeline_preflight.json`.
2. **Clones**: `.stconvs2s` (o modelo) e `.wildfirespreadts` (referência oficial), no commit fixado.
3. **Download** (~45 GiB; o tempo depende do link: a 10 MiB/s, cerca de 1h20): `reports/download_manifest.json`.
4. **Conversão** para HDF5 (~30 min): um arquivo por evento, conferido antes de entrar. `reports/hdf5_manifest.json`
   confere as contagens do Zenodo (607 eventos, 13.607 imagens).
5. **Run curto de conferência** (~1 min): fold 2, 2 eventos por ano, 4 passos. Nunca avalia o teste. Se falhar, o
   pipeline para antes das horas de treino.
6. **Treino** dos 12 folds, do zero, um de cada vez (`tin1_es_rtx`: `T_in = 1`, receita `paper_es`, runs
   `tin1_es_rtx_f00` a `tin1_es_rtx_f11`): treina, escolhe o checkpoint na validação e avalia o teste uma
   única vez. Numa GPU de 6 GB cada fold levou de 2 a 17 h; numa RTX 5090 deve ser bem menos.
7. **Pacote** `reports/entrega_tin1_es_rtx_<data>.zip`.

## Retomada

O treino começa do zero nesta máquina: nenhum fold vem de outra rodada. A retomada abaixo vale só para quedas
durante a própria execução. Rode exatamente o mesmo comando. Nada é refeito à toa:

- o download continua do arquivo `.part` e confere o MD5 no fim;
- a conversão pula os eventos que já têm HDF5 e sidecar conferidos;
- um fold com relatório completo é pulado;
- um fold interrompido continua do `last.pt` da última época (em `<data-root>/runs/<run>/last.pt`). O `last.pt` é
  gravado a cada época, então uma queda perde no máximo uma época.

Duas cópias do pipeline nos mesmos dados não rodam ao mesmo tempo (trava em `<data-root>/.pipeline.lock`; se a
máquina caiu, a trava velha é ignorada).

Se a GPU sumir no meio (driver, superaquecimento), o executor para sem iniciar outros folds, em vez de cair para a
CPU. Um fold que fica 45 min sem terminar uma época é encerrado pelo watchdog. Nos dois casos, resolva a causa e rode
o comando de novo.

## O que devolver

No fim, o pipeline imprime o nome do pacote: `reports/entrega_tin1_es_rtx_<data>.zip`. Ele traz o resumo do
experimento, o relatório JSON e o log de cada fold, o livro-razão do teste (`test_ledger.jsonl`), os manifestos dos
dados e o registro do ambiente (GPU, versões). É só esse arquivo que precisa voltar para quem pediu o run. O log do
próprio pipeline fica em `reports/logs/pipeline.log`.

Se algum fold ficar pendente, a última linha diz quais. Rode o comando de novo.

## Não faça

- **Não use `--force`** e não apague `reports/test_ledger.jsonl`: o teste de cada run é avaliado uma única vez, e o
  livro-razão é o que garante isso.
- **Não mude `--accumulate`** (padrão 2): os folds são comparados pareados com os de outras máquinas, com o mesmo
  lote efetivo.
- **Não troque `tin1_es_rtx` por `tin1_es`**: o `tin1_es` é uma rodada anterior, feita em outra máquina. O
  `tin1_es_rtx` refaz os 12 folds do zero, com nomes e resumo próprios, para que as duas nunca se misturem.

## Outras opções

```bash
python scripts/run_pipeline.py --train none            # só baixa e converte os dados
python scripts/run_pipeline.py --train paper_es        # baseline (receita paper_es) nos 12 folds
python scripts/run_pipeline.py --smoke                  # para depois do run curto de conferência
python scripts/run_pipeline.py --help
```
