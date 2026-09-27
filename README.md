# full_gcn

Classificação de sementes de soja por **regras de subgrafos** com duas visões de cada região: features
manuais (**full**) e o embedding de uma rede neural de grafos (**GCN**). Uma imagem só recebe classe quando
as regras das duas visões votam na mesma classe; nas outras, o classificador se abstém. Por isso o
resultado tem dois números: **acerto** (entre as imagens classificadas) e **cobertura** (fração das
imagens classificadas).

O código segue o [relatório](docs/relatorio_full_gcn.pdf) (`docs/relatorio_full_gcn.pdf`), seção por seção.

## Reproduzir

```bash
pip install -r requirements.txt
python gcn.py        # Seção 3: treina a GCN -> runs/gcn/Z.npy          (~3 min com GPU)
python evaluate.py   # Seções 4-6: regras com a configuração do relatório (~8 min, CPU)
```

Para refazer a busca dos parâmetros das regras (Seção 6):

```bash
python search.py --trials 30 --workers 3   # Optuna/TPE, só na validação (~2 h)
python search.py --report
```

A primeira execução extrai as regiões e as features de todas as imagens e guarda em `cache/`
(~15 min com 12 núcleos); as seguintes reaproveitam.

### Resultado esperado (split do repositório)

| | Acerto | Cobertura |
|---|---|---|
| Validação | 89,8% | 67,6% |
| Teste | 86,6% | 64,6% |

Configuração: K = 188, M = 8, σ_min = 1, τ_full = 0,67, m_full = 2, τ_GCN = 0,94, m_GCN = 1.
O treino da GCN na GPU não é bit a bit determinístico, então os números podem variar um pouco.

## Dados

`data/images/` tem as 5.202 imagens usadas (5 classes: Broken, Immature, Intact, Skin-damaged, Spotted),
já sem as 311 duplicatas, redimensionadas de 227×227 para 224×224 e salvas em PNG sem perda.
`data/split.json` é a divisão 70/20/10 estratificada: 3.641 de treino, 1.041 de validação e 520 de teste.
Rótulos só são usados no treino; a validação escolhe os hiperparâmetros e o teste é reportado uma vez.

### Origem e licença

Dataset **Soybean Seeds** (5.513 imagens, 5 classes, recortes de 227×227 de fotos de 3072×2048,
classificados segundo a norma GB1352-2009), distribuído sob
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> Lin, Wei; Fu, Youhao; Xu, Peiquan; Liu, Shuo; Ma, Daoyi; Jiang, Zitian; Zang, Siyang; Yao, Heyang;
> Su, Qin (2023), "Soybean Seeds", Mendeley Data, V5, doi: [10.17632/v6vzvfszj6.5](https://doi.org/10.17632/v6vzvfszj6.5)

Obtido pelo Kaggle: <https://www.kaggle.com/datasets/aryashah2k/soybean-seedsclassification-dataset>.
**Modificações feitas aqui:** remoção de 311 imagens duplicadas; redimensionamento de 227×227 para 224×224
(LANCZOS); conversão de BMP para PNG sem perda.

## Arquivos

| [Relatório](docs/relatorio_full_gcn.pdf) | Arquivo |
|---|---|
| 1. Grafo de regiões de cada imagem | `graph_data.py`, `region_graph.py`, `imaging.py`, `segmenters.py` |
| 2. Visão full (344 features, z-score) | `graph_data.py`, `region_graph.py` |
| 3. Visão GCN | `gcn.py`, `config_gcn57.json` |
| 4. Regras de subgrafos | `rules.py` |
| 5. Decisão: as duas visões concordam | `rules.decide` |
| 6. Hiperparâmetros e resultado | `search.py`, `evaluate.py` |

## Detalhes de implementação

Pontos em que o texto principal do relatório não fixa exatamente o que fazer; o
[relatório](docs/relatorio_full_gcn.pdf) os descreve em notas nas Seções 3 e 4, e o código segue isto:

1. **Hiperparâmetros da GCN (configuração #57, `config_gcn57.json`).** Adam com taxa de aprendizado
   0,0026 e decaimento de peso 0,0015; margens μ_ssl = 0,16, μ_rank = 0,52 e μ_comm = 0,90; pesos
   w_ssl = 1,20, w_rank = 0,99 e w_comm = 0,49; 1.024 triplets de cada perda por época; rebuild a cada
   200 épocas com os 99.160 pares mais similares; no máximo 25 rebuilds.
2. **Parada do treino da GCN.** "O agrupamento nas comunidades deixa de melhorar na validação" é medido
   pelas comunidades 100% puras no treino (pelo menos 2 imagens de treino, todas da mesma classe): acerto e
   cobertura delas na validação, resumidos pela média harmônica (F). O treino roda pelo menos 5 rebuilds e
   para após 3 rebuilds seguidos sem melhora; fica o checkpoint do melhor ponto.
3. **Ocorrência na imagem de origem.** Os candidatos são definidos para as outras imagens; na imagem de
   onde o padrão foi tirado, ele sempre ocorre (cada região mapeada nela mesma). Por isso, a imagem de
   origem entra no suporte.
4. **Empate no voto.** Se duas classes empatam em número de votos, a visão fica com a primeira delas na
   ordem em que os votos foram contados.
5. **Salvaguarda da mineração.** Mesmo sem limite de tamanho, alguns parâmetros (limiar baixo, suporte
   mínimo 1) podem fazer o número de padrões explodir. Se um nível da mineração passar de 2 milhões de
   padrões, a configuração é descartada na busca (`rules.TooManyPatterns`), em vez de seguir com os padrões
   cortados. Assim, todo resultado reportado foi minerado sem corte; na busca, a trava nunca disparou.
