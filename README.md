# full_gcn

Classificação de sementes de soja por **regras de subgrafos** com duas visões de cada região: features
manuais (**full**) e o embedding de uma rede neural de grafos (**GCN**). Uma imagem só recebe classe quando
as regras das duas visões votam na mesma classe; nas outras, o classificador se abstém. Por isso o
resultado tem dois números: **acerto** (entre as imagens classificadas) e **cobertura** (fração das
imagens classificadas).

O código segue o relatório `relatorio_full_gcn`, seção por seção.

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
já sem as 311 duplicatas, em 224×224 (as originais de 227×227 preenchidas com preto até ficarem
quadradas e redimensionadas), salvas em PNG sem perda.
`data/split.json` é a divisão 70/20/10 estratificada: 3.641 de treino, 1.041 de validação e 520 de teste.
Rótulos só são usados no treino; a validação escolhe os hiperparâmetros e o teste é reportado uma vez.

## Arquivos

| Relatório | Arquivo |
|---|---|
| 1. Grafo de regiões de cada imagem | `graph_data.py`, `region_graph.py`, `imaging.py`, `segmenters.py` |
| 2. Visão full (344 features, z-score) | `graph_data.py`, `region_graph.py` |
| 3. Visão GCN | `gcn.py`, `config_gcn57.json` |
| 4. Regras de subgrafos | `rules.py` |
| 5. Decisão: as duas visões concordam | `rules.decide` |
| 6. Hiperparâmetros e resultado | `search.py`, `evaluate.py` |

## Escolhas que o relatório não fixa

- Hiperparâmetros da GCN: os da configuração #57 (`config_gcn57.json`).
- Parada do treino da GCN: F (média harmônica de acerto e cobertura) das comunidades 100% puras no treino
  (>= 2 imagens de treino) na validação; mínimo de 5 rebuilds e paciência de 3.
- Ocorrência de um padrão na própria imagem de origem: a identidade conta.
- Empate no voto: a primeira classe mais votada.
- Salvaguarda da mineração: se um nível passar de 2 milhões de padrões, o trial falha (em vez de truncar).
