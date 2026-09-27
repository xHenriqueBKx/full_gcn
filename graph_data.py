"""Seções 1 e 2 do relatório: grafo de regiões de cada imagem e visão full.

Usa o extrator de regiões/features de region_graph.py, que segue o texto:
  - Felzenszwalb (scale 300, sigma 0,8, min_size 100);
  - descarta regiões com área < 0,5% da imagem ou brilho médio < 0,08 (fundo preto);
  - arestas entre regiões que se tocam;
  - uma versão aumentada por imagem (recorte, espelhos, rotação de 90°, cor), com o MESMO mapa de regiões
    transformado: a região r da aumentada é a região física r da original.
Visão full: x_v com 344 features (lit2), padronizadas por z-score, sem rótulo.
Visão GCN (entrada): features lit (92), padronizadas por z-score.
Caminhos relativos à raiz do repositório: data/images/, data/split.json, cache/ (gerado).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from region_graph import FEATURE_SETS, build_region_graph, select_features, standardize  # noqa: E402

SPLIT = "data/split.json"            # 70/20/10 estratificado, sem as 311 duplicatas
MIN_AREA = 0.005                       # 0,5% da imagem, nas duas versões
DARK = 0.08


def split():
    s = json.loads(Path(SPLIT).read_text())
    items = sorted((it["file"], it["label"], k) for k in ("train", "val", "test") for it in s[k])
    return [Path(f) for f, _, _ in items], np.array([y for _, y, _ in items]), np.array([k for *_, k in items])


def region_graph(features):
    """x (padronizado), meta (imagem, versão 0=original/1=aumentada, id da região), arestas de adjacência."""
    files, img_y, img_split = split()
    x, meta, edges = build_region_graph(files, "data/images", "felzenszwalb", n_aug=1, crop_scale=(0.5, 1.0),
                                        min_area=MIN_AREA, min_area_aug=MIN_AREA, ignore_dark=DARK,
                                        feature_set=features)
    groups = "all" if features == "lit2" else None
    xs, names = select_features(x, FEATURE_SETS[features][1], groups)
    return standardize(xs, names, False).astype(np.float32), meta, edges, img_y, img_split
