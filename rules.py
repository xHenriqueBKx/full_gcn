"""Seções 4 e 5 do relatório: regras de subgrafos em cada visão e decisão por concordância.

Texto -> código (o mesmo procedimento em cada visão, só com as regiões das imagens ORIGINAIS):
  - Candidatos: cand_tau(v, I) = { u em kNN_K(v) ∩ V_I : cos(v, u) >= tau }, para cada outra imagem I
    (kNN_K = as K regiões mais parecidas com v entre as de outras imagens). Na própria imagem de origem, a
    ocorrência trivial (a identidade) conta.
  - Padrão P = subgrafo conexo do grafo de uma imagem de treino. P ocorre em I se existe phi injetivo com
    phi(p) em cand_tau(p, I) para todo p e (p, q) em E_P => (phi(p), phi(q)) em E_I.
  - Suporte sigma(P) = nº de imagens de TREINO em que P ocorre.
  - Mineração Apriori por níveis: padrões de 1 região; estende cada padrão por uma região vizinha na
    imagem de origem, estendendo junto as ocorrências; mantém enquanto sigma >= sigma_min. Sem limite de
    tamanho. Guardam-se no máximo M mapeamentos por imagem.
  - Regra da classe c: padrão que ocorre em >= m imagens de treino, TODAS da classe c. Cada regra vota em c
    em toda imagem de validação/teste em que ocorre; a visão prediz a classe mais votada.
  - Decisão (Seção 5): a imagem recebe classe só se as duas visões votaram e na mesma classe.
Salvaguarda: se o nº de padrões de um nível passar de MAX_PATTERNS, levanta erro (o trial falha) em vez
de truncar em silêncio, para que "sem limite de tamanho" valha para todo resultado reportado.
"""
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PATTERNS = 2_000_000


class TooManyPatterns(RuntimeError):
    pass


def knn(vectors, nodes, node_img, K, block=256):
    """Para cada região original: as K mais parecidas (cosseno) entre as regiões originais de OUTRAS imagens."""
    E = F.normalize(torch.tensor(vectors[nodes], dtype=torch.float32, device=DEV), dim=1)
    img = torch.tensor(node_img[nodes], device=DEV)
    nb, sims = [], []
    for s in range(0, len(nodes), block):
        sim = E[s:s + block] @ E.T
        sim[img[s:s + block, None] == img[None, :]] = -2
        v, i = sim.topk(K, dim=1)
        nb.append(i.cpu().numpy())
        sims.append(v.cpu().numpy())
    return nodes[np.concatenate(nb)], np.concatenate(sims)


def mine(nodes, node_img, img_split, adj, nbrs, sims, tau, sigma_min, M):
    """Devolve [(imagens em que o padrão ocorre)] para todos os padrões com sigma >= sigma_min."""
    is_train = img_split == "train"
    cand = {}
    for row, v in enumerate(nodes):
        d = defaultdict(list)
        d[node_img[v]].append(v)                      # ocorrência trivial na própria imagem
        for u, s in zip(nbrs[row], sims[row]):
            if s >= tau:
                d[node_img[u]].append(u)
        cand[v] = d
    sigma = lambda maps: sum(is_train[i] for i in maps)  # noqa: E731

    level = {}
    for v in nodes:
        if is_train[node_img[v]]:
            maps = {i: [(u,) for u in us[:M]] for i, us in cand[v].items()}
            if sigma(maps) >= sigma_min:
                level[(v,)] = maps
    frequent = {p[0] for p in level}
    patterns = []
    while level:
        patterns.extend(tuple(maps) for maps in level.values())
        nxt, seen = {}, set()
        for key, maps in level.items():
            S = set(key)
            for w in {u for v in key for u in adj[v]} - S:     # região vizinha na imagem de origem
                if w not in frequent:
                    continue
                fs = frozenset(S | {w})
                if fs in seen:
                    continue
                seen.add(fs)
                links = [j for j, v in enumerate(key) if v in adj[w]]
                new = {}
                for im, ms in maps.items():                 # estende as ocorrências
                    cw = cand[w].get(im)
                    if not cw:
                        continue
                    out = []
                    for m in ms:
                        used = set(m)
                        for u in cw:
                            if u not in used and all(m[j] in adj[u] for j in links):
                                out.append(m + (u,))
                                if len(out) >= M:
                                    break
                        if len(out) >= M:
                            break
                    if out:
                        new[im] = out
                if sigma(new) >= sigma_min:
                    nxt[key + (w,)] = new
                    if len(nxt) > MAX_PATTERNS:
                        raise TooManyPatterns(f"mais de {MAX_PATTERNS} padrões num nível")
        level = nxt
    return patterns


def vote(patterns, img_y, img_split, m):
    """Regras (>= m imagens de treino, todas da mesma classe) e voto: {imagem val/teste: classe mais votada}."""
    votes = defaultdict(Counter)
    n_rules = 0
    for imgs in patterns:
        tr = [i for i in imgs if img_split[i] == "train"]
        if len(tr) < m or len(set(img_y[tr].tolist())) != 1:
            continue
        n_rules += 1
        c = int(img_y[tr[0]])
        for i in imgs:
            if img_split[i] != "train":
                votes[i][c] += 1
    return {i: v.most_common(1)[0][0] for i, v in votes.items()}, n_rules


def decide(pred_full, pred_gcn):
    """Seção 5: classe só quando as duas visões votaram e na mesma classe."""
    return {i: c for i, c in pred_full.items() if pred_gcn.get(i) == c}


def score(pred, img_y, img_split, s):
    imgs = np.flatnonzero(img_split == s)
    got = [i for i in imgs if i in pred]
    acc = float(np.mean([pred[i] == img_y[i] for i in got])) if got else 0.0
    return acc, len(got) / len(imgs)
