"""Seção 3 do relatório: visão GCN (embedding aprendido).

    python full_gcn/gcn.py          # treina no split original e exporta full_gcn/runs/gcn/Z.npy

Texto -> código:
  - "Todas as regiões de todas as imagens, originais e aumentadas, formam um único grafo com as arestas
    de adjacência da Seção 1": X (lit, 92), E0 (adjacência).
  - GCN de duas camadas com atalho linear:  Z = Â(Â X W1 + b1) W2 + b2 + 0,3 (X W0 + b0),
    Â = D~^(-1/2) (A + I) D~^(-1/2)  (GCNConv), z_v em R^5.
  - l(i, p, n; mu) = max(0, mu - [<z_i, z_p> - <z_i, z_n>])  (produto interno);
    L = w_ssl L_ssl + w_rank L_rank + w_comm L_comm.
  - L_ssl : âncora = região AUMENTADA, positivo = a sua região original, negativo = região de outra imagem.
  - L_rank: âncora = região de treino, positivo = região de treino da mesma classe de OUTRA imagem,
            negativo = região de treino de outra classe.
  - L_comm: periodicamente o grafo é reconstruído com os pares de embeddings MAIS SIMILARES (top-k global),
            e dele se extraem comunidades por propagação de rótulos. Pares de treino de classes diferentes
            que caem na MESMA comunidade (a do rebuild atual) viram negativos difíceis: âncora = região de
            treino numa comunidade com treino de outra classe, negativo = essa região de outra classe,
            positivo = região de treino da mesma classe de outra imagem.
  - Seleção do checkpoint: o treino para quando o agrupamento nessas comunidades deixa de melhorar na
    validação (F das comunidades 100% puras no treino, como no #57); salva o melhor ponto.
  - Hiperparâmetros: os da configuração #57 (config_gcn57.json).
  - Sem vazamento: só rótulos de treino no treino; a validação só escolhe o checkpoint.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_data as data  # noqa: E402  (faz chdir para a raiz do repositório)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "gcn"


class GCN(nn.Module):
    """Z = Â(Â X W1 + b1) W2 + b2 + 0,3 (X W0 + b0)."""

    def __init__(self, d_in, d_out=5, hidden=16):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)        # W0, b0
        self.conv1 = GCNConv(d_in, hidden)       # W1, b1
        self.conv2 = GCNConv(hidden, d_out)      # W2, b2

    def forward(self, x, e):
        return self.conv2(self.conv1(x, e), e) + 0.3 * self.lin(x)


def triplet_loss(Z, i, p, n, mu):
    return torch.relu(mu - ((Z[i] * Z[p]).sum(1) - (Z[i] * Z[n]).sum(1))).mean()


def resample(idx, bad, pool_fn):
    while bad().any():
        b = bad()
        idx[b] = pool_fn(int(b.sum()))
    return idx


def pick(pool, n):
    return pool[torch.randint(len(pool), (n,), device=DEV)]


class Sampler:
    def __init__(self, y, img, ver, reg, train):
        self.y, self.img = y, img
        self.train = train
        self.per_class = {c: train[y[train] == c] for c in range(int(y.max()) + 1)}
        self.N = len(y)
        key = img.long() * 100_000 + reg.long()
        orig = torch.nonzero(ver == 0).squeeze(1)
        aug = torch.nonzero(ver > 0).squeeze(1)
        orig_of = dict(zip(key[orig].tolist(), orig.tolist()))
        pairs = [(j, orig_of[k]) for k, j in zip(key[aug].tolist(), aug.tolist()) if k in orig_of]
        self.ssl = torch.tensor(pairs, device=DEV)  # (região aumentada, sua região original)

    def same_class_other_image(self, anchors):
        ya = self.y[anchors]
        p = torch.empty_like(anchors)
        for c, pool in self.per_class.items():
            m = ya == c
            p[m] = pick(pool, int(m.sum()))
        bad = lambda: self.img[p] == self.img[anchors]  # noqa: E731
        while bad().any():
            b = bad()
            for c, pool in self.per_class.items():
                m = b & (ya == c)
                p[m] = pick(pool, int(m.sum()))
        return p

    def ssl_triplets(self, n):
        s = pick(self.ssl, n)
        i, p = s[:, 0], s[:, 1]
        neg = torch.randint(self.N, (n,), device=DEV)
        neg = resample(neg, lambda: self.img[neg] == self.img[i], lambda k: torch.randint(self.N, (k,), device=DEV))
        return i, p, neg

    def rank_triplets(self, n):
        i = pick(self.train, n)
        p = self.same_class_other_image(i)
        neg = pick(self.train, n)
        neg = resample(neg, lambda: self.y[neg] == self.y[i], lambda k: pick(self.train, k))
        return i, p, neg


class CommunityPairs:
    """Pares (i, j) de treino, de classes diferentes, na MESMA comunidade do rebuild atual."""

    def __init__(self, comm, y, train):
        tr = train.cpu().numpy()
        c, yy = comm[tr], y.cpu().numpy()[tr]
        n_cls = np.bincount(np.unique(np.stack([c, yy], 1), axis=0)[:, 0], minlength=comm.max() + 1)
        mixed = n_cls[c] >= 2
        order = np.lexsort((yy[mixed], c[mixed]))
        self.nodes = tr[mixed][order]
        self.c = c[mixed][order]
        self.yy = yy[mixed][order]
        self.mixed_frac = float(mixed.mean())
        s = np.searchsorted(self.c, self.c, "left")
        e = np.searchsorted(self.c, self.c, "right")
        t = lambda a: torch.tensor(a, device=DEV)  # noqa: E731
        self.t_nodes, self.t_y, self.t_s, self.t_len = t(self.nodes), t(self.yy), t(s), t(e - s)

    def sample(self, n, sampler):
        if len(self.nodes) == 0:
            return None
        k = torch.randint(len(self.nodes), (n,), device=DEV)
        i, yi = self.t_nodes[k], self.t_y[k]
        neg = torch.full_like(i, -1)
        todo = torch.ones_like(i, dtype=torch.bool)
        while todo.any():  # sorteia na mesma comunidade até achar classe diferente (existe por construção)
            pos = self.t_s[k] + (torch.rand(n, device=DEV) * self.t_len[k]).long()
            ok = todo & (self.t_y[pos] != yi)
            neg[ok] = self.t_nodes[pos][ok]
            todo &= ~ok
        return i, sampler.same_class_other_image(i), neg


def topk_pairs(Z, k):
    """Os k pares (u < v) de maior cosseno no grafo todo (exato, árvore k-d; cosseno = distância nos
    vetores normalizados). Confere a exatidão contando os pares abaixo do limiar."""
    from scipy.spatial import cKDTree
    X = torch.nn.functional.normalize(Z, dim=1).cpu().numpy().astype(np.float64)
    n, m = len(X), 10
    tree = cKDTree(X)
    while True:
        d, nb = tree.query(X, k=min(m + 1, n), workers=-1)
        u = np.repeat(np.arange(n), nb.shape[1])
        v, dist = nb.ravel(), d.ravel()
        ok = (v < n) & (v != u)
        key, ix = np.unique(np.minimum(u[ok], v[ok]).astype(np.int64) * n + np.maximum(u[ok], v[ok]), return_index=True)
        dist = dist[ok][ix]
        sel = np.argpartition(dist, k - 1)[:k]
        T = dist[sel].max()
        if (dist[sel] < T).sum() == (tree.count_neighbors(tree, np.nextafter(T, 0)) - n) // 2:
            break
        m *= 2
    return np.stack([key[sel] // n, key[sel] % n], 1)


def label_propagation(pairs, N, seed):
    import igraph as ig
    random.seed(seed)
    return np.array(ig.Graph(n=N, edges=pairs.tolist()).community_label_propagation().membership)


def pure_filter(comm, node_img, img_y, img_split, m_train):
    """Comunidades 100% puras no treino (>= m_train imagens de treino distintas, todas da mesma classe);
    na validação: acerto (pares comunidade-imagem na classe do treino) e cobertura (imagens com região
    em alguma dessas comunidades). F = média harmônica."""
    pairs = np.unique(np.stack([comm, node_img], 1), axis=0)
    c, im = pairs[:, 0], pairs[:, 1]
    tr = img_split[im] == "train"
    counts = np.zeros((comm.max() + 1, int(img_y.max()) + 1), np.int64)
    np.add.at(counts, (c[tr], img_y[im[tr]]), 1)
    pure = (counts.sum(1) >= m_train) & ((counts > 0).sum(1) == 1)
    out = {"pure_comms": int(pure.sum())}
    for s in ("val", "test"):
        m = (img_split[im] == s) & pure[c]
        out[f"{s}_acc"] = float((img_y[im[m]] == counts.argmax(1)[c[m]]).mean()) if m.any() else 0.0
        out[f"{s}_cov"] = float(len(np.unique(im[m])) / (img_split == s).sum())
        a, v = out[f"{s}_acc"], out[f"{s}_cov"]
        out[f"{s}_f"] = 2 * a * v / (a + v) if a + v else 0.0
    return out


def train(hp, out):
    random.seed(hp["seed"])
    np.random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])
    x, meta, edges0, img_y, img_split = data.region_graph("lit")
    node_img = meta[:, 0]
    N = len(x)
    X = torch.tensor(x, device=DEV)
    E0 = to_undirected(torch.tensor(edges0.T, dtype=torch.long, device=DEV), num_nodes=N)
    y = torch.tensor(img_y[node_img], device=DEV)
    img = torch.tensor(node_img, device=DEV)
    train_nodes = torch.nonzero(torch.tensor(img_split[node_img] == "train", device=DEV)).squeeze(1)
    S = Sampler(y, img, torch.tensor(meta[:, 1], device=DEV), torch.tensor(meta[:, 2], device=DEV), train_nodes)
    print(f"{N} vértices, {E0.size(1) // 2} arestas de adjacência, {len(S.ssl)} pares orig<->aug, "
          f"{len(train_nodes)} regiões de treino", flush=True)
    model = GCN(X.size(1), hp["emb_dim"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    comm_pairs, best, stale, log = None, -1.0, 0, []
    for rb in range(1, hp["max_rebuilds"] + 1):
        t0 = time.time()
        model.train()
        for _ in range(hp["rebuild_interval"]):
            Z = model(X, E0)
            parts = {"ssl": triplet_loss(Z, *S.ssl_triplets(hp["ssl_batch"]), hp["ssl_margin"]),
                     "rank": triplet_loss(Z, *S.rank_triplets(hp["rank_batch"]), hp["rank_margin"])}
            loss = hp["w_ssl"] * parts["ssl"] + hp["w_rank"] * parts["rank"]
            if comm_pairs is not None:
                t = comm_pairs.sample(hp["comm_batch"], S)
                if t is not None:
                    parts["comm"] = triplet_loss(Z, *t, hp["comm_margin"])
                    loss = loss + hp["w_comm"] * parts["comm"]
            opt.zero_grad()
            loss.backward()
            opt.step()
        # rebuild: pares mais similares -> comunidades -> negativos difíceis da L_comm
        model.eval()
        with torch.no_grad():
            Z = model(X, E0)
        pairs = topk_pairs(Z, hp["k_gsl"])
        comm = label_propagation(pairs, N, hp["seed"] * 1000 + rb)
        comm_pairs = CommunityPairs(comm, y, train_nodes)
        pf = pure_filter(comm, node_img, img_y, img_split, hp["pure_min_train"])
        same = (img_y[node_img[pairs[:, 0]]] == img_y[node_img[pairs[:, 1]]]).mean()
        rec = {"rebuild": rb, "epoch": rb * hp["rebuild_interval"], "loss": float(loss),
               **{k: float(v) for k, v in parts.items()}, "communities": int(comm.max() + 1),
               "train_mixed_frac": comm_pairs.mixed_frac, "pairs_same_class": float(same),
               **{f"pure_{k}": v for k, v in pf.items()}, "time": time.time() - t0}
        log.append(rec)
        print(f"rebuild {rb:02d} | loss {rec['loss']:.3f} (ssl {rec['ssl']:.3f} rank {rec['rank']:.3f} "
              f"comm {rec.get('comm', 0):.3f}) | {rec['communities']} comunidades, treino misturado "
              f"{rec['train_mixed_frac']:.1%}, pares top-k mesma classe {same:.1%} | puras {pf['pure_comms']}: "
              f"val acerto {pf['val_acc']:.1%} cob {pf['val_cov']:.1%} F {pf['val_f']:.3f} | {rec['time']:.0f}s",
              flush=True)
        if pf["val_f"] > best:
            best, stale = pf["val_f"], 0
            torch.save({"model": model.state_dict(), "hp": hp, "rebuild": rb}, out / "best.pt")
            np.save(out / "Z.npy", Z.cpu().numpy().astype(np.float32))  # z_v no grafo de adjacência
            np.save(out / "communities.npy", comm)
        else:
            stale += 1
        if rb > hp["early_rebuilds"] and stale >= hp["patience"]:
            print(f"parada no rebuild {rb} (melhor val F {best:.3f})")
            break
    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))


if __name__ == "__main__":
    hp = json.loads((HERE / "config_gcn57.json").read_text())
    out = OUT
    v = 2
    while out.exists():  # não sobrescreve
        out = OUT.with_name(f"gcn_v{v}")
        v += 1
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(hp, indent=2))
    train(hp, out)
    print("saída:", out)
