"""Constrói o grafo de regiões: cada nó = uma região de uma versão de uma imagem.

Para cada imagem geramos a versão original + `n_aug` versões aumentadas. A
segmentação é feita só na original; nas aumentadas o mapa de rótulos é
transformado junto com a imagem, então a região r da original e a região r de
cada versão aumentada são a MESMA região (é o par positivo self-supervised).

Saídas por nó:
    x        features handcrafted (cor, textura, forma)
    image    índice da imagem de origem
    version  0 = original, 1..n_aug = aumentadas
    region   id da região na segmentação da original
Arestas iniciais: regiões vizinhas (que se tocam) dentro da mesma versão.
"""
import hashlib
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from imaging import color_augment, geometric_view, load_rgb, segment_cache

LBP_BINS = 10  # LBP uniforme com P=8 -> valores 0..9
FEATURE_NAMES = (
    ["r", "g", "b", "r_std", "g_std", "b_std", "hue_cos", "hue_sin", "sat", "val",
     "L", "a", "b_lab", "L_std", "grad"]
    + [f"lbp{i}" for i in range(LBP_BINS)]
    + ["area", "eccentricity", "solidity"]
)


def region_features(img, seg, ids):
    """img float (H, W, 3) em [0, 1]; seg (H, W) int; ids: regiões a descrever.
    Retorna (len(ids), F)."""
    from skimage.color import rgb2hsv, rgb2lab
    from skimage.feature import local_binary_pattern
    from skimage.filters import sobel
    from skimage.measure import regionprops

    n = int(seg.max()) + 1
    flat = seg.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    safe = np.maximum(area, 1)
    mean = lambda ch: np.bincount(flat, weights=ch.ravel(), minlength=n) / safe
    std = lambda ch, m: np.sqrt(np.maximum(mean(ch ** 2) - m ** 2, 0))

    hsv, lab = rgb2hsv(img), rgb2lab(img)
    gray = img @ np.array([0.299, 0.587, 0.114])
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform").astype(np.int64)
    lbp_hist = np.bincount(flat * LBP_BINS + lbp.ravel(), minlength=n * LBP_BINS).reshape(n, LBP_BINS) / safe[:, None]

    rgb_m = [mean(img[..., c]) for c in range(3)]
    feats = np.stack(
        rgb_m
        + [std(img[..., c], rgb_m[c]) for c in range(3)]
        + [mean(np.cos(2 * np.pi * hsv[..., 0])), mean(np.sin(2 * np.pi * hsv[..., 0])),
           mean(hsv[..., 1]), mean(hsv[..., 2])]
        + [mean(lab[..., c]) for c in range(3)]
        + [std(lab[..., 0], mean(lab[..., 0])), mean(sobel(gray))],
        axis=1)
    shape = np.zeros((n, 2))
    for p in regionprops(seg + 1):  # regionprops ignora o rótulo 0
        if p.label - 1 in ids:
            shape[p.label - 1] = (p.eccentricity, p.solidity)
    feats = np.concatenate([feats, lbp_hist, (area / flat.size)[:, None], shape], axis=1)
    return feats[ids].astype(np.float32)


GLCM_LEVELS = 16
LIT_FEATURE_NAMES = (
    [f"{c}_{m}" for c in ("L", "a", "b") for m in ("mean", "std", "skew")]
    + ["hue_cos", "hue_sin", "sat"]
    + [f"lbp_{c}_r{r}_{i}" for c in ("L", "a", "b") for r in (1, 2) for i in range(LBP_BINS)]
    + [f"grad_{c}_{m}" for c in ("L", "a", "b") for m in ("mean", "std")]
    + [f"glcm_{p}" for p in ("contrast", "homogeneity", "energy", "correlation", "entropy")]
    + ["area", "perimeter_norm", "circularity", "eccentricity", "solidity", "extent", "axis_ratio",
       "dist_center", "touch_background"]
)


def region_features_lit(img, seg, ids, dark=0.08):
    """Features da literatura de qualidade de grão, por região:
    - cor: momentos (média, desvio, assimetria) em L*a*b* + matiz/saturação
    - textura: LBP uniforme multicanal (L, a, b; raios 1 e 2), na linha do MCLBP
      (Chandra & Yohannes 2026), magnitude do gradiente por canal (o "+M"), e
      GLCM (contraste, homogeneidade, energia, correlação, entropia) só com os
      pixels da região
    - forma/posição: área, perímetro, circularidade, excentricidade, solidez,
      extensão, razão de eixos, distância ao centro da semente, contato com o fundo
    img float (H, W, 3) em [0, 1]."""
    from scipy.ndimage import binary_dilation
    from skimage.color import rgb2hsv, rgb2lab
    from skimage.feature import graycomatrix, local_binary_pattern
    from skimage.filters import sobel
    from skimage.measure import regionprops

    n = int(seg.max()) + 1
    flat = seg.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    safe = np.maximum(area, 1)
    mean = lambda ch: np.bincount(flat, weights=ch.ravel(), minlength=n) / safe

    lab = rgb2lab(img)
    lab_n = np.stack([lab[..., 0] / 100, (lab[..., 1] + 128) / 255, (lab[..., 2] + 128) / 255], -1)
    hsv = rgb2hsv(img)
    cols = []
    for c in range(3):
        ch = lab[..., c]
        m = mean(ch)
        var = np.maximum(mean(ch ** 2) - m ** 2, 0)
        sd = np.sqrt(var)
        m3 = mean(ch ** 3) - 3 * m * mean(ch ** 2) + 2 * m ** 3
        cols += [m, sd, m3 / np.maximum(sd ** 3, 1e-6)]
    cols += [mean(np.cos(2 * np.pi * hsv[..., 0])), mean(np.sin(2 * np.pi * hsv[..., 0])), mean(hsv[..., 1])]
    feats = [np.stack(cols, 1)]

    for c in range(3):
        u8 = (np.clip(lab_n[..., c], 0, 1) * 255).astype(np.uint8)
        for r in (1, 2):
            lbp = local_binary_pattern(u8, P=8, R=r, method="uniform").astype(np.int64)
            feats.append(np.bincount(flat * LBP_BINS + lbp.ravel(), minlength=n * LBP_BINS)
                         .reshape(n, LBP_BINS) / safe[:, None])
    g = []
    for c in range(3):
        gm = sobel(lab_n[..., c])
        m = mean(gm)
        g += [m, np.sqrt(np.maximum(mean(gm ** 2) - m ** 2, 0))]
    feats.append(np.stack(g, 1))

    # GLCM por região: L quantizado em 1..16, fora da região = 0 (descartado)
    Lq = 1 + np.minimum((np.clip(lab_n[..., 0], 0, 1) * GLCM_LEVELS).astype(np.int64), GLCM_LEVELS - 1)
    glcm = np.zeros((n, 5))
    seed_mask = img.mean(2) >= dark
    ys, xs = np.nonzero(seed_mask)
    cy, cx = (ys.mean(), xs.mean()) if len(ys) else (img.shape[0] / 2, img.shape[1] / 2)
    radius = np.sqrt(max(seed_mask.sum(), 1) / np.pi)
    shape = np.zeros((n, 9))
    background = binary_dilation(~seed_mask)
    for p in regionprops(seg + 1):
        r = p.label - 1
        if r not in ids:
            continue
        r0, c0, r1, c1 = p.bbox
        q = np.where(p.image, Lq[r0:r1, c0:c1], 0).astype(np.uint8)
        M = graycomatrix(q, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=GLCM_LEVELS + 1)
        M = M[1:, 1:].sum((2, 3)).astype(np.float64)
        M = M + M.T
        if M.sum() > 0:
            P = M / M.sum()
            i, j = np.indices(P.shape)
            mu_i, mu_j = (i * P).sum(), (j * P).sum()
            sd_i, sd_j = np.sqrt(((i - mu_i) ** 2 * P).sum()), np.sqrt(((j - mu_j) ** 2 * P).sum())
            glcm[r] = [((i - j) ** 2 * P).sum(), (P / (1 + np.abs(i - j))).sum(), (P ** 2).sum(),
                       ((i - mu_i) * (j - mu_j) * P).sum() / max(sd_i * sd_j, 1e-6),
                       -(P[P > 0] * np.log(P[P > 0])).sum()]
        per = max(p.perimeter, 1.0)
        maj, mnr = p.axis_major_length, max(p.axis_minor_length, 1e-6)
        dist = np.hypot(p.centroid[0] - cy, p.centroid[1] - cx) / max(radius, 1)
        touch = background[r0:r1, c0:c1][p.image].mean()
        shape[r] = [p.area / seg.size, per / np.sqrt(p.area), 4 * np.pi * p.area / per ** 2, p.eccentricity,
                    p.solidity, p.extent, maj / mnr, dist, touch]
    feats += [glcm, shape]
    return np.concatenate(feats, 1)[ids].astype(np.float32)


MCLBP_PAIRS = [("L", "L"), ("a", "a"), ("b", "b"), ("L", "a"), ("L", "b"), ("a", "b")]  # (centro, vizinhos)
MCLBP_RADII = (1, 2)
HUE_BINS = 12


def _riu2(bits):
    """bits (P, H, W) -> código LBP uniforme invariante a rotação: 0..P, P+1 = não uniforme."""
    P = bits.shape[0]
    trans = np.abs(bits - np.roll(bits, 1, axis=0)).sum(0)
    return np.where(trans <= 2, bits.sum(0), P + 1).astype(np.int64)


def mclbp_m_codes(lab_n, P=8):
    """MCLBP+M (Shu et al. 2021; usado neste dataset por Chandra & Yohannes 2026).
    Para cada par (canal do centro, canal dos vizinhos) e raio R:
      sinal:     bit_p = [viz_p >= centro]
      magnitude: bit_p = [|viz_p - centro| >= média global de |viz - centro|]
    Pares iguais = LBP intra-canal; pares diferentes = correlação entre canais."""
    from scipy.ndimage import shift
    ch = {"L": lab_n[..., 0], "a": lab_n[..., 1], "b": lab_n[..., 2]}
    codes = []
    for R in MCLBP_RADII:
        offsets = [(-R * np.sin(2 * np.pi * p / P), R * np.cos(2 * np.pi * p / P)) for p in range(P)]
        nb = {c: np.stack([shift(v, (-dy, -dx), order=1, mode="nearest") for dy, dx in offsets]) for c, v in ch.items()}
        for cc, cn in MCLBP_PAIRS:
            diff = nb[cn] - ch[cc][None]
            mag = np.abs(diff)
            codes.append(_riu2((diff >= 0).astype(np.int8)))
            codes.append(_riu2((mag >= mag.mean()).astype(np.int8)))
    return codes  # lista de mapas (H, W) com valores 0..P+1


LIT2_FEATURE_NAMES = (
    LIT_FEATURE_NAMES
    + [f"mclbp_{cc}{cn}_r{R}_{k}_{i}" for R in MCLBP_RADII for cc, cn in MCLBP_PAIRS for k in ("s", "m")
       for i in range(LBP_BINS)]
    + [f"huehist_{i}" for i in range(HUE_BINS)]
)


def region_features_lit2(img, seg, ids):
    """lit + MCLBP+M multicanal (sinal e magnitude, 6 pares de canais, raios 1 e 2)
    + histograma de matiz ponderado pela saturação (Jitanan & Chimlek 2019)."""
    from skimage.color import rgb2hsv, rgb2lab
    base = region_features_lit(img, seg, ids)
    n = int(seg.max()) + 1
    flat = seg.ravel()
    safe = np.maximum(np.bincount(flat, minlength=n), 1).astype(np.float64)
    lab = rgb2lab(img)
    lab_n = np.stack([lab[..., 0] / 100, (lab[..., 1] + 128) / 255, (lab[..., 2] + 128) / 255], -1)
    hists = [np.bincount(flat * LBP_BINS + c.ravel(), minlength=n * LBP_BINS).reshape(n, LBP_BINS) / safe[:, None]
             for c in mclbp_m_codes(lab_n)]
    hsv = rgb2hsv(img)
    hb = np.minimum((hsv[..., 0] * HUE_BINS).astype(np.int64), HUE_BINS - 1)
    w = hsv[..., 1].ravel()
    hh = np.bincount(flat * HUE_BINS + hb.ravel(), weights=w, minlength=n * HUE_BINS).reshape(n, HUE_BINS)
    hh = hh / np.maximum(hh.sum(1, keepdims=True), 1e-6)
    return np.concatenate([base] + [h[ids] for h in hists] + [hh[ids]], 1).astype(np.float32)


FEATURE_SETS = {"basic": (region_features, FEATURE_NAMES), "lit": (region_features_lit, LIT_FEATURE_NAMES),
                "lit2": (region_features_lit2, LIT2_FEATURE_NAMES)}

# grupos de features (para equilibrar o peso de cada grupo no cosseno)
GROUP_PREFIXES = [("color", ("L_", "a_", "b_", "hue_", "sat", "r", "g", "b", "L", "a")),
                  ("mclbp", ("mclbp_",)), ("lbp", ("lbp",)), ("grad", ("grad",)), ("glcm", ("glcm_",)),
                  ("huehist", ("huehist_",))]


def feature_groups(names):
    """Grupo de cada feature: prefixos específicos primeiro; o que sobra é forma."""
    groups = []
    for n in names:
        g = "shape"
        for gname, prefixes in [p for p in GROUP_PREFIXES if p[0] != "color"] + [GROUP_PREFIXES[0]]:
            if n.startswith(prefixes) and not (gname == "color" and n in (
                    "area", "eccentricity", "solidity", "extent", "axis_ratio", "perimeter_norm", "circularity")):
                g = gname
                break
        groups.append(g)
    return np.array(groups)


def standardize(x, names, balance=False):
    """z-score por feature; com balance=True cada grupo é dividido por sqrt(dim do grupo),
    para todos os grupos pesarem igual no cosseno."""
    xs = (x - x.mean(0)) / (x.std(0) + 1e-6)
    if balance:
        g = feature_groups(names)
        for name in np.unique(g):
            m = g == name
            xs[:, m] /= np.sqrt(m.sum())
    return xs


def adjacent_pairs(seg, keep):
    """Pares (a, b) de regiões mantidas que se tocam (vizinhança 4)."""
    a = np.concatenate([seg[:, 1:].ravel(), seg[1:, :].ravel()])
    b = np.concatenate([seg[:, :-1].ravel(), seg[:-1, :].ravel()])
    m = (a != b) & keep[a] & keep[b]
    pairs = np.unique(np.sort(np.stack([a[m], b[m]], 1), axis=1), axis=0)
    return pairs


def _image_nodes(job):
    idx, path, seg0, n_aug, seed, crop_scale, min_area, min_area_aug, ignore_dark, feature_set = job
    feat_fn, feat_names = FEATURE_SETS[feature_set]
    torch.set_num_threads(1)
    random.seed(seed * 1_000_003 + idx)
    img0 = load_rgb(path)
    seg0 = seg0.astype(np.int64)
    n = int(seg0.max()) + 1

    # regiões válidas definidas na ORIGINAL: descarta minúsculas e fundo preto
    flat = seg0.ravel()
    area = np.bincount(flat, minlength=n)
    bright = np.bincount(flat, weights=(img0.mean(2) / 255).ravel(), minlength=n) / np.maximum(area, 1)
    keep0 = (area >= min_area * flat.size) & (bright >= ignore_dark)

    versions = [(img0.astype(np.float32) / 255, seg0)]
    img_t = torch.from_numpy(img0.copy()).permute(2, 0, 1).float() / 255
    seg_t = torch.from_numpy(seg0)
    for _ in range(n_aug):
        v, s = geometric_view(img_t, seg_t, img0.shape[0], crop_scale)
        versions.append((color_augment(v).permute(1, 2, 0).numpy(), s.numpy()))

    feats, meta, edges, offset = [], [], [], 0
    for vi, (img, seg) in enumerate(versions):
        a = np.bincount(seg.ravel(), minlength=n)
        keep = keep0 & (a >= (min_area if vi == 0 else min_area_aug) * seg.size)
        ids = np.flatnonzero(keep)
        if len(ids) == 0:
            continue
        feats.append(feat_fn(img, seg, ids))
        meta.append(np.stack([np.full(len(ids), idx), np.full(len(ids), vi), ids], 1))
        local = np.full(n, -1)
        local[ids] = np.arange(len(ids)) + offset
        pairs = adjacent_pairs(seg, keep)
        if len(pairs):
            edges.append(local[pairs])
        offset += len(ids)
    if not feats:
        return np.zeros((0, len(feat_names)), np.float32), np.zeros((0, 3), np.int64), np.zeros((0, 2), np.int64)
    return (np.concatenate(feats), np.concatenate(meta).astype(np.int64),
            np.concatenate(edges) if edges else np.zeros((0, 2), np.int64))


def build_region_graph(files, root, segmenter="felzenszwalb", n_aug=1, seed=0, crop_scale=(0.5, 1.0),
                       min_area=0.005, min_area_aug=0.002, ignore_dark=0.08, workers=12, cache_dir="cache",
                       feature_set="basic"):
    params = f"{segmenter}|{n_aug}|{seed}|{crop_scale}|{min_area}|{min_area_aug}|{ignore_dark}|" + "|".join(map(str, files))
    if feature_set != "basic":  # mantém a chave antiga para o conjunto basic
        params += f"|features={feature_set}"
    key = hashlib.md5(params.encode()).hexdigest()[:10]
    path = Path(cache_dir) / f"graph_{feature_set}_aug{n_aug}_seed{seed}_{key}.npz"
    old = Path(cache_dir) / f"graph_aug{n_aug}_seed{seed}_{key}.npz"
    if feature_set == "basic" and old.exists():
        path = old
    if not path.exists():
        segs = segment_cache(files, root, segmenter, cache_dir=cache_dir, workers=workers)
        jobs = [(i, str(f), np.asarray(segs[i]), n_aug, seed, crop_scale, min_area, min_area_aug, ignore_dark,
                 feature_set)
                for i, f in enumerate(files)]
        print(f"extraindo regiões de {len(files)} imagens (n_aug={n_aug}) -> {path}")
        with Pool(workers) as pool:
            out = pool.map(_image_nodes, jobs, chunksize=16)
        xs, metas, edges, offset = [], [], [], 0
        for x, meta, e in out:
            xs.append(x)
            metas.append(meta)
            edges.append(e + offset)
            offset += len(x)
        np.savez(path, x=np.concatenate(xs), meta=np.concatenate(metas), edges=np.concatenate(edges))
    d = np.load(path)
    return d["x"], d["meta"], d["edges"]


def select_features(x, names, groups=None):
    """Mantém só as colunas dos grupos pedidos (ex.: "color,shape,mclbp"). None/"all" = tudo."""
    names = np.asarray(names)
    if not groups or groups == "all":
        return x, names
    wanted = set(groups.split(","))
    g = feature_groups(names)
    unknown = wanted - set(g)
    if unknown:
        raise ValueError(f"grupos desconhecidos {sorted(unknown)}; opções: {sorted(set(g))}")
    m = np.isin(g, list(wanted))
    return x[:, m], names[m]
