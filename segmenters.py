"""Segmentadores plugáveis.

Todo segmentador recebe uma imagem RGB uint8 (H, W, 3) e devolve um mapa de
rótulos int (H, W), onde cada valor inteiro é uma região. É só isso que o resto
do pipeline precisa, então trocar o método de segmentação = registrar uma nova
função aqui.

Especificação por string, com parâmetros opcionais:
    "felzenszwalb"                      -> parâmetros padrão
    "felzenszwalb:scale=500,min_size=200"
    "slic:n_segments=25"
    "grid:n=3"
    "precomputed:dir=masks_sam"         -> lê PNGs de rótulos gerados fora daqui
"""
from pathlib import Path

import numpy as np
from PIL import Image

SEGMENTERS = {}


def register(name):
    def deco(fn):
        SEGMENTERS[name] = fn
        return fn
    return deco


@register("grid")
def grid(img, n=4):
    """Baseline espacial do DetCon: grade n x n."""
    h, w = img.shape[:2]
    rows = np.minimum(np.arange(h) * n // h, n - 1)
    cols = np.minimum(np.arange(w) * n // w, n - 1)
    return rows[:, None] * n + cols[None, :]


@register("felzenszwalb")
def felzenszwalb(img, scale=300, sigma=0.8, min_size=100):
    from skimage.segmentation import felzenszwalb as fz
    return fz(img, scale=scale, sigma=sigma, min_size=min_size)


@register("slic")
def slic(img, n_segments=16, compactness=10):
    from skimage.segmentation import slic as sl
    return sl(img, n_segments=n_segments, compactness=compactness, start_label=0)


@register("precomputed")
def precomputed(img, dir, path=None):
    """Máscaras geradas externamente (ex.: SAM), salvas como PNG de rótulos
    com o mesmo caminho relativo da imagem original."""
    return np.asarray(Image.open(Path(dir) / Path(path).with_suffix(".png")))


def _parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_spec(spec):
    name, _, args = spec.partition(":")
    kwargs = dict(kv.split("=", 1) for kv in args.split(",") if kv)
    return name, {k: _parse_value(v) for k, v in kwargs.items()}


def get_segmenter(spec):
    name, kwargs = parse_spec(spec)
    if name not in SEGMENTERS:
        raise ValueError(f"segmentador '{name}' desconhecido; opções: {sorted(SEGMENTERS)}")
    fn = SEGMENTERS[name]

    def segment(img, path=None):
        if name == "precomputed":
            labels = fn(img, path=path, **kwargs)
        else:
            labels = fn(img, **kwargs)
        # rótulos consecutivos 0..R-1
        _, labels = np.unique(labels, return_inverse=True)
        return labels.reshape(img.shape[:2]).astype(np.int16)

    return segment
