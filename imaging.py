"""Dataset de pares de views (original aumentada 2x) + mapas de regiões alinhados.

A segmentação é feita UMA vez na imagem original e cacheada em disco. Em cada
amostra, as mesmas transformações geométricas (crop, flips, rotações) são
aplicadas à imagem e ao mapa de rótulos, então a região r da view 1 corresponde
à região r da view 2 — é essa correspondência que a loss contrastiva usa.
"""
import hashlib
import math
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from segmenters import get_segmenter

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root):
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    files, targets = [], []
    for ci, c in enumerate(classes):
        for f in sorted((root / c).iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                files.append(f)
                targets.append(ci)
    return files, np.array(targets), classes


def stratified_split(targets, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c in np.unique(targets):
        idx = rng.permutation(np.flatnonzero(targets == c))
        n_test = int(round(len(idx) * test_frac))
        test += idx[:n_test].tolist()
        train += idx[n_test:].tolist()
    return np.array(sorted(train)), np.array(sorted(test))


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------- cache de segmentação

def _segment_one(args):
    spec, root, path = args
    return get_segmenter(spec)(load_rgb(path), path=Path(path).relative_to(root))


def segment_cache(files, root, spec, cache_dir="cache", workers=8):
    """Mapas de rótulos (N, H, W) para `files`, cacheados por (segmentador, lista de arquivos)."""
    key = hashlib.md5((spec + "|" + "|".join(map(str, files))).encode()).hexdigest()[:10]
    safe = spec.replace(":", "_").replace(",", "_").replace("=", "-").replace("/", "-")
    path = Path(cache_dir) / f"seg_{safe}_{key}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"segmentando {len(files)} imagens com '{spec}' -> {path}")
        with Pool(workers) as pool:
            segs = pool.map(_segment_one, [(spec, root, str(f)) for f in files], chunksize=32)
        segs = np.stack(segs)
        segs = segs.astype(np.uint8 if segs.max() < 256 else np.int16)
        np.save(path, segs)
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------- augmentations

def random_resized_crop_box(h, w, scale, ratio=(3 / 4, 4 / 3)):
    for _ in range(10):
        area = h * w * random.uniform(*scale)
        r = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = int(round(math.sqrt(area * r))), int(round(math.sqrt(area / r)))
        if 0 < cw <= w and 0 < ch <= h:
            return random.randint(0, h - ch), random.randint(0, w - cw), ch, cw
    return 0, 0, h, w


def gaussian_blur(img, sigma):
    k = int(2 * math.ceil(3 * sigma) + 1)
    x = torch.arange(k, dtype=img.dtype) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    img = F.conv2d(F.pad(img[None], (k // 2, k // 2, 0, 0), mode="reflect"),
                   g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    img = F.conv2d(F.pad(img, (0, 0, k // 2, k // 2), mode="reflect"),
                   g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    return img[0]


def gray(img):
    return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])[None]


def color_augment(img):
    if random.random() < 0.8:
        ops = [
            lambda x: x * random.uniform(0.6, 1.4),  # brilho
            lambda x: (x - gray(x).mean()) * random.uniform(0.6, 1.4) + gray(x).mean(),  # contraste
            lambda x: gray(x) + (x - gray(x)) * random.uniform(0.8, 1.2),  # saturação
        ]
        random.shuffle(ops)
        for op in ops:
            img = op(img).clamp(0, 1)
    if random.random() < 0.2:
        img = gray(img).expand(3, -1, -1)
    if random.random() < 0.5:
        img = gaussian_blur(img, random.uniform(0.1, 2.0))
    return img.clamp(0, 1)


def geometric_view(img, seg, size, scale):
    """Mesmas operações geométricas na imagem (bilinear) e nos rótulos (nearest)."""
    _, h, w = img.shape
    i, j, ch, cw = random_resized_crop_box(h, w, scale)
    img = F.interpolate(img[None, :, i:i + ch, j:j + cw], size=(size, size),
                        mode="bilinear", align_corners=False, antialias=True)[0]
    seg = F.interpolate(seg[None, None, i:i + ch, j:j + cw].float(), size=(size, size),
                        mode="nearest")[0, 0].long()
    if random.random() < 0.5:
        img, seg = img.flip(-1), seg.flip(-1)
    if random.random() < 0.5:
        img, seg = img.flip(-2), seg.flip(-2)
    rot = random.randint(0, 3)  # sementes não têm orientação canônica
    return img.rot90(rot, (-2, -1)), seg.rot90(rot, (-2, -1))


# ---------------------------------------------------------------- dataset

class RegionPairDataset(Dataset):
    """Retorna (view1, seg1, view2, seg2, region_ids).

    region_ids: (K,) ids de regiões sorteadas na imagem original; -1 = padding.
    A região region_ids[k] em seg1 é a correspondente de region_ids[k] em seg2.
    """

    def __init__(self, files, segs, k=16, size=224, crop_scale=(0.3, 1.0),
                 min_area=0.005, ignore_dark=0.08):
        self.files, self.segs = files, segs
        self.k, self.size, self.crop_scale = k, size, crop_scale
        self.min_area, self.ignore_dark = min_area, ignore_dark

    def __len__(self):
        return len(self.files)

    def sample_regions(self, img, seg):
        n = int(seg.max()) + 1
        flat = seg.flatten()
        area = torch.bincount(flat, minlength=n).float()
        brightness = torch.bincount(flat, weights=img.mean(0).flatten(), minlength=n) / area.clamp_min(1)
        # descarta regiões minúsculas e o fundo preto (idêntico entre imagens -> falsos negativos)
        ok = (area >= self.min_area * flat.numel()) & (brightness >= self.ignore_dark)
        cand = torch.nonzero(ok).flatten().tolist()
        random.shuffle(cand)
        cand = cand[:self.k]
        return torch.tensor(cand + [-1] * (self.k - len(cand)), dtype=torch.long)

    def __getitem__(self, idx):
        img = torch.from_numpy(load_rgb(self.files[idx]).copy()).permute(2, 0, 1).float() / 255
        seg = torch.from_numpy(self.segs[idx].astype(np.int64))
        ids = self.sample_regions(img, seg)
        out = []
        for _ in range(2):
            v, s = geometric_view(img, seg, self.size, self.crop_scale)
            out += [(color_augment(v) - MEAN) / STD, s]
        return (*out, ids)


class PlainDataset(Dataset):
    """Imagens sem augmentation, para avaliação (kNN / linear probe)."""

    def __init__(self, files, targets):
        self.files, self.targets = files, targets

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = torch.from_numpy(load_rgb(self.files[idx]).copy()).permute(2, 0, 1).float() / 255
        return (img - MEAN) / STD, int(self.targets[idx])
