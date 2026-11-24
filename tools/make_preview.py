# -*- coding: utf-8 -*-
"""生成样本预览图：影像 + 掩膜叠加，用于肉眼抽查标注质量。

用法：
    python make_preview.py --root ../datasets/train --out ../previews/samples_train.png --n 6
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect(root: str, img_suffix: str, mask_suffix: str):
    pairs = []
    for dirpath, _, files in os.walk(root):
        imgs = {f[: -len(img_suffix)]: f for f in files if f.endswith(img_suffix)}
        masks = {f[: -len(mask_suffix)]: f for f in files if f.endswith(mask_suffix)}
        for k in sorted(set(imgs) & set(masks)):
            pairs.append((os.path.join(dirpath, imgs[k]), os.path.join(dirpath, masks[k])))
    return pairs


def overlay(img: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    m = (mask > 0)
    arr[m] = arr[m] * (1 - alpha) + np.array([255, 40, 40], np.float32) * alpha
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--images-suffix", default="_sat.jpg")
    ap.add_argument("--masks-suffix", default="_mask.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    pairs = collect(args.root, args.images_suffix, args.masks_suffix)
    if not pairs:
        print("没有找到样本")
        return 1
    rng = random.Random(args.seed)
    # 一半随机、一半挑前景占比高的，保证图里能看到东西
    sample = rng.sample(pairs, min(args.n // 2, len(pairs)))
    rest = [p for p in pairs if p not in sample]
    sample += rng.sample(rest, min(args.n - len(sample), len(rest)))

    cols = 3 if len(sample) > 4 else 2
    rows = (len(sample) + cols - 1) // cols
    cell, pad, label_h = args.cell, 8, 22
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (28, 28, 30))
    draw = ImageDraw.Draw(canvas)

    for i, (ip, mp) in enumerate(sample):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        with Image.open(ip) as im, Image.open(mp) as mi:
            comp = overlay(im, np.array(mi))
        canvas.paste(comp.resize((cell, cell), Image.BILINEAR), (x, y))
        fg = float((np.array(Image.open(mp)) > 0).mean()) * 100
        draw.text((x + 2, y + cell + 5), f"{os.path.basename(ip)}  road {fg:.1f}%", fill=(220, 220, 220))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    print(f"已保存 {args.out}（{len(sample)} 个样本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
