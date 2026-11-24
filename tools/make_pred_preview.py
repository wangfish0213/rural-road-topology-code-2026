# -*- coding: utf-8 -*-
"""对比图：影像 / 真值 / 预测 三联，用于直观检查断裂与碎片。

用法：
    python make_pred_preview.py --manifest ../manifests/test.csv \
        --pred ../runs/unet_gpu_smoke/pred --out ../previews/pred_compare.png --n 4
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay(img, mask: np.ndarray, color, alpha=0.45) -> np.ndarray:
    """把 mask 以半透明色叠到影像上，返回 uint8 RGB 数组（入参可为 ndarray 或 PIL）。"""
    arr = (np.array(img) if isinstance(img, Image.Image) else img).astype(np.float32)
    arr = arr.copy()
    m = mask > 0
    arr[m] = arr[m] * (1 - alpha) + np.array(color, np.float32) * alpha
    return arr.clip(0, 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../manifests/test.csv")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--skip", type=int, default=0, help="跳过前 N 个有道路的样本")
    args = ap.parse_args(argv)

    with open(args.manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    picked = []
    for r in rows:
        pp = os.path.join(args.pred, r["id"] + ".png")
        if not os.path.exists(pp):
            continue
        gt = np.array(Image.open(r["mask"]))
        if (gt > 0).mean() < 0.005:      # 跳过几乎没有道路的图
            continue
        picked.append((r, pp))
        if len(picked) >= args.n + args.skip:
            break
    picked = picked[args.skip:args.skip + args.n]
    if not picked:
        print("没有可用的样本")
        return 1

    s = args.size
    pad, top, bot = 6, 26, 20
    W = len(picked) * (s + pad) + pad
    H = top + 3 * (s + pad) + bot
    canvas = Image.new("RGB", (W, H), (26, 26, 28))
    d = ImageDraw.Draw(canvas)
    font = None
    for fp in (r"C://Windows//Fonts//msyh.ttc", r"C://Windows//Fonts//simhei.ttf"):
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 14)
                break
            except Exception:
                pass
    def txt(xy, s_, fill=(230, 230, 230)):
        d.text(xy, s_, fill=fill, font=font)
    txt((pad + 2, 6), "影像")
    txt((pad + s + 8, 6), "真值")
    txt((pad + 2 * s + 14, 6), "预测（红=真值，绿=预测）")

    for i, (r, pp) in enumerate(picked):
        x0 = pad + i * (s + pad)
        img = np.array(Image.open(r["image"]).convert("RGB"))
        gt = np.array(Image.open(r["mask"]))
        pr = np.array(Image.open(pp))
        canvas.paste(Image.fromarray(img).resize((s, s)), (x0, top))
        canvas.paste(Image.fromarray(overlay(img, gt, (255, 60, 60))).resize((s, s)), (x0, top + s + pad))
        pr_img = Image.fromarray(overlay(overlay(img, gt, (255, 60, 60)), pr, (60, 220, 120))).resize((s, s))
        canvas.paste(pr_img, (x0, top + 2 * (s + pad)))
        txt((x0 + 2, H - 18), f"{r['id']}  GT {(gt>0).mean()*100:.1f}%  pred {(pr>0).mean()*100:.1f}%", (200, 200, 200))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    print(f"已保存 {args.out}（{len(picked)} 个样本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
