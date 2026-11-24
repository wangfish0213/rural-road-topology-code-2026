# -*- coding: utf-8 -*-
"""滑窗推理：对 1024 切片逐块预测并写回全分辨率概率图，输出二值掩膜 PNG。

用法：
    python predict.py --ckpt ../runs/unet_smoke/best.pt --manifest ../manifests/test.csv \
        --out ../runs/unet_smoke/pred --limit 100

输出的 PNG 与真值同名（去掉 _mask 后），可直接交给 eval/eval_dir.py 评测。
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from unet import UNet

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(ck):
    """按权重文件里的记录构建模型：U-Net / UNetOri（含方向场头）/ torchvision 分割模型。"""
    if "arch" in ck:
        from train_tv import ARCHS
        return ARCHS[ck["arch"]](False), "tv"
    has_ori = any(k.startswith("ori_head") for k in ck["model"].keys())
    if has_ori:
        from orientation import UNetOri
        return UNetOri(base=ck["args"].get("base", 32)), "ori"
    return UNet(base=ck["args"].get("base", 32)), "unet"


def tile_coords(h: int, w: int, tile: int, stride: int):
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    return [(y, x) for y in ys for x in xs]


@torch.no_grad()
def predict_image(model, img: np.ndarray, dev, tile: int, stride: int, batch: int = 8,
                  kind: str = "unet") -> np.ndarray:
    h, w = img.shape[:2]
    prob = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    coords = tile_coords(h, w, tile, stride)
    buf, buf_coords = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().to(dev) / 255.0
        if kind in ("unet", "ori"):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        else:
            # torchvision 分割模型：ImageNet 归一化 + 取道路类 softmax
            x = (x - MEAN.to(dev)) / STD.to(dev)
            p = torch.softmax(model(x)["out"], 1)[:, 1].cpu().numpy()
        for (y, xx), pi in zip(buf_coords, p):
            prob[y:y + tile, xx:xx + tile] += pi
            weight[y:y + tile, xx:xx + tile] += 1.0
        buf.clear(); buf_coords.clear()

    for y, x in coords:
        buf.append(img[y:y + tile, x:x + tile])
        buf_coords.append((y, x))
        if len(buf) >= batch:
            flush()
    flush()
    weight[weight == 0] = 1.0
    return prob / weight


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="../manifests/whu_rroad_test.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-prob", action="store_true", help="同时保存概率图为 .npy，便于后处理调阈值")
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                       (args.device if args.device != "auto" else "cpu"))
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model, kind = build_model(ck)
    model = model.to(dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=(kind != "ori"))
    if kind == "ori":
        print(f"方向场权重按非严格加载（缺 {len(missing)} / 多 {len(unexpected)} 个键，属正常）")
    print(f"模型类型 {kind}" + (f"（{ck.get('arch')}）" if "arch" in ck else ""))
    eta = ck.get("epoch", "?")
    print(f"权重第 {eta} 轮，验证 IoU {ck.get('val_iou', float('nan')):.4f}")
    model.eval()

    with open(args.manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    if args.save_prob:
        os.makedirs(os.path.join(args.out, "prob"), exist_ok=True)

    for i, r in enumerate(rows, 1):
        img = np.array(Image.open(r["image"]).convert("RGB"))
        prob = predict_image(model, img, dev, args.tile, args.stride, args.batch, kind)
        Image.fromarray(((prob > args.thr) * 255).astype(np.uint8)).save(
            os.path.join(args.out, r["id"] + ".png"))
        if args.save_prob:
            np.save(os.path.join(args.out, "prob", r["id"] + ".npy"), prob.astype(np.float16))
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)
    print(f"完成，掩膜写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
