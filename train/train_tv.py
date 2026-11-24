# -*- coding: utf-8 -*-
"""torchvision 分割基线（DeepLabv3+ / LRASPP / FCN），与 train_unet.py 同数据同设置。

为什么单列一个脚本：这几种模型的输出是 (B,2,H,W) 的 logits（背景/道路两类），
输入需要 ImageNet 归一化，与 U-Net 的单通道 sigmoid 口径不同，混在一个脚本里容易出错。

用法：
    python train_tv.py --arch deeplabv3_mobilenet_v3_large --epochs 12 --limit-train 5435
    python train_tv.py --arch deeplabv3_resnet50 --pretrained --epochs 8 --limit-train 2000
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from unet import RoadTile

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ARCHS = {
    "deeplabv3_mobilenet_v3_large": lambda pretrained: torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone="DEFAULT" if pretrained else None, num_classes=2),
    "lraspp_mobilenet_v3_large": lambda pretrained: torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=None, weights_backbone="DEFAULT" if pretrained else None, num_classes=2),
    "deeplabv3_resnet50": lambda pretrained: torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None, weights_backbone="DEFAULT" if pretrained else None, num_classes=2),
    "fcn_resnet50": lambda pretrained: torchvision.models.segmentation.fcn_resnet50(
        weights=None, weights_backbone="DEFAULT" if pretrained else None, num_classes=2),
}


def fg_scores(logits: torch.Tensor, target: torch.Tensor):
    """前景 IoU / F1 / P / R。logits: (B,2,H,W)，target: (B,1,H,W) 的 0/1。"""
    pred = logits.argmax(1, keepdim=True).float()
    t = (target > 0.5).float()
    tp = (pred * t).sum().item()
    fp = (pred * (1 - t)).sum().item()
    fn = ((1 - pred) * t).sum().item()
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return iou, f1, prec, rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCHS))
    ap.add_argument("--manifest", default="../manifests/whu_road_1m_train.csv")
    ap.add_argument("--val-manifest", default="../manifests/whu_road_1m_val.csv")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=5.0, help="道路类的交叉熵权重")
    ap.add_argument("--limit-train", type=int, default=5435)
    ap.add_argument("--limit-val", type=int, default=120)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet 预训练骨干（省轮次）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="../runs/tv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="从 last.pt 继续训练")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    os.makedirs(args.out, exist_ok=True)

    tr = RoadTile(args.manifest, args.crop, args.limit_train, train=True)
    va = RoadTile(args.val_manifest, args.crop, args.limit_val, train=False)
    print(f"[{args.arch}] 训练 {len(tr)} / 验证 {len(va)} | crop {args.crop} | batch {args.batch} | 预训练 {args.pretrained}")
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    dlv = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = ARCHS[args.arch](args.pretrained).to(dev)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量 {n_param:.1f} M | 设备 {dev}")

    mean, std = MEAN.to(dev), STD.to(dev)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_weight], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * steps))

    log = os.path.join(args.out, "train_log.csv")
    if not os.path.exists(log):
        with open(log, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["epoch", "iter", "loss", "sec_per_iter"])

    best, start_ep = -1.0, 1
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optim" in ck:
            opt.load_state_dict(ck["optim"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        start_ep = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("best_iou", ck.get("val_iou", -1.0)))
        print(f"续训：从第 {start_ep} 轮开始（最佳 IoU {best:.4f}）")

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0, seen, run = time.time(), 0, 0.0
        for it, (img, msk) in enumerate(dl, 1):
            img = ((img.to(dev) - mean) / std)
            msk = msk.to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(img)["out"]
            loss = crit(out, msk[:, 0].long())
            loss.backward()
            opt.step()
            sched.step()
            seen += 1
            run += loss.item()
            if it % 50 == 0 or it == 1:
                print(f"  ep{ep} it{it}/{steps} loss {run/seen:.4f} {(time.time()-t0)/seen:.2f}s/it", flush=True)
            with open(log, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([ep, it, f"{loss.item():.5f}", f"{(time.time()-t0)/seen:.4f}"])

        model.eval()
        ious, f1s = [], []
        with torch.no_grad():
            for img, msk in dlv:
                img = (img.to(dev) - mean) / std
                out = model(img)["out"]
                iou, f1, *_ = fg_scores(out, msk.to(dev))
                ious.append(iou); f1s.append(f1)
        miou = float(np.mean(ious))
        print(f"epoch {ep} 完成，用时 {time.time()-t0:.0f}s | 验证 IoU {miou:.4f} F1 {np.mean(f1s):.4f}", flush=True)
        ck = {"arch": args.arch, "model": model.state_dict(), "args": vars(args), "epoch": ep,
              "val_iou": miou, "best_iou": best, "optim": opt.state_dict(), "sched": sched.state_dict()}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if miou > best:
            best = miou
            ck["best_iou"] = best
            torch.save(ck, os.path.join(args.out, "best.pt"))
    print(f"[{args.arch}] 最佳验证 IoU {best:.4f}，权重在 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
