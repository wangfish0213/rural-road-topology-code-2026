# -*- coding: utf-8 -*-
"""训练入口：本机预实验用，参数默认值刻意设小，跑通优先于刷分。

用法：
    python train_unet.py --epochs 3 --limit-train 1200 --batch 4 --crop 256 --device cuda
    python train_unet.py --epochs 1 --limit-train 200 --max-iters 50 --device cpu   # 冒烟测试
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from unet import BCEDiceLoss, RoadTile, UNet, pixel_scores


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../manifests/whu_rroad_train.csv")
    ap.add_argument("--val-manifest", default="../manifests/whu_rroad_test.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--pos-weight", type=float, default=6.0)
    ap.add_argument("--limit-train", type=int, default=1200)
    ap.add_argument("--limit-val", type=int, default=100)
    ap.add_argument("--max-iters", type=int, default=0, help="每轮最多迭代次数，0 表示不限")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="CUDA 下用混合精度，省显存")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="../runs/unet_smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="从 last.pt 继续训练（含优化器与调度器状态）")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("警告：请求 cuda 但不可用，回退到 cpu")
        dev = torch.device("cpu")

    print(f"设备：{dev}" + (f"（{torch.cuda.get_device_name(0)}，"
                           f"{torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB）"
                           if dev.type == "cuda" else ""))
    print("数据集：")
    tr = RoadTile(args.manifest, args.crop, args.limit_train, train=True)
    va = RoadTile(args.val_manifest, args.crop, args.limit_val, train=False)
    print(f"  训练 {len(tr)} 张，验证 {len(va)} 张，裁剪 {args.crop}×{args.crop}，batch {args.batch}")
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = UNet(base=args.base).to(dev)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量 {n_param:.2f} M")

    crit = BCEDiceLoss(pos_weight=args.pos_weight).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(dl_tr) if not args.max_iters else min(args.max_iters, len(dl_tr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and dev.type == "cuda"))

    log_path = os.path.join(args.out, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["epoch", "iter", "loss", "sec_per_iter"])

    best_iou, start_ep = -1.0, 1
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optim" in ck:
            opt.load_state_dict(ck["optim"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        start_ep = int(ck.get("epoch", 0)) + 1
        best_iou = float(ck.get("best_iou", ck.get("val_iou", -1.0)))
        print(f"续训：从第 {start_ep} 轮开始（已完成 {start_ep-1} 轮，最佳 IoU {best_iou:.4f}）")

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        for it, (img, msk) in enumerate(dl_tr, 1):
            img, msk = img.to(dev, non_blocking=True), msk.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(img)
            # 损失强制在 fp32 下算：fp16 的 BCEWithLogits(pos_weight) 会溢出成 NaN
            loss = crit(logits.float(), msk)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            running += loss.item()
            n_seen += 1
            if it % 20 == 0 or it == 1:
                print(f"  ep{ep} it{it}/{steps_per_epoch} loss {running/n_seen:.4f} "
                      f"{(time.time()-t0)/n_seen:.2f}s/it", flush=True)
            with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([ep, it, f"{loss.item():.6f}", f"{(time.time()-t0)/n_seen:.4f}"])
            if args.max_iters and it >= args.max_iters:
                break

        model.eval()
        ious, f1s, precs, recs = [], [], [], []
        with torch.no_grad():
            for img, msk in dl_va:
                img, msk = img.to(dev), msk.to(dev)
                logits = model(img)
                iou, f1, p, r = pixel_scores(logits, msk)
                ious.append(iou); f1s.append(f1); precs.append(p); recs.append(r)
        miou, mf1 = float(np.mean(ious)), float(np.mean(f1s))
        print(f"epoch {ep} 完成，用时 {time.time()-t0:.0f}s | 验证 IoU {miou:.4f} F1 {mf1:.4f} "
              f"P {np.mean(precs):.4f} R {np.mean(recs):.4f}", flush=True)

        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": ep, "val_iou": miou,
                "best_iou": best_iou, "optim": opt.state_dict(), "sched": sched.state_dict()}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if miou > best_iou:
            best_iou = miou
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print(f"  已保存最佳权重（IoU {miou:.4f}）")

    print(f"训练结束，最佳验证 IoU {best_iou:.4f}；权重在 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
