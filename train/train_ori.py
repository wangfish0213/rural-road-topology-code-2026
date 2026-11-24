# -*- coding: utf-8 -*-
"""训练带方向场分支的模型（创新点 1 的实现入口）。

与 train_unet.py 的区别只有两处：
  1. 网络换成 UNetOri（掩膜头 + 方向场头）；
  2. 损失多了方向场项 λ·(1 - cos 2Δθ)，只在道路像素上算。

用法：
    python train_ori.py --epochs 3 --limit-train 800 --batch 4 --device cuda --ori-weight 0.5
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from orientation import UNetOri, ori_loss, orientation_from_mask
from unet import BCEDiceLoss, RoadTile, pixel_scores


class RoadTileOri(Dataset):
    """在 RoadTile 基础上附带方向场监督；增强后再算方向，保证与影像一致。"""

    def __init__(self, base: RoadTile):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, msk = self.base[idx]
        m = msk[0].numpy() > 0.5
        if m.any():
            c2, s2, w = orientation_from_mask(m, rad=6)
        else:
            z = np.zeros(m.shape, np.float32)
            c2 = s2 = w = z
        return (img, msk,
                torch.from_numpy(c2)[None], torch.from_numpy(s2)[None], torch.from_numpy(w)[None])


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
    ap.add_argument("--ori-weight", type=float, default=0.5)
    ap.add_argument("--limit-train", type=int, default=800)
    ap.add_argument("--limit-val", type=int, default=60)
    ap.add_argument("--max-iters", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="../runs/unet_ori")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="从 out 目录的 last.pt 续训（run_queue 统一传此参数）")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    os.makedirs(args.out, exist_ok=True)

    tr = RoadTileOri(RoadTile(args.manifest, args.crop, args.limit_train, train=True))
    va = RoadTileOri(RoadTile(args.val_manifest, args.crop, args.limit_val, train=False))
    print(f"训练 {len(tr)} 张 / 验证 {len(va)} 张 | crop {args.crop} | batch {args.batch} | 设备 {dev}")
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)

    model = UNetOri(base=args.base).to(dev)
    crit = BCEDiceLoss(pos_weight=args.pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(dl) if not args.max_iters else min(args.max_iters, len(dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * steps))

    log = os.path.join(args.out, "train_log.csv")
    # 续训时追加而不是截断，否则轮次统计和曲线会丢
    if args.resume and os.path.exists(log):
        logf = open(log, "a", newline="", encoding="utf-8-sig")
    else:
        logf = open(log, "w", newline="", encoding="utf-8-sig")
        csv.writer(logf).writerow(["epoch", "iter", "loss", "loss_seg", "loss_ori", "sec_per_iter"])
    logf.close()

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
        best = float(ck.get("best_iou", -1.0))
        print(f"续训：从第 {start_ep} 轮开始（最佳 IoU {best:.4f}）", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0, seen = time.time(), 0
        agg = {"loss": 0.0, "seg": 0.0, "ori": 0.0}
        for it, (img, msk, c2, s2, w) in enumerate(dl, 1):
            img, msk = img.to(dev), msk.to(dev)
            c2, s2, w = c2.to(dev), s2.to(dev), w.to(dev)
            opt.zero_grad(set_to_none=True)
            logits, ori = model(img)
            l_seg = crit(logits, msk)
            l_ori = ori_loss(ori, c2, s2, w)
            loss = l_seg + args.ori_weight * l_ori
            loss.backward()
            opt.step()
            sched.step()
            seen += 1
            agg["loss"] += loss.item()
            agg["seg"] += l_seg.item()
            agg["ori"] += l_ori.item()
            if it % 10 == 0 or it == 1:
                print(f"  ep{ep} it{it}/{steps} loss {agg['loss']/seen:.4f} "
                      f"(seg {agg['seg']/seen:.4f}, ori {agg['ori']/seen:.4f}) "
                      f"{(time.time()-t0)/seen:.2f}s/it", flush=True)
            with open(log, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([ep, it, f"{loss.item():.4f}", f"{l_seg.item():.4f}",
                                        f"{l_ori.item():.4f}", f"{(time.time()-t0)/seen:.4f}"])
            if args.max_iters and it >= args.max_iters:
                break

        model.eval()
        ious, oris = [], []
        with torch.no_grad():
            for img, msk, c2, s2, w in DataLoader(va, batch_size=args.batch, shuffle=False):
                img, msk = img.to(dev), msk.to(dev)
                logits, ori = model(img)
                iou, *_ = pixel_scores(logits, msk)
                ious.append(iou)
                oris.append(ori_loss(ori, c2.to(dev), s2.to(dev), w.to(dev)).item())
        print(f"epoch {ep} 完成 | 验证 IoU {np.mean(ious):.4f} | 方向场损失 {np.mean(oris):.4f}", flush=True)
        if np.mean(ious) > best:
            best = float(np.mean(ious))
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep,
                    "best_iou": best, "optim": opt.state_dict(), "sched": sched.state_dict()},
                   os.path.join(args.out, "last.pt"))
        if np.mean(ious) >= best:
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep,
                        "best_iou": best},
                       os.path.join(args.out, "best.pt"))
    print(f"权重已存至 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
