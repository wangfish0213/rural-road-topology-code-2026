# -*- coding: utf-8 -*-
"""创新点 3 / E4：线—面跨任务一致性软约束（IRSAMap V2，同景建筑+路面+中心线）。

设计（2026-11-09，v2 修订见文末；v3 为 2026-11-09 晚定稿）：
  - 共享 U-Net 编码器 + 三个 1x1 头：building / roadarea / roadline。
  - 三条**可开关**的训练期软约束，推理时不需要真值。v3 默认 anchor=gt
    （以另一任务的**真值**作固定锚，消除平凡解）：
      L_excl   = mean(p_b * GT_roadarea) + mean(p_r * GT_building)
      L_dist   = mean(boundary(p_b) * dilate(GT_line, r))
      L_linein = mean(p_l * (1 - GT_roadarea))
    v2 的 anchor=pred（p_b*p_r / boundary(p_b)*dilate(p_l) / p_l*(1-p_r)）
    存在**平凡解**：把路面头或线头整体关掉即可让约束损失趋零，实测导致
    roadarea_iou→0、预测线带覆盖整图，故 v3 改为真值锚定并保留开关供消融复现。
    前两条来自研究方案，第三条来自 tools/e4_gt_consistency.py 的实测：
    真值中 99.3%~99.99% 的中心线像素落在路面内。
  - 基线 lam=0 与加约束变体用**同一随机种子、同一数据顺序、同一轮数、同一初始权重**，
    统一取最后一个 epoch，不按验证集挑优，避免选择偏差。
  - 评测：建筑/路面像素 IoU+F1；中心线走 eval/metrics.evaluate（线F1/APLS/断裂率等）；
    另报一致性指标在预测侧与**真值参照侧**的违规率，以及预测覆盖率诊断量。

数据口径红线：
  1. IRSAMap 栅格标签是**类别值**（building=31、roadarea=32），必须 >0 二值化，
     不能沿用 eval/metrics.py 的默认 127 阈值（否则整层被判成背景）。
  2. 中心线在 1024 切片里只占约 1.3% 像素，缩到 256/384 后更稀疏；
     **v1（细线直接监督）在 6 epochs/140 张下中心线头完全没学出来**
     （e4_base：建筑 IoU 0.328 但线 IoU/线F1/APLS 全 0），故 v2 改为「线带监督」：
     训练 GT 把中心线膨胀 --line-dilate 次（默认 2）形成可学习的带状目标，
     评测时把预测线带骨架化回细线，再与原始细线 GT 比较（三变体同口径）。
  3. 训练数据一次性预缓存到内存，避免每 step 重复解码 1024 PNG（v1 慢的主因）。

用法：
  python train/e4_consistency.py run --name e4_base --lam-excl 0 --lam-dist 0 --lam-linein 0
  python train/e4_consistency.py run --name e4_cons --lam-excl 0.2 --lam-dist 0.2
  python train/e4_consistency.py eval --name e4_base --test-limit 50

产物（全部脚本写盘）：
  runs/{name}/last.pt, train_log.csv, eval/{per_image.csv,summary.csv,config.json},
  pred/*.png（前 20 张预测，便于独立回读校验）
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "eval"))
from unet import conv_block, BCEDiceLoss  # noqa: E402
from metrics import evaluate, EvalConfig, pixel_metrics, skeletonize  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

MANIFESTS = ROOT / "manifests"
RUNS = ROOT / "runs"
SEED = 7
KEYS = ("building", "roadarea", "roadline")


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mask(path: str, size: int, dilate: int = 0) -> np.ndarray:
    """IRSAMap 类别值标签 -> 0/1；最近邻缩放保持细线不断；可选膨胀成线带（仅训练 GT）。"""
    im = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    a = np.asarray(im) > 0
    if dilate:
        a = ndi.binary_dilation(a, iterations=dilate)
    return a.astype(np.float32)


def load_image(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(im).transpose(2, 0, 1)).astype(np.float32) / 255.0


class E4Tile(Dataset):
    """训练集返回 4 通道标签：建筑 / 路面 / 线带(监督) / 真值线的 r 邻域(约束锚)。

    第 4 通道只在训练时使用；评测侧直接读原始细线，不走这里。
    """

    def __init__(self, split: str, size: int, limit: int, train: bool,
                 line_dilate: int = 0, cache: bool = False, radius: int = 3):
        path = MANIFESTS / f"irsamap_e4_{split}.csv"
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        if limit:
            rows = rows[:limit]
        self.rows, self.size, self.train = rows, size, train
        self.line_dilate, self.radius = line_dilate, radius
        self._items = []
        if cache:
            t0 = time.time()
            for r in rows:
                x = load_image(r["image"], size)
                ys = np.stack([load_mask(r[k], size,
                                         line_dilate if k == "roadline" else 0)
                               for k in KEYS], axis=0)
                if train:
                    thin = load_mask(r["roadline"], size) > 0
                    near = ndi.binary_dilation(thin, iterations=radius).astype(np.float32)
                    ys = np.concatenate([ys, near[None]], axis=0)
                self._items.append((x, ys))
            print(f"  预缓存 {len(rows)} 张到内存（{size}x{size}，{ys.shape[0]}ch），"
                  f"耗时 {time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if self._items:
            x, ys = self._items[idx]
        else:
            r = self.rows[idx]
            x = load_image(r["image"], self.size)
            ys = np.stack([load_mask(r[k], self.size,
                                     self.line_dilate if k == "roadline" else 0)
                           for k in KEYS], axis=0)
            if self.train:
                thin = load_mask(r["roadline"], self.size) > 0
                near = ndi.binary_dilation(thin, iterations=self.radius).astype(np.float32)
                ys = np.concatenate([ys, near[None]], axis=0)
        if self.train:
            if random.random() < 0.5:
                x, ys = x[:, :, ::-1], ys[:, :, ::-1]
            if random.random() < 0.5:
                x, ys = x[:, ::-1, :], ys[:, ::-1, :]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(ys)))


class UNet3Head(nn.Module):
    """共享编码器/解码器 + 三个独立 1x1 头。base=16 以便 2 GB 显存可跑。"""

    def __init__(self, base: int = 16):
        super().__init__()
        b = base
        self.e1 = conv_block(3, b)
        self.e2 = conv_block(b, b * 2)
        self.e3 = conv_block(b * 2, b * 4)
        self.e4 = conv_block(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.c3 = conv_block(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.c2 = conv_block(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.c1 = conv_block(b * 2, b)
        self.h_building = nn.Conv2d(b, 1, 1)
        self.h_roadarea = nn.Conv2d(b, 1, 1)
        self.h_roadline = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        s4 = self.e4(F.max_pool2d(s3, 2))
        x = self.c3(torch.cat([self.up3(s4), s3], 1))
        x = self.c2(torch.cat([self.up2(x), s2], 1))
        x = self.c1(torch.cat([self.up1(x), s1], 1))
        return torch.cat([self.h_building(x), self.h_roadarea(x),
                          self.h_roadline(x)], dim=1)


def consistency_losses(logits, gt, lam, radius: int = 3, anchor: str = "gt"):
    """返回 (总一致性损失, 逐项 tensor dict)。

    gt: [B,4,H,W]，顺序 building / roadarea / line_band / line_near。
    anchor=gt  : 以另一任务的真值为固定锚（v3 默认，无平凡解）。
    anchor=pred: v2 原式，纯预测侧，仅用于复现/消融对照。
    """
    p = torch.sigmoid(logits)
    pb, pr, pl = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    erosion = 1 - F.max_pool2d(1 - pb, 3, stride=1, padding=1)
    bd = torch.clamp(pb - erosion, 0, 1)
    items = {}
    if anchor == "gt":
        gb, gr, gnear = gt[:, 0:1], gt[:, 1:2], gt[:, 3:4]
        if lam["excl"]:
            items["excl"] = (pb * gr).mean() + (pr * gb).mean()
        if lam["dist"]:
            items["dist"] = (bd * gnear).mean()
        if lam["linein"]:
            items["linein"] = (pl * (1 - gr)).mean()
    else:
        if lam["excl"]:
            items["excl"] = (pb * pr).mean()
        if lam["dist"]:
            dil = F.max_pool2d(pl, 2 * radius + 1, stride=1, padding=radius)
            items["dist"] = (bd * dil).mean()
        if lam["linein"]:
            items["linein"] = (pl * (1 - pr)).mean()
    total = sum(lam[k] * v for k, v in items.items()) if items else logits.new_zeros(())
    return total, {k: v.detach() for k, v in items.items()}


def train(args):
    seed_all(args.seed)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    lam = {"excl": args.lam_excl, "dist": args.lam_dist, "linein": args.lam_linein}

    ds = E4Tile("train", args.size, args.train_limit, True,
                line_dilate=args.line_dilate, cache=not args.no_cache,
                radius=args.radius)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)
    model = UNet3Head(args.base).to(dev)
    if args.init:
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"  从公共初始化加载权重：{args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = args.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # v3：三个头统一温和前景权重，靠 Dice 处理类别不平衡；
    # v2 的 3/5/8 让线带覆盖整图、路面头几乎不预测，已实测为退化源。
    pw = args.posw
    pw_line = args.posw_line if args.posw_line is not None else pw
    crit = [BCEDiceLoss(pos_weight=pw), BCEDiceLoss(pos_weight=pw),
            BCEDiceLoss(pos_weight=pw_line)]

    log_path = out / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(["epoch", "loss", "loss_excl", "loss_dist",
                                "loss_linein", "seconds"])
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        # 只在 epoch 末做一次 GPU->CPU 同步；v2 每 step 都 float(...cpu()) 是主要瓶颈
        agg = {k: torch.zeros((), device=dev)
               for k in ("loss", "excl", "dist", "linein")}
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16,
                                enabled=use_amp):
                logits = model(x)
                sup = sum(crit[i](logits[:, i:i + 1], y[:, i:i + 1]) for i in range(3))
                con, items = consistency_losses(logits, y, lam, args.radius, args.anchor)
                loss = sup + con
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            agg["loss"] += loss.detach()
            for k, v in items.items():
                agg[k] += v
        n = max(1, len(dl))
        row = [ep, round(float(agg["loss"]) / n, 5), round(float(agg["excl"]) / n, 6),
               round(float(agg["dist"]) / n, 6), round(float(agg["linein"]) / n, 6),
               round(time.time() - t0, 1)]
        with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(row)
        print(f"  epoch {ep}/{args.epochs} loss={row[1]:.4f} "
              f"excl={row[2]:.5f} dist={row[3]:.5f} linein={row[4]:.5f} ({row[5]}s)")
        torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(args)},
                   out / "last.pt")
    print(f"训练完成：{out/'last.pt'}，耗时 {time.time()-t0:.1f}s，设备 {dev}")
    return 0


def np_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))


def e4_metrics(pb: np.ndarray, pr: np.ndarray, pl_band: np.ndarray,
               gb: np.ndarray, gr: np.ndarray, gl: np.ndarray,
               radius: int, cfg: EvalConfig) -> dict:
    """pl_band 为模型输出的线带；先骨架化回细线再评测，保证与细线 GT 同口径。"""
    out = {}
    for name, p, g in (("building", pb, gb), ("roadarea", pr, gr)):
        m = pixel_metrics(p, g)
        out[f"{name}_iou"] = m["iou"]
        out[f"{name}_f1"] = m["f1"]
    pl = skeletonize(pl_band.astype(np.uint8) * 255) > 0
    ln = evaluate(pl.astype(np.uint8) * 255, gl.astype(np.uint8) * 255, cfg)
    for k in ("iou", "f1", "line_f1", "apls", "connectivity", "breakage_rate",
              "false_connection_rate", "cc_ratio", "length_ratio"):
        out[f"line_{k}"] = ln.get(k, float("nan"))
    # 预测侧一致性违规率（pred-vs-pred；分母依赖预测像素数，欠拟合时可能失真）
    out["excl_violation"] = float((pb & pr).sum() / max(1, pr.sum()))
    bd = np_boundary(pb)
    near = ndi.distance_transform_edt(~pl) <= radius
    out["boundary_line_violation"] = float((bd & near).sum() / max(1, bd.sum()))
    out["line_in_roadarea"] = float((pl & pr).sum() / max(1, pl.sum()))
    # 真值参照指标（pred-vs-GT）：分母固定，不受「模型干脆不预测某类」影响，更稳健
    near_gt = ndi.distance_transform_edt(~gl) <= radius
    out["gt_excl_violation"] = float((pb & gr).sum() / max(1, gr.sum()))
    out["gt_boundary_line_violation"] = float((bd & near_gt).sum() / max(1, bd.sum()))
    out["gt_line_in_roadarea"] = float((pl & gr).sum() / max(1, pl.sum()))
    # 退化诊断量：预测覆盖率过高说明头未收敛，此时拓扑类指标不可信
    out["pred_line_coverage"] = float(pl_band.mean())
    out["pred_roadarea_coverage"] = float(pr.mean())
    out["pred_building_coverage"] = float(pb.mean())
    return out


def eval_only(args):
    seed_all(args.seed)
    out = RUNS / args.name
    ck = out / "last.pt"
    if not ck.is_file():
        print(f"缺权重 {ck}")
        return 1
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(ck, map_location=dev, weights_only=False)
    model = UNet3Head(args.base).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 评测侧 GT 一律用**未膨胀**的原始细线；训练集才用线带监督
    ds = E4Tile("test", args.size, args.test_limit, False, line_dilate=0, cache=False)
    cfg = EvalConfig(gsd=1.0, tol_px=2.0, apls_samples=40, seed=0)
    pred_dir = out / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, r in enumerate(ds.rows):
            x = torch.from_numpy(load_image(r["image"], args.size))[None].to(dev)
            p = torch.sigmoid(model(x))[0].cpu().numpy()
            pb, pr, pl_band = p[0] > 0.5, p[1] > 0.5, p[2] > 0.5
            gb = load_mask(r["building"], args.size) > 0
            gr = load_mask(r["roadarea"], args.size) > 0
            gl = load_mask(r["roadline"], args.size) > 0
            m = {"id": r["id"], **e4_metrics(pb, pr, pl_band, gb, gr, gl,
                                              args.radius, cfg)}
            rows.append(m)
            if i < 20:  # 只留 20 张预测图，供独立回读与可视化
                Image.fromarray(np.stack([pb, pr, pl_band], -1).astype(np.uint8) * 255).save(
                    pred_dir / f"{r['id']}.png")
            if (i + 1) % 20 == 0:
                print(f"  评测 {i+1}/{len(ds.rows)}，耗时 {time.time()-t0:.0f}s")
    keys = [k for k in rows[0] if k != "id"]
    with open(out / "eval" / "per_image.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + keys)
        w.writeheader()
        w.writerows(rows)
    with open(out / "eval" / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median"])
        for k in keys:
            v = np.asarray([r[k] for r in rows], dtype=float)
            v = v[np.isfinite(v)]
            w.writerow([k, f"{v.mean():.6f}" if v.size else "nan",
                        f"{np.median(v):.6f}" if v.size else "nan"])
    a = ckpt["args"]
    cfg_out = {"name": args.name, "size": args.size, "radius": args.radius,
               "test_limit": args.test_limit, "seed": args.seed,
               "line_dilate": a.get("line_dilate"),
               "anchor": a.get("anchor", "pred"),   # 旧 v2 权重按 pred 口径
               "posw": a.get("posw", "legacy(3/5/8)"),
               "posw_line": a.get("posw_line"),
               "line_pos_weight": a.get("line_pos_weight"),
               "train_limit": a.get("train_limit"), "epochs": a.get("epochs"),
               "lr": a.get("lr"), "batch": a.get("batch"), "base": a.get("base"),
               "amp": a.get("amp", False), "init": a.get("init"),
               "lam": {"excl": a.get("lam_excl"),
                       "dist": a.get("lam_dist"),
                       "linein": a.get("lam_linein")},
               "epoch": ckpt["epoch"], "n_images": len(rows),
               "elapsed_s": round(time.time() - t0, 1)}
    with open(out / "eval" / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, ensure_ascii=False, indent=2)
    print(f"评测完成：{len(rows)} 张 → {out/'eval'/'summary.csv'}，耗时 {time.time()-t0:.1f}s")
    for k in ("building_iou", "line_iou", "line_line_f1", "line_apls",
              "excl_violation", "boundary_line_violation", "line_in_roadarea"):
        if k in rows[0]:
            print(f"  {k}: {np.mean([r[k] for r in rows]):.4f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "train", "eval"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-limit", type=int, default=200)
    ap.add_argument("--test-limit", type=int, default=50)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--line-dilate", type=int, default=2,
                    help="训练 GT 中心线膨胀次数（0=细线直监督，v1 已证实学不出来）")
    ap.add_argument("--line-pos-weight", type=float, default=8.0,
                    help="v2 遗留参数，v3 不再使用（统一由 --posw 控制）")
    ap.add_argument("--posw", type=float, default=1.0,
                    help="三个头统一的前景 BCE 权重；v2 的 3/5/8 已证实导致退化")
    ap.add_argument("--posw-line", type=float, default=None,
                    help="线头单独的前景权重；None 表示跟随 --posw")
    ap.add_argument("--anchor", choices=["gt", "pred"], default="gt",
                    help="一致性约束锚：gt=v3 真值锚定（默认），pred=v2 原式（平凡解对照）")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--lam-excl", type=float, default=0.0)
    ap.add_argument("--lam-dist", type=float, default=0.0)
    ap.add_argument("--lam-linein", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="CUDA 混合精度；MX450 实测产生 NaN 且不提速，勿用")
    ap.add_argument("--init", default=None,
                    help="公共初始化权重路径：四变体从同一 last.pt 各微调相同轮数，严格对照")
    args = ap.parse_args()
    if args.cmd in ("run", "train"):
        rc = train(args)
        if rc:
            return rc
    if args.cmd in ("run", "eval"):
        return eval_only(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
