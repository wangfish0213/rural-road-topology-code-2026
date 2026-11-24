# -*- coding: utf-8 -*-
"""创新点 2：图节点的语义纠错（补连 / 删除 / 保留）。

与几何启发式的区别：不是"两个端点近就连"，而是看影像语义——连线经过的地方
到底像不像路。隔着一道田埂的两条路端点也很近，几何方法必然连错；
语义判别器可以从影像块与连线覆盖情况把它区分开。

流程：
  1. 预测掩膜 → 骨架 → 图 → 枚举候选端点对（road_graph.py）
  2. 每个候选对提特征（距离、方向一致性、预测覆盖、影像块）
  3. 用真值给候选对打标签：连线上真值覆盖率 ≥ 0.8 → 该连（1），否则不该连（0）
  4. 训练小分类器；推理时对高置信候选画线补连

用法：
    python correction.py --pred ../runs/unet_gpu_smoke/pred --gt ../datasets/WHU-RRoad/test \
        --limit 100 --epochs 300
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import road_graph as G                      # noqa: E402
from metrics import (EvalConfig, connectivity_metrics, skeletonize,  # noqa: E402
                     to_binary)


# ---------------------------------------------------------------- 特征

PATCH = 16      # 取中点周围 16×16，降采样到 8×8


def _angle_consistency(a: G.Node, b: G.Node) -> float:
    """两端出边方向与连线方向的一致程度（1=完全顺接）。"""
    if not a.dirs or not b.dirs:
        return 0.0
    vy, vx = b.y - a.y, b.x - a.x
    n = np.hypot(vy, vx) or 1.0
    vy, vx = vy / n, vx / n
    da = max((a.dirs[k][0] * vy + a.dirs[k][1] * vx) for k in range(len(a.dirs)))
    db = max((-b.dirs[k][0] * vy + -b.dirs[k][1] * vx) for k in range(len(b.dirs)))
    return float((da + db) / 2.0)


def _patch_feature(gray: np.ndarray, y: int, x: int) -> np.ndarray:
    h, w = gray.shape
    r = PATCH // 2
    y0, y1 = max(0, y - r), min(h, y + r)
    x0, x1 = max(0, x - r), min(w, x + r)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(64, np.float32)
    img = Image.fromarray((patch * 255).astype(np.uint8)).resize((8, 8), Image.BILINEAR)
    return (np.asarray(img, np.float32) / 255.0).ravel()


def pair_features(gray, pred, a: G.Node, b: G.Node, d: float):
    """候选对的特征向量：4 个标量 + 8×8 影像块。全部只用推理时可获得的信息。"""
    cov = G.line_coverage(pred, a.y, a.x, b.y, b.x, pad=2)
    ac = _angle_consistency(a, b)
    mid_y, mid_x = (a.y + b.y) // 2, (a.x + b.x) // 2
    vec = np.concatenate([[d / 40.0, cov, ac, float(a.deg + b.deg) / 6.0],
                          _patch_feature(gray, mid_y, mid_x)]).astype(np.float32)
    return vec


def label_pair(gt, a: G.Node, b: G.Node) -> int:
    """真值连线覆盖率 ≥0.8 认为该连（含容差膨胀 2 px）。"""
    return int(G.line_coverage(gt, a.y, a.x, b.y, b.x, pad=2) >= 0.8)


# ---------------------------------------------------------------- 分类器

class PairNet(nn.Module):
    def __init__(self, dim: int = 68, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- 主流程

def load_tile(pred_path: str, gt_path: str):
    pred = to_binary(np.array(Image.open(pred_path)))
    gt = to_binary(np.array(Image.open(gt_path)))
    img = np.array(Image.open(gt_path.replace("_mask.png", "_sat.jpg")).convert("L"), np.float32) / 255.0
    return pred, gt, img


def build_dataset(pred_dir, gt_dir, ids):
    X, y, stat = [], [], {"pairs": 0, "pos": 0, "tiles": 0}
    for i in ids:
        pp = os.path.join(pred_dir, i + ".png")
        gp = os.path.join(gt_dir, i + "_mask.png")
        if not (os.path.exists(pp) and os.path.exists(gp)):
            continue
        pred, gt, gray = load_tile(pp, gp)
        sk = skeletonize(pred)
        nodes, _ = G.skeleton_to_graph(sk)
        for a, b, d in G.endpoint_pairs(nodes, max_dist=40):
            X.append(pair_features(gray, pred, a, b, d))
            y.append(label_pair(gt, a, b))
        stat["pairs"] += len(G.endpoint_pairs(nodes, max_dist=40))
        stat["pos"] += sum(y[-len(G.endpoint_pairs(nodes, max_dist=40)):]) if G.endpoint_pairs(nodes, 40) else 0
        stat["tiles"] += 1
    return np.stack(X) if X else np.zeros((0, 68), np.float32), np.array(y, np.int64), stat


def draw_line(mask: np.ndarray, a, b, thick: int = 3):
    """在掩膜上把两点连起来（粗 thick 像素），返回新的布尔掩膜。"""
    m = mask.copy()
    ys, xs = G.sample_line(a[0], a[1], b[0], b[1], n=max(8, int(np.hypot(b[0] - a[0], b[1] - a[1]))))
    r = thick // 2
    for y, x in zip(ys, xs):
        m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    return m


def evaluate_tiles(model, pred_dir, gt_dir, ids, thr=0.5):
    cfg = EvalConfig(seed=0, apls_samples=40)
    before, after, applied, cand_total = [], [], 0, 0
    for i in ids:
        pp = os.path.join(pred_dir, i + ".png")
        gp = os.path.join(gt_dir, i + "_mask.png")
        if not (os.path.exists(pp) and os.path.exists(gp)):
            continue
        pred, gt, gray = load_tile(pp, gp)
        if gt.sum() == 0:
            continue
        sk = skeletonize(pred)
        nodes, _ = G.skeleton_to_graph(sk)
        pairs = G.endpoint_pairs(nodes, max_dist=40)
        cand_total += len(pairs)
        m0 = connectivity_metrics(pred, gt, cfg, pred_skel=sk)["connectivity"]
        new = pred.copy()
        for a, b, d in pairs:
            with torch.no_grad():
                p = torch.softmax(model(torch.from_numpy(pair_features(gray, pred, a, b, d))[None]), -1)[0, 1].item()
            if p >= thr:
                new = draw_line(new, (a.y, a.x), (b.y, b.x))
                applied += 1
        m1 = connectivity_metrics(new, gt, cfg)["connectivity"]
        before.append(m0)
        after.append(m1)
    return before, after, applied, cand_total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=20, help="最后的若干张留作测试")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="../runs/correction")
    args = ap.parse_args(argv)

    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.pred) if f.endswith(".png"))[:args.limit]
    test_ids = ids[-args.holdout:]
    train_ids = ids[:-args.holdout]
    os.makedirs(args.out, exist_ok=True)

    print(f"构建训练集（{len(train_ids)} 张）…")
    X, y, stat = build_dataset(args.pred, args.gt, train_ids)
    print(f"候选端点对 {len(y)} 个，其中应补连 {int(y.sum())} 个（{100*y.mean():.1f}%）")
    if len(y) == 0 or y.sum() == 0:
        print("没有可用样本，停止")
        return 1

    torch.manual_seed(0)
    net = PairNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # 类别不平衡：正样本很少，按比例加权
    pos_w = torch.tensor([len(y) / max(1, y.sum()) - 1], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=torch.cat([torch.ones(1), pos_w]))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for ep in range(1, args.epochs + 1):
        net.train()
        opt.zero_grad()
        loss = crit(net(Xt), yt)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == 1:
            with torch.no_grad():
                acc = (net(Xt).argmax(1) == yt).float().mean().item()
            print(f"  ep{ep} loss {loss.item():.4f} acc {acc:.3f}")

    print(f"\n在 {len(test_ids)} 张留出图上评估补连效果…")
    before, after, applied, cand = evaluate_tiles(net, args.pred, args.gt, test_ids)
    b, a = float(np.mean(before)), float(np.mean(after))
    print(f"候选对 {cand} 个，实际补连 {applied} 条")
    print(f"连通率  补连前 {b:.4f} → 补连后 {a:.4f}  （Δ {a-b:+.4f}）")
    torch.save({"model": net.state_dict(), "args": vars(args)}, os.path.join(args.out, "pairnet.pt"))
    with open(os.path.join(args.out, "result.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tiles", "candidates", "applied", "connectivity_before", "connectivity_after"])
        w.writerow([len(before), cand, applied, f"{b:.4f}", f"{a:.4f}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
