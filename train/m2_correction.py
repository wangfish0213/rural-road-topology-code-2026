# -*- coding: utf-8 -*-
"""创新点 2 / M2：图结构纠错（语义补连判别 + 尺寸感知孤立分量剪枝）。

论文表述口径（2026-11-09 王老师拍板）：
  - LINK 分支是唯一的**学习式语义判别**贡献；
  - DELETE 分支虽训练了 CNN，但与「删掉所有 <=64 px 分量」的几何规则在消融中
    全部指标 Δ=0，方法章节一律表述为**非学习的几何后处理规则「尺寸感知孤立分量剪枝」**，
    不得写成「语义删除判别」，也不得宣称其有学习增量；
  - 消融与工作点数字见 runs/m2_report.md（tools/summarize_m2.py 生成）。

与旧 correction.py 的区别（论文方法章节对应）：
  1. 由「只有补连」扩为「补连 / 尺寸剪枝」两条措施；任务 2 的连通分量归因表明 M1 的
     主要残余误差是大量小型孤立碎片，剪枝正是针对它。
  2. 特征由 8x8 灰度 MLP 升级为 32x32 多通道 CNN 块 + 几何标量。
  3. 训练数据无泄漏：纠错器在 M1 对 WHU-RRoad *train* 划分的预测图上训练，
     用 train 的 GT 打标签；最终在 *test* 的 M1 预测图上应用并评测（同 2608 张口径）。
  4. 候选定义只用推理时可得的信息（影像 + 预测掩膜）；GT 只在 build 阶段用于打标签，
     apply 阶段完全不读 GT。
  5. 「跨连通分量」约束：同一分量内的两个近端点不算补连候选，避免把已连通的
     短路段误标为正样本（旧版标签污染的来源）。

工程说明：候选骨架在半分辨率（2x2 max-pool）上生成，纯为加速（实测约 10x），
端点坐标乘 DS 映射回全分辨率；最终评测一律走 eval/metrics.py 的全分辨率原口径。

子命令：
  build : 从 train 预测图 + train GT 构造两类候选样本（npz + 统计 json）
  train : 训练两个 PatchCNN，并在 val 上按候选级 F1 选阈值
  apply : 在指定预测目录上应用纠错，写出纠错后掩膜 PNG + 统计 json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_HERE, "..", "eval")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import road_graph as G                                    # noqa: E402
from metrics import skeletonize, to_binary                # noqa: E402

# ------------------------------------------------------------------ 常量
DS = 2                      # 候选骨架的降采样倍率
MAX_DIST = 40.0             # 全分辨率像素：端点对最大距离
MIN_COMP = 8                # 半分辨率下端点所属分量的最小像素数
MAX_DEL_SIZE = 64           # 全分辨率像素：只删小型孤立碎片；较大的零重叠分量
                            # 可能是 WHU-RRoad 稀疏标注漏标的真路，保守起见不纳入
LINK_CROP = 64              # 补连块原始边长（覆盖 40px 连线），缩到 32
PATCH = 32
GT_TOL = 3                  # 与 eval_dir.py 的 --tol 一致
LINK_LABEL_THR = 0.8        # 真值连线覆盖率 >= 0.8 视为「该连」
LINK_LINE_THICK = 3
S8 = np.ones((3, 3), dtype=int)
SEED = 7


# ------------------------------------------------------------------ 基础工具

def crop(arr: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    """以 (cy,cx) 为中心裁 size×size，越界补零。支持 2D/3D。"""
    h, w = arr.shape[:2]
    r = size // 2
    y0, x0 = cy - r, cx - r
    y1, x1 = cy + r, cx + r
    out = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)
    if sy1 <= sy0 or sx1 <= sx0:
        return out
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def draw_line_np(canvas: np.ndarray, y0, x0, y1, x1, thick: int = 3, value: int = 255):
    """在 2D 画布上画粗线（纯 numpy，无 cv2 依赖）。"""
    n = max(8, int(math.hypot(y1 - y0, x1 - x0)))
    ys, xs = G.sample_line(y0, x0, y1, x1, n=n)
    h, w = canvas.shape
    r = thick // 2
    for y, x in zip(ys, xs):
        if 0 <= y < h and 0 <= x < w:
            canvas[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = value


def fast_cov(mask: np.ndarray, y0, x0, y1, x1) -> float:
    """连线落在 mask 上的比例。mask 由调用方预先膨胀一次，避免逐候选重复膨胀 1024² 图。"""
    yy, xx = G.sample_line(y0, x0, y1, x1)
    h, w = mask.shape
    ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    yy, xx = yy[ok], xx[ok]
    return float(mask[yy, xx].mean()) if len(yy) else 0.0


def angle_consistency(a: G.Node, b: G.Node) -> float:
    if not a.dirs or not b.dirs:
        return 0.0
    vy, vx = b.y - a.y, b.x - a.x
    n = math.hypot(vy, vx) or 1.0
    vy, vx = vy / n, vx / n
    da = max((d[0] * vy + d[1] * vx) for d in a.dirs)
    db = max((-d[0] * vy + -d[1] * vx) for d in b.dirs)
    return float((da + db) / 2.0)


def to_chw(arr: np.ndarray) -> np.ndarray:
    """(H,W,C) uint8 -> (C,32,32) uint8；非 32 时用双线性缩放。"""
    if arr.shape[0] != PATCH or arr.shape[1] != PATCH:
        k = PATCH / arr.shape[0]
        arr = ndi.zoom(arr.astype(np.float32), (k, k, 1), order=1)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


# ------------------------------------------------------------------ 候选提取

def collect_candidates(pred_path: str, image_dir: str, stem: str,
                       gt_dir: str | None = None):
    """返回 {'link': [...], 'delete': [...]}。

    GT 只用于生成 label；gt_dir=None 时 label 字段为 None（apply 阶段）。
    每个 link 项含 patch/scalar/label/(ay,ax,by,bx)；
    每个 delete 项含 patch/scalar/label/(lab_id, slice)。
    """
    pr = to_binary(np.array(Image.open(pred_path)))
    h, w = pr.shape
    rgb = np.array(Image.open(os.path.join(image_dir, stem + "_sat.jpg")).convert("RGB"))
    pr_u8 = (pr.astype(np.uint8) * 255)

    has_gt = gt_dir is not None
    if has_gt:
        gt = to_binary(np.array(Image.open(os.path.join(gt_dir, stem + "_mask.png"))))
        gt_dil = ndi.binary_dilation(gt, iterations=2)
        gt_band = ndi.distance_transform_edt(~gt) <= GT_TOL
    pr_dil = ndi.binary_dilation(pr, iterations=2)

    # 半分辨率骨架 + 图
    hh, ww = h // DS, w // DS
    small = pr[:hh * DS, :ww * DS].reshape(hh, DS, ww, DS).max(axis=(1, 3))
    sk = skeletonize(small)
    nodes, _ = G.skeleton_to_graph(sk)
    lab_s, _ = ndi.label(small, S8)
    cnt_s = np.bincount(lab_s.ravel())
    pairs = G.endpoint_pairs(nodes, max_dist=MAX_DIST / DS)

    # 全分辨率连通分量（删除分支）
    lab_f, _ = ndi.label(pr, S8)
    cnt_f = np.bincount(lab_f.ravel())
    big = (lab_f > 0) & (cnt_f[lab_f] >= 200)
    dist_big = ndi.distance_transform_edt(~big)

    links, deletes = [], []

    for a, b, d in pairs:
        ia, ib = int(lab_s[a.y, a.x]), int(lab_s[b.y, b.x])
        if ia == 0 or ib == 0 or ia == ib:          # 必须跨连通分量
            continue
        if cnt_s[ia] < MIN_COMP or cnt_s[ib] < MIN_COMP:
            continue
        ay, ax = int(a.y * DS), int(a.x * DS)
        by, bx = int(b.y * DS), int(b.x * DS)
        d_full = d * DS
        cov_pred = fast_cov(pr_dil, ay, ax, by, bx)
        my, mx = (ay + by) // 2, (ax + bx) // 2

        line = np.zeros((LINK_CROP, LINK_CROP), np.uint8)
        oy, ox = my - LINK_CROP // 2, mx - LINK_CROP // 2
        draw_line_np(line, ay - oy, ax - ox, by - oy, bx - ox, thick=LINK_LINE_THICK)
        patch = to_chw(np.concatenate([
            crop(rgb, my, mx, LINK_CROP),
            crop(pr_u8, my, mx, LINK_CROP)[..., None],
            line[..., None],
        ], axis=2))
        scalar = np.array([
            d_full / 40.0,
            cov_pred,
            angle_consistency(a, b),
            (a.deg + b.deg) / 6.0,
            math.log1p((cnt_s[ia] + cnt_s[ib]) / 2.0) / 8.0,
        ], np.float32)
        label = int(fast_cov(gt_dil, ay, ax, by, bx) >= LINK_LABEL_THR) if has_gt else None
        links.append({"patch": patch, "scalar": scalar, "label": label,
                      "a": (ay, ax), "b": (by, bx)})

    for i, sl in enumerate(ndi.find_objects(lab_f), 1):
        if sl is None:
            continue
        comp = (lab_f[sl] == i)
        sz = int(comp.sum())
        if sz == 0 or sz > MAX_DEL_SIZE:
            continue
        ys, xs = np.nonzero(comp)
        cy = int(sl[0].start + ys.mean())
        cx = int(sl[1].start + xs.mean())
        # 只在局部块里重建该分量，避免每候选分配一张 1024² 全图
        comp_c = (crop(lab_f, cy, cx, PATCH) == i).astype(np.uint8) * 255
        patch = to_chw(np.concatenate([
            crop(rgb, cy, cx, PATCH),
            crop(pr_u8, cy, cx, PATCH)[..., None],
            comp_c[..., None],
        ], axis=2))
        bh, bw = comp.shape
        scalar = np.array([
            math.log1p(sz) / 8.0,
            min(10.0, bw / max(1, bh)),
            sz / max(1, bh * bw),
            min(1.0, float(dist_big[cy, cx]) / 100.0),
        ], np.float32)
        label = int(not bool((comp & gt_band[sl]).any())) if has_gt else None
        deletes.append({"patch": patch, "scalar": scalar, "label": label,
                        "lab": lab_f, "comp_id": i, "sl": sl})

    return {"link": links, "delete": deletes, "pred_px": int(pr.sum())}


def sample_items(items: list, cap: int, rng: random.Random) -> list:
    """每图限流：保留全部正样本（最多 cap//2），负样本随机补足到 cap。"""
    if len(items) <= cap:
        return items
    pos = [x for x in items if x["label"] == 1]
    neg = [x for x in items if x["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    keep_pos = pos[:max(1, cap // 2)]
    return keep_pos + neg[:cap - len(keep_pos)]


# ------------------------------------------------------------------ 网络

class PatchCNN(nn.Module):
    def __init__(self, cin: int, n_scalar: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(48 + n_scalar, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x, s):
        z = self.conv(x).flatten(1)
        return self.fc(torch.cat([z, s], dim=1))


def load_net(ck_path: str, cin: int, n_scalar: int, device: torch.device):
    net = PatchCNN(cin, n_scalar).to(device)
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


@torch.no_grad()
def score_items(net, items: list, device: torch.device, batch: int = 512) -> np.ndarray:
    """返回每个候选属于正类（该连 / 该删）的概率。"""
    if not items:
        return np.zeros(0, np.float32)
    X = np.stack([x["patch"] for x in items])
    S = np.stack([x["scalar"] for x in items])
    out = []
    for i in range(0, len(items), batch):
        xp = torch.from_numpy(X[i:i + batch]).float().to(device) / 255.0
        sp = torch.from_numpy(S[i:i + batch]).float().to(device)
        p = torch.softmax(net(xp, sp), dim=1)[:, 1].cpu().numpy()
        out.append(p)
    return np.concatenate(out).astype(np.float32)


def choose_threshold(y: np.ndarray, p: np.ndarray, target_precision: float) -> dict:
    """在 val 上选阈值：满足目标精度（防乱连 / 防误删）的前提下最大化召回。

    为什么不用 F1 最大：删除任务约 8 成候选本就该删，「全部删除」即可拿到很高 F1，
    会让语义判别器退化成固定启发式。精度约束迫使模型必须真正区分候选。
    """
    grid = []
    for t in np.arange(0.10, 0.995, 0.025):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        grid.append({"thr": round(float(t), 3), "precision": float(prec),
                     "recall": float(rec), "f1": float(f1),
                     "tp": tp, "fp": fp, "fn": fn,
                     "applied_rate": float(pred.mean())})
    ok = [g for g in grid if g["precision"] >= target_precision and g["tp"] > 0]
    if ok:
        best = max(ok, key=lambda g: (g["recall"], g["precision"], -g["thr"]))
        best["mode"] = "precision_constrained"
    else:
        # 模型区分能力不足：退化为精度最高的阈值，并在日志里如实标记
        best = max(grid, key=lambda g: (g["precision"], g["recall"]))
        best["mode"] = "fallback_max_precision"
    best["target_precision"] = target_precision
    return best


# ------------------------------------------------------------------ build

def cmd_build(args) -> int:
    rng = random.Random(SEED)
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.pred, "*.png")))[:args.limit]
    rng.shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac))
    val_files, train_files = files[:n_val], files[n_val:]
    print(f"候选样本：train {len(train_files)} 张 / val {len(val_files)} 张", flush=True)

    buckets = {("train" if k == 0 else "val"): {"link": [], "delete": []}
               for k in (0, 1)}
    tile_stat = []
    t0 = time.time()
    for split, group in (("train", train_files), ("val", val_files)):
        for j, p in enumerate(group, 1):
            stem = os.path.splitext(os.path.basename(p))[0]
            c = collect_candidates(p, args.image_dir, stem, args.gt)
            # 训练集限流控制体量；验证集保留全部候选，保证阈值选择反映真实分布
            lk = sample_items(c["link"], args.cap_link, rng) if split == "train" else c["link"]
            dl = sample_items(c["delete"], args.cap_del, rng) if split == "train" else c["delete"]
            buckets[split]["link"].extend(lk)
            buckets[split]["delete"].extend(dl)
            tile_stat.append({"split": split, "id": stem,
                              "link_cand": len(c["link"]), "link_kept": len(lk),
                              "del_cand": len(c["delete"]), "del_kept": len(dl),
                              "pred_px": c["pred_px"]})
            if j % 50 == 0 or j == len(group):
                print(f"  {split} {j}/{len(group)}  耗时 {time.time()-t0:.0f}s", flush=True)

    stats = {"seed": SEED, "tiles": len(tile_stat), "train_tiles": len(train_files),
             "val_tiles": len(val_files), "settings": {
                 "DS": DS, "MAX_DIST": MAX_DIST, "MIN_COMP": MIN_COMP,
                 "MAX_DEL_SIZE": MAX_DEL_SIZE, "GT_TOL": GT_TOL,
                 "LINK_LABEL_THR": LINK_LABEL_THR, "cap_link": args.cap_link,
                 "cap_del": args.cap_del}}
    for split in ("train", "val"):
        for task in ("link", "delete"):
            items = buckets[split][task]
            if not items:
                raise RuntimeError(f"{split}/{task} 无样本")
            np.savez(os.path.join(args.out, f"{split}_{task}.npz"),
                     X=np.stack([x["patch"] for x in items]),
                     S=np.stack([x["scalar"] for x in items]),
                     Y=np.array([x["label"] for x in items], np.int64))
            y = np.array([x["label"] for x in items])
            stats[f"{split}_{task}"] = {"n": len(items), "pos": int(y.sum()),
                                        "pos_rate": float(y.mean())}
    with open(os.path.join(args.out, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "tiles.csv"), "w", newline="", encoding="utf-8-sig") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(tile_stat[0].keys()))
        w.writeheader()
        w.writerows(tile_stat)
    print(f"完成，耗时 {time.time()-t0:.0f}s；写 {args.out}", flush=True)
    print(json.dumps({k: v for k, v in stats.items() if isinstance(v, dict) and "n" in v},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


# ------------------------------------------------------------------ train

def cmd_train(args) -> int:
    torch.manual_seed(SEED)
    dev = torch.device("cpu" if args.device == "cpu" else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out, exist_ok=True)
    log = {"device": str(dev), "epochs": args.epochs, "seed": SEED, "tasks": {}}

    # 精度红线：补连不许乱连，删除不许误删（69% 以上候选本就该删，必须靠语义区分）
    targets = {"link": 0.60, "delete": 0.90}
    for task, cin, n_scalar in (("link", 5, 5), ("delete", 5, 4)):
        tr = np.load(os.path.join(args.data, f"train_{task}.npz"))
        va = np.load(os.path.join(args.data, f"val_{task}.npz"))
        X, S, Y = tr["X"], tr["S"], tr["Y"]
        net = PatchCNN(cin, n_scalar).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        pos_rate = float(Y.mean())
        # 少数类加权：正类权重 = 负类频率/正类频率，避免模型退化成全预测负类
        weight = torch.tensor([1.0, (1.0 - pos_rate) / max(pos_rate, 1e-3)],
                              dtype=torch.float32).to(dev)
        crit = nn.CrossEntropyLoss(weight=weight)
        rng = np.random.default_rng(SEED)
        losses = []
        for ep in range(1, args.epochs + 1):
            net.train()
            order = rng.permutation(len(Y))
            tot = 0.0
            for i in range(0, len(order), args.batch):
                idx = order[i:i + args.batch]
                xp = torch.from_numpy(X[idx]).float().to(dev) / 255.0
                sp = torch.from_numpy(S[idx]).float().to(dev)
                yp = torch.from_numpy(Y[idx]).long().to(dev)
                opt.zero_grad()
                loss = crit(net(xp, sp), yp)
                loss.backward()
                opt.step()
                tot += float(loss.item()) * len(idx)
            losses.append(tot / len(Y))
            if ep % 5 == 0 or ep == 1:
                print(f"  {task} ep{ep}/{args.epochs} loss {losses[-1]:.4f}", flush=True)

        net.eval()
        p = score_items(net, [{"patch": x, "scalar": s} for x, s in zip(va["X"], va["S"])], dev)
        thr = choose_threshold(va["Y"], p, targets[task])
        ck = os.path.join(args.out, f"{task}_net.pt")
        torch.save({"model": net.state_dict(), "cin": cin, "n_scalar": n_scalar,
                    "epochs": args.epochs, "threshold": thr}, ck)
        log["tasks"][task] = {"train_n": int(len(Y)), "train_pos_rate": pos_rate,
                              "loss_first": losses[0], "loss_last": losses[-1],
                              "target_precision": targets[task], "val": thr, "ckpt": ck}
        print(f"  {task}: thr={thr['thr']} [{thr['mode']}] "
              f"P={thr['precision']:.3f} R={thr['recall']:.3f} "
              f"应用率={thr['applied_rate']:.3f}", flush=True)

    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"模型与训练记录写入 {args.out}", flush=True)
    return 0


# ------------------------------------------------------------------ apply

def cmd_apply(args) -> int:
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    link_net = load_net(os.path.join(args.model, "link_net.pt"), 5, 5, dev)
    del_net = load_net(os.path.join(args.model, "delete_net.pt"), 5, 4, dev)
    link_ck = torch.load(os.path.join(args.model, "link_net.pt"), map_location="cpu", weights_only=False)
    del_ck = torch.load(os.path.join(args.model, "delete_net.pt"), map_location="cpu", weights_only=False)
    thr_link = float(args.thr_link) if args.thr_link else float(link_ck["threshold"]["thr"])
    thr_del = float(args.thr_del) if args.thr_del else float(del_ck["threshold"]["thr"])
    print(f"阈值：link={thr_link}  delete={thr_del}  设备={dev}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.out_dir_stats, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.pred, "*.png")))[:args.limit]
    rows, t0 = [], time.time()
    skipped = 0
    for j, p in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(args.out, stem + ".png")
        if (not args.overwrite) and os.path.exists(dst):
            skipped += 1
            continue
        c = collect_candidates(p, args.image_dir, stem, None)
        pr = to_binary(np.array(Image.open(p)))
        new = pr.copy()

        # 先删除：高置信「该删」的小分量
        pd = score_items(del_net, c["delete"], dev)
        n_del = 0
        for item, prob in zip(c["delete"], pd):
            if prob >= thr_del:
                sl, cid = item["sl"], item["comp_id"]
                sub = new[sl]
                sub[item["lab"][sl] == cid] = False
                n_del += 1

        # 再补连：高置信「该连」的跨分量端点对
        pl = score_items(link_net, c["link"], dev)
        n_link = 0
        for item, prob in zip(c["link"], pl):
            if prob >= thr_link:
                ay, ax = item["a"]
                by, bx = item["b"]
                ys, xs = G.sample_line(ay, ax, by, bx,
                                       n=max(8, int(math.hypot(by - ay, bx - ax))))
                r = LINK_LINE_THICK // 2
                h, w = new.shape
                for y, x in zip(ys, xs):
                    if 0 <= y < h and 0 <= x < w:
                        new[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
                n_link += 1

        Image.fromarray((new.astype(np.uint8) * 255)).save(dst)
        rows.append({"id": stem, "link_cand": len(c["link"]), "link_applied": n_link,
                     "del_cand": len(c["delete"]), "del_applied": n_del,
                     "px_before": int(pr.sum()), "px_after": int(new.sum())})
        if j % 100 == 0 or j == len(files):
            print(f"  {j}/{len(files)}  耗时 {time.time()-t0:.0f}s", flush=True)

    import csv
    keys = ["id", "link_cand", "link_applied", "del_cand", "del_applied",
            "px_before", "px_after"]
    if rows:
        with open(os.path.join(args.out_dir_stats, "apply_per_image.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    agg = ({k: float(np.mean([r[k] for r in rows]))
            for k in ("link_cand", "link_applied", "del_cand", "del_applied",
                      "px_before", "px_after")} if rows else {})
    png_total = len(glob.glob(os.path.join(args.out, "*.png")))
    stats = {"tiles_processed_this_run": len(rows), "tiles_skipped_existing": skipped,
             "png_total_in_out_dir": png_total,
             "thr_link": thr_link, "thr_del": thr_del,
             "pred_dir": args.pred, "out_dir": args.out,
             "mean_per_tile": agg,
             "total": {k: int(sum(r[k] for r in rows))
                       for k in ("link_cand", "link_applied", "del_cand", "del_applied")},
             "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir_stats, "apply_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"完成：本次 {len(rows)} 张，跳过 {skipped} 张，目录共 {png_total} 张，"
          f"耗时 {stats['elapsed_sec']}s", flush=True)
    return 0


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M2 图节点语义纠错")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--pred", required=True)
    b.add_argument("--gt", required=True)
    b.add_argument("--image-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit", type=int, default=600)
    b.add_argument("--val-frac", type=float, default=0.2)
    b.add_argument("--cap-link", type=int, default=50)
    b.add_argument("--cap-del", type=int, default=40)
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("apply")
    a.add_argument("--model", required=True)
    a.add_argument("--pred", required=True)
    a.add_argument("--image-dir", required=True)
    a.add_argument("--out", required=True, help="纠错后掩膜目录")
    a.add_argument("--out-dir-stats", required=True, help="统计文件目录")
    a.add_argument("--limit", type=int, default=3000)
    a.add_argument("--thr-link", type=float, default=None)
    a.add_argument("--thr-del", type=float, default=None)
    a.add_argument("--overwrite", action="store_true",
                   help="强制覆盖已有掩膜；默认跳过，便于本机静默中断后断点续跑")
    a.add_argument("--device", default="auto")
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
