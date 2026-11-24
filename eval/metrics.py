# -*- coding: utf-8 -*-
"""道路中心线/掩膜评测核心指标。

依赖：numpy、scipy、Pillow（均已在本机隔离环境内可用，不依赖 scikit-image）。

指标口径（论文 Table 2 用这一套，写进正文时要原样说明）：
  1. 像素级：IoU / F1 / Precision / Recall
  2. 线状：骨架化后 3 px 容差的 precision / recall / F1（衡量"中心线位置对不对"）
  3. 拓扑：APLS（Average Path Length Similarity，采样点对最短路径长度的一致性）
  4. 结构：断裂率（一个真值连通分量被预测切成 >=2 段）、伪连接率
     （一个预测连通分量跨接 >=2 个真值分量）、连通分量比

设计说明：
  - 骨架化自实现 Zhang-Suen 细化，避免额外依赖，且行为可控；
  - APLS 用 scipy.sparse.csgraph.dijkstra 一次性多源计算，避免逐对调用；
  - 断裂/伪连接用 scipy.ndimage.label（8 邻域），按覆盖比例阈值过滤毛刺。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

SQRT2 = math.sqrt(2.0)


@dataclass
class EvalConfig:
    """评测配置。gsd 为地面采样距离（米/像素），用于把像素长度换算成米。"""
    gsd: float = 1.0
    tol_px: float = 3.0
    apls_samples: int = 60
    min_link_px: int = 30        # 判定"实质跨接"的最小像素数（约 9 m）
    seed: int = 0
    mask_threshold: int = 127
    skip_empty_gt: bool = True   # 真值为空（整片无道路）的切片不计入汇总，避免虚高


# ---------------------------------------------------------------- 基础工具

def to_binary(arr, threshold: int = 127) -> np.ndarray:
    """任意 dtype 的影像转布尔掩膜。兼容 0/1、0/255、0/65535。"""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == bool:
        return a
    if a.max(initial=0) <= 1:
        return a > 0
    return a > threshold


def _build_zhang_suen_lut():
    """预计算 Zhang-Suen 两个子迭代的删除查找表（索引 = 8 邻域位模式）。"""
    lut0 = np.zeros(256, dtype=np.uint8)
    lut1 = np.zeros(256, dtype=np.uint8)
    for idx in range(256):
        # 位序：bit0=p2(上), bit1=p3(右上), ... bit7=p9(左上)
        n = [(idx >> k) & 1 for k in range(8)]
        p2, p3, p4, p5, p6, p7, p8, p9 = n
        b = sum(n)
        seq = n + [n[0]]
        a = sum(1 for i in range(8) if seq[i] == 0 and seq[i + 1] == 1)
        if b < 2 or b > 6 or a != 1:
            continue
        if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
            lut0[idx] = 1
        if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
            lut1[idx] = 1
    return lut0, lut1


_LUT0, _LUT1 = _build_zhang_suen_lut()


def skeletonize(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen 细化（查表实现）。输入布尔掩膜，输出单像素宽骨架（布尔）。

    查表版与逐条件判断版逻辑完全等价，但每次迭代只做 8 次移位或运算 + 1 次查表，
    在 1024×1024 的真实掩膜上约快 4 倍。
    """
    img = mask.astype(np.uint8)
    if img.sum() == 0:
        return img.astype(bool)

    for _ in range(max_iter):
        changed = False
        for lut in (_LUT0, _LUT1):
            p = np.pad(img, 1, mode="constant")
            idx = (p[:-2, 1:-1].astype(np.uint8)
                   | (p[:-2, 2:].astype(np.uint8) << 1)
                   | (p[1:-1, 2:].astype(np.uint8) << 2)
                   | (p[2:, 2:].astype(np.uint8) << 3)
                   | (p[2:, 1:-1].astype(np.uint8) << 4)
                   | (p[2:, :-2].astype(np.uint8) << 5)
                   | (p[1:-1, :-2].astype(np.uint8) << 6)
                   | (p[:-2, :-2].astype(np.uint8) << 7))
            cond = (img == 1) & (lut[idx] == 1)
            if cond.any():
                img[cond] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def _skeleton_graph(skel: np.ndarray):
    """把骨架像素建成 8 邻接稀疏图。

    返回 (coords, index_map, csr)。coords 为 (N,2) 的 (row, col)；
    index_map 为与 skel 同形状的数组，骨架像素处为节点编号，其余为 -1。
    """
    ys, xs = np.nonzero(skel)
    n = len(ys)
    index_map = np.full(skel.shape, -1, dtype=np.int64)
    index_map[ys, xs] = np.arange(n)

    rows, cols, vals = [], [], []
    # 右、下、右下、左下 四个方向即可覆盖无向 8 邻接
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)):
        y2, x2 = ys + dy, xs + dx
        inb = (y2 >= 0) & (y2 < skel.shape[0]) & (x2 >= 0) & (x2 < skel.shape[1])
        valid = np.zeros(len(ys), dtype=bool)
        valid[inb] = skel[y2[inb], x2[inb]]
        idx_src = np.nonzero(valid)[0]
        idx_dst = index_map[y2[valid], x2[valid]]
        rows.extend(idx_src.tolist())
        cols.extend(idx_dst.tolist())
        vals.extend([w] * len(idx_src))

    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # 对称化
    return np.column_stack([ys, xs]), index_map, graph


def skeleton_length(skel: np.ndarray) -> float:
    """骨架总长度（像素单位，对角线按 √2 计）。"""
    if skel.sum() == 0:
        return 0.0
    _, _, g = _skeleton_graph(skel)
    return float(g.data.sum() / 2.0)


# ---------------------------------------------------------------- 指标实现

def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """像素级 IoU / F1 / Precision / Recall。全空视为完全正确。"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return {"iou": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0, "empty": True}

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "empty": False}


def line_metrics(pred_skel: np.ndarray, gt_skel: np.ndarray, tol_px: float = 3.0) -> dict:
    """骨架级 precision/recall：在容差带内的骨架像素视为命中。"""
    pred_skel = pred_skel.astype(bool)
    gt_skel = gt_skel.astype(bool)
    n_pred, n_gt = int(pred_skel.sum()), int(gt_skel.sum())

    if n_pred == 0 and n_gt == 0:
        return {"line_precision": 1.0, "line_recall": 1.0, "line_f1": 1.0}
    if n_pred == 0 or n_gt == 0:
        return {"line_precision": 0.0, "line_recall": 0.0, "line_f1": 0.0}

    dist_to_gt = ndi.distance_transform_edt(~gt_skel)
    dist_to_pred = ndi.distance_transform_edt(~pred_skel)

    precision = float((dist_to_gt[pred_skel] <= tol_px).sum()) / n_pred
    recall = float((dist_to_pred[gt_skel] <= tol_px).sum()) / n_gt
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"line_precision": precision, "line_recall": recall, "line_f1": f1}


def apls(pred_skel: np.ndarray, gt_skel: np.ndarray, cfg: EvalConfig) -> float:
    """APLS：在真值骨架上采样点对，比较预测图与真值图的最短路径长度一致性。

    任一端点在容差内找不到对应预测像素，或预测图中两点不连通，该点对记 0 分。

    已知局限（写论文时必须说明，否则会被审稿人抓）：
      APLS 只看路径"长度"是否一致，不惩罚那些不改变长度的伪连接，
      也不惩罚整体平移（长度不变）。因此 APLS 必须与 伪连接率、线P(3px) 同时报告，
      单独用 APLS 会高估质量。
    """
    gt_pts, _, gt_graph = _skeleton_graph(gt_skel)
    if len(gt_pts) < 2:
        return 1.0 if pred_skel.sum() == 0 else 0.0
    pred_pts, _, pred_graph = _skeleton_graph(pred_skel)
    if len(pred_pts) < 2:
        return 0.0

    rng = random.Random(cfg.seed)

    # 只在同一真值连通分量内采样，避免跨分量点对本身无路径
    gt_lab, n_gt = ndi.label(gt_skel, structure=np.ones((3, 3), dtype=int))
    labels_of_pts = gt_lab[gt_pts[:, 0], gt_pts[:, 1]]
    comps = {}
    for i, lab in enumerate(labels_of_pts):
        comps.setdefault(int(lab), []).append(i)
    usable = [v for v in comps.values() if len(v) >= 2]
    if not usable:
        return 1.0 if pred_skel.sum() == 0 else 0.0

    pred_tree = cKDTree(pred_pts)
    pairs = []
    tries = 0
    while len(pairs) < cfg.apls_samples and tries < cfg.apls_samples * 50:
        tries += 1
        comp = rng.choice(usable)
        i, j = rng.sample(comp, 2)
        if i == j:
            continue
        pairs.append((i, j))
    if not pairs:
        return 0.0

    src_gt = [p[0] for p in pairs]
    src_dst = [p[1] for p in pairs]

    # 一次多源 dijkstra，取每对需要的行列
    d_gt = dijkstra(gt_graph, directed=False, indices=sorted(set(src_gt)))
    src_order = {s: k for k, s in enumerate(sorted(set(src_gt)))}

    # 端点映射到预测骨架
    mapped = []
    for idx in range(len(pairs)):
        pair_pts = []
        for node in (pairs[idx][0], pairs[idx][1]):
            d, nn = pred_tree.query(gt_pts[node])
            pair_pts.append(int(nn) if d <= cfg.tol_px else None)
        mapped.append(pair_pts)

    valid_pairs = [(k, m) for k, m in enumerate(mapped) if m[0] is not None and m[1] is not None]
    if not valid_pairs:
        return 0.0

    pred_sources = sorted({p[1][0] for p in valid_pairs} | {p[1][1] for p in valid_pairs})
    d_pred = dijkstra(pred_graph, directed=False, indices=pred_sources)
    pred_order = {s: k for k, s in enumerate(pred_sources)}

    scores = []
    for k, (a, b) in valid_pairs:
        l_gt = d_gt[src_order[src_gt[k]], src_dst[k]]
        l_pred = d_pred[pred_order[a], b]
        if not np.isfinite(l_gt) or l_gt <= 0 or not np.isfinite(l_pred):
            scores.append(0.0)
            continue
        scores.append(max(0.0, 1.0 - abs(l_gt - l_pred) / l_gt))

    # 未映射成功的点对按 0 分计入
    total = len(pairs)
    return float(sum(scores) / total) if total else 0.0


def _samples_by_component(gt_skel: np.ndarray, n: int, seed: int):
    """在真值骨架的同一连通分量内采样点对，返回 (点坐标, 点对列表, 分量像素索引)。"""
    pts, _, _ = _skeleton_graph(gt_skel)
    if len(pts) < 2:
        return pts, [], {}
    lab, _ = ndi.label(gt_skel, structure=np.ones((3, 3), dtype=int))
    labels = lab[pts[:, 0], pts[:, 1]]
    comps = {}
    for i, l in enumerate(labels):
        comps.setdefault(int(l), []).append(i)
    usable = {k: v for k, v in comps.items() if len(v) >= 2}
    if not usable:
        return pts, [], {}

    rng = random.Random(seed)
    keys = list(usable)
    pairs = []
    tries = 0
    while len(pairs) < n and tries < n * 20:
        tries += 1
        k = rng.choice(keys)
        i, j = rng.sample(usable[k], 2)
        if i != j:
            pairs.append((i, j, k))
    return pts, pairs, usable


def connectivity_metrics(pred: np.ndarray, gt: np.ndarray, cfg: EvalConfig,
                          pred_skel: np.ndarray | None = None,
                          gt_skel: np.ndarray | None = None) -> dict:
    """连通率、断裂率、伪连接率与连通分量统计。

    断裂率的定义（2026-10-29 修订）：原实现按"真值分量被 >=2 个预测分量覆盖一定比例"计数，
    实测表明它对覆盖阈值极敏感（阈值 0.02→0.30 时同一批数据从 0.54 降到 0.00），不可用于论文。
    现改为**阈值无关**的图连通判据：在真值骨架的同分量内采样点对，检查映射到预测骨架后是否仍连通；
    断裂率 = 1 - 连通率。伪连接率改用绝对像素下限（默认 30 px ≈ 9 m）判定"实质跨接"，不再用比例阈值。
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    pred_skel = skeletonize(pred_b) if pred_skel is None else pred_skel
    gt_skel = skeletonize(gt_b) if gt_skel is None else gt_skel

    struct = np.ones((3, 3), dtype=int)
    gt_lab, n_gt = ndi.label(gt_b, structure=struct)
    pred_lab, n_pred = ndi.label(pred_b, structure=struct)

    # ---- 连通性：真值点对在预测中是否仍连通
    connectivity, breakage, split_rate = 0.0, 1.0, 1.0
    gt_pts, pairs, usable = _samples_by_component(gt_skel, cfg.apls_samples, cfg.seed)
    pred_pts, _, pred_graph = _skeleton_graph(pred_skel)

    if pairs and len(pred_pts) >= 2:
        pred_tree = cKDTree(pred_pts)
        mapped = []
        for i, j, k in pairs:
            ends = []
            for node in (i, j):
                d, nn = pred_tree.query(gt_pts[node])
                ends.append(int(nn) if d <= cfg.tol_px else None)
            mapped.append((ends[0], ends[1], k))

        srcs = sorted({m[0] for m in mapped if m[0] is not None} |
                      {m[1] for m in mapped if m[1] is not None})
        d_pred = dijkstra(pred_graph, directed=False, indices=srcs) if srcs else np.zeros((0, 0))
        order = {s: t for t, s in enumerate(srcs)}

        ok, comp_status = [], {}
        for a, b, k in mapped:
            if a is None or b is None:
                ok.append(False)
                comp_status[k] = False
            else:
                dist = d_pred[order[a], b]
                good = bool(np.isfinite(dist))
                ok.append(good)
                comp_status[k] = comp_status.get(k, True) and good
        connectivity = float(np.mean(ok)) if ok else 0.0
        breakage = 1.0 - connectivity
        sampled = [v for v in comp_status.values()]
        split_rate = float(sum(1 for v in sampled if not v) / len(sampled)) if sampled else 0.0

    # ---- 伪连接：一个预测分量实质跨接 >=2 个真值分量
    false_link = 0
    for pi in range(1, n_pred + 1):
        sel = pred_lab == pi
        if not sel.any():
            continue
        vals, counts = np.unique(gt_lab[sel], return_counts=True)
        real = [v for v, c in zip(vals, counts) if v != 0 and c >= cfg.min_link_px]
        if len(real) >= 2:
            false_link += 1

    return {
        "connectivity": connectivity,
        "breakage_rate": breakage,
        "split_component_rate": split_rate,
        "false_connection_rate": float(false_link / n_pred) if n_pred else 0.0,
        "gt_components": int(n_gt),
        "pred_components": int(n_pred),
        "cc_ratio": float(n_pred / n_gt) if n_gt else float("inf"),
    }


def evaluate(pred_mask: np.ndarray, gt_mask: np.ndarray, cfg: EvalConfig | None = None) -> dict:
    """一次性算出全部指标。输入为二值或灰度掩膜（自动二值化）。"""
    cfg = cfg or EvalConfig()
    pred = to_binary(pred_mask, cfg.mask_threshold)
    gt = to_binary(gt_mask, cfg.mask_threshold)

    out = {}
    out.update(pixel_metrics(pred, gt))

    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)

    out.update(line_metrics(pred_skel, gt_skel, cfg.tol_px))
    out["apls"] = apls(pred_skel, gt_skel, cfg)
    out.update(connectivity_metrics(pred, gt, cfg, pred_skel, gt_skel))

    len_pred = float(pred_skel.sum())
    len_gt = float(gt_skel.sum())
    out["length_ratio"] = len_pred / len_gt if len_gt else (1.0 if len_pred == 0 else float("inf"))
    out["length_error_m"] = abs(len_pred - len_gt) * cfg.gsd
    return out


METRIC_KEYS = [
    "iou", "f1", "precision", "recall",
    "line_precision", "line_recall", "line_f1",
    "apls", "connectivity", "breakage_rate", "split_component_rate",
    "false_connection_rate", "cc_ratio", "length_ratio",
]

METRIC_NAMES = {
    "iou": "IoU",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "line_precision": "线P(3px)",
    "line_recall": "线R(3px)",
    "line_f1": "线F1",
    "apls": "APLS",
    "connectivity": "连通率",
    "breakage_rate": "断裂率",
    "split_component_rate": "分量断裂率",
    "false_connection_rate": "伪连接率",
    "cc_ratio": "连通分量比",
    "length_ratio": "长度比",
}
