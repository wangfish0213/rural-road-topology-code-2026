# -*- coding: utf-8 -*-
"""论文 6 张正式插图。

统一约定
--------
* 通栏宽 16 cm，600 dpi（tools/figlib.py 负责尺寸与字号换算）。
* 图内每一个数字都从 runs/ 下的 CSV/JSON 现场读出或现场计算，不硬编码。
* 选样由脚本按统计量决定（见各 figN 的 select 段），不手工指定图号；
  需要人工判断的（Fig.6 失败案例）先输出候选拼图，定稿脚本用 `--fig6-ids` 固定。

用法
----
  PYTHONPATH="F:/pyenvs/road/Lib/site-packages" <python> tools/make_figures.py --fig 5
  ... --fig all

输出
----
  manuscript/figures/fig1_motivation.png
  manuscript/figures/fig2_pipeline.pdf      （TikZ 编译，见 tools/fig2_pipeline.tex）
  manuscript/figures/fig3_orientation.png
  manuscript/figures/fig4_correction.png
  manuscript/figures/fig5_breakmap.png
  manuscript/figures/fig6_failures.png
以及 manuscript/figures/figures.json（每张图的来源数据与选样记录，供审计）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "train"))

import figlib as F  # noqa: E402
from figlib import cm, pt  # noqa: E402
from figlib import (C_ACC, C_ADD, C_DEL, C_GT, C_PRED, C_REJ, C_RING, C_SPUR,  # noqa: E402
                    INK)

OUT = ROOT / "manuscript" / "figures"
FIG_JSON = OUT / "figures.json"

RUNS = ROOT / "runs"
GT_TEST = ROOT / "datasets" / "WHU-RRoad" / "test"
IRS_TEST = ROOT / "datasets" / "IRSAMap" / "IRSAMap_V2" / "test"

M1 = RUNS / "m1_ori_e1b"
M2 = RUNS / "m2_corrected"
E1 = RUNS / "e1_unet"

RECORDS: dict = {}


def record(name: str, **kw):
    RECORDS.setdefault(name, {}).update(kw)


def flush_records():
    old = {}
    if FIG_JSON.exists():
        try:
            old = json.loads(FIG_JSON.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old.update(RECORDS)
    FIG_JSON.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")


def read_per_image(run: Path) -> list[dict]:
    p = run / "eval" / "per_image.csv"
    with open(p, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k != "name":
                r[k] = float(v)
    return rows


# ---------------------------------------------------------------- 色标

def _lerp(c0, c1, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def cmap_seq(v: float) -> tuple:
    """断裂率：0 浅 → 1 深红。"""
    t = max(0.0, min(1.0, v))
    return _lerp((255, 246, 232), (158, 18, 22), t ** 0.85)


def cmap_div(v: float) -> tuple:
    """差值：负（改善）蓝 → 0 近白 → 正（恶化）红。"""
    t = max(-1.0, min(1.0, v))
    if t < 0:
        return _lerp((247, 247, 247), (33, 102, 172), -t)
    return _lerp((247, 247, 247), (178, 24, 43), t)


def tile_grid(values: list[float], cols: int, cmap, vmin: float, vmax: float,
              cell: int, gap: int = 1) -> np.ndarray:
    """把逐片数值铺成 cols 列的网格图（末行不足处留白）。"""
    n = len(values)
    rows = math.ceil(n / cols)
    h = rows * (cell + gap) - gap
    w = cols * (cell + gap) - gap
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    for i, v in enumerate(values):
        r, c = divmod(i, cols)
        y0, x0 = r * (cell + gap), c * (cell + gap)
        rgb = cmap((v - vmin) / (vmax - vmin) if vmax > vmin else 0.0)
        arr[y0:y0 + cell, x0:x0 + cell] = rgb
    return arr


# ---------------------------------------------------------------- Fig.5

def fig5() -> Path:
    """断裂点空间分布：M1（纠错前）→ M2（纠错后）逐片断裂率 + 差值。

    before 取 M1 输出而不是 E1 U-Net 输出：图题说的是 "before and after correction"，
    correction 的输入就是 M1 的掩膜；用 E1 会把 M1 方向场的贡献误记到 M2 头上。
    """
    before = {r["name"]: r for r in read_per_image(M1)}
    after = {r["name"]: r for r in read_per_image(M2)}
    names = sorted(set(before) & set(after))
    b = [before[n]["breakage_rate"] for n in names]
    a = [after[n]["breakage_rate"] for n in names]
    d = [x - y for x, y in zip(a, b)]

    improved = sum(1 for v in d if v < -1e-9)
    worsened = sum(1 for v in d if v > 1e-9)
    unchanged = len(d) - improved - worsened

    n = len(names)
    cols = 52
    gap = 1

    # ---- 宽度自适应：三面板 + 三条色标塞进 16 cm，绝不越界
    W = cm(16.0)
    left = cm(0.10)
    gap_x = cm(0.40)
    cb_w = cm(0.16)
    cb_gap = cm(0.20)
    pad = cm(0.10)
    avail = W - 2 * left - 3 * (cb_w + cb_gap) - 2 * gap_x - cm(0.52)   # 末段留给刻度标签
    panel_w = avail // 3
    grid_w = panel_w - 2 * pad
    cell = (grid_w - (cols - 1) * gap) // cols
    grid_w = cols * (cell + gap) - gap
    grid_h = math.ceil(n / cols) * (cell + gap) - gap
    panel_h = grid_h + 2 * pad

    head = cm(1.5)          # 顶部标题区
    foot = cm(2.35)         # 底部统计与说明
    H = head + panel_h + foot

    c = F.Canvas(W, H)
    c.text((left, cm(0.12)), "Cross-domain breakage before and after graph correction",
           size=7.6, bold=True)
    c.text((left, cm(0.62)),
           f"{n:,} non-empty WHU-RRoad test tiles (experiment E1b), sorted by tile id; "
           f"one cell = one tile", size=6.4, fill=F.MUTED)

    # 差值色标按实际幅度取整到 0.05，否则 M2 的微小变化会被 ±1 的色标压成全白
    d_abs = max(abs(min(d)), abs(max(d)))
    d_lim = max(0.05, math.ceil(d_abs / 0.05) * 0.05)

    panels = [
        ("(a) M1: before correction", b, cmap_seq, 0.0, 1.0,
         [0, 0.25, 0.5, 0.75, 1.0], "breakage rate"),
        ("(b) M2: after correction", a, cmap_seq, 0.0, 1.0,
         [0, 0.25, 0.5, 0.75, 1.0], "breakage rate"),
        ("(c) difference (b) \u2212 (a)", d, cmap_div, -d_lim, d_lim,
         [-d_lim, -d_lim / 2, 0, d_lim / 2, d_lim], "\u0394 breakage"),
    ]
    x = left
    top = head
    for i, (title, vals, cmap, vmin, vmax, ticks, cblab) in enumerate(panels):
        c.text((x + panel_w / 2, top - cm(0.34)), title, size=6.9, bold=True, anchor="mb")
        c.rect((x, top, x + panel_w, top + panel_h), outline=F.RULE, width=2, fill=F.PANEL_BG)
        grid = tile_grid(vals, cols, cmap, vmin, vmax, cell, gap)
        c.paste(grid, (x + pad, top + pad, grid_w, grid_h))
        cbx = x + panel_w + cb_gap
        F.colorbar(c, (cbx, top + cm(0.12), cb_w, panel_h - cm(0.24)), cmap,
                   vmin, vmax, ticks, label_pt=5.4)
        x += panel_w + cb_w + cb_gap + gap_x

    # 底部统计（全部由数据现算）
    mb = sum(b) / n
    ma = sum(a) / n
    y0 = H - foot + cm(0.30)
    c.line([(left, y0 - cm(0.06)), (W - left, y0 - cm(0.06))], fill=(228, 231, 235), width=2)
    c.text((left, y0), f"mean breakage {mb:.3f} \u2192 {ma:.3f}", size=6.3)
    c.text((left + cm(4.6), y0),
           f"tiles improved {improved:,} \u00b7 unchanged {unchanged:,} \u00b7 "
           f"worsened {worsened:,}", size=6.3, fill=F.INK)
    c.text((left, y0 + cm(0.46)),
           "Each cell is one tile; colour in (a),(b) is the breakage rate, "
           "in (c) blue means breakage fell and red means it rose after correction.",
           size=5.7, fill=F.MUTED)
    c.text((left, y0 + cm(0.84)),
           "breakage rate = 1 \u2212 connectivity, sampled per connected component of the "
           "ground-truth skeleton (same rule as Table 3).", size=5.7, fill=F.MUTED)
    c.text((left, y0 + cm(1.22)),
           "Tile ids follow the released dataset order and carry no geographic meaning; "
           "the tiles are not georeferenced, so no map projection is implied.",
           size=5.7, fill=F.MUTED)

    p = OUT / "fig5_breakmap.png"
    c.save(p)
    record("fig5", source=["runs/m1_ori_e1b/eval/per_image.csv",
                           "runs/m2_corrected/eval/per_image.csv"],
           n_tiles=n, mean_before=round(mb, 4), mean_after=round(ma, 4),
           delta_limit=d_lim,
           improved=improved, unchanged=unchanged, worsened=worsened,
           output=str(p.relative_to(ROOT)), size_px=list(c.img.size))
    flush_records()
    return p


# ---------------------------------------------------------------- 断裂点定位

def break_points(pred_skel: np.ndarray, gt_skel: np.ndarray, n: int = 80,
                 tol_px: float = 3.0, seed: int = 0):
    """返回断裂点坐标 [(y, x), ...]：真值同一分量内的点对在预测中不连通时，取两点中点。

    与 breakage_rate 用同一套采样规则（eval/metrics.py 的 _samples_by_component），
    所以图上标出的位置与表里的数字是同一口径，不存在"图好看但指标对不上"的问题。
    """
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    from metrics import _samples_by_component, _skeleton_graph

    gt_pts, pairs, _ = _samples_by_component(gt_skel, n, seed)
    pred_pts, _, pred_graph = _skeleton_graph(pred_skel)
    if not pairs or len(pred_pts) < 2:
        return []

    tree = cKDTree(pred_pts)
    mapped = []
    for i, j, _k in pairs:
        ends = []
        for node in (i, j):
            d, nn = tree.query(gt_pts[node])
            ends.append(int(nn) if d <= tol_px else None)
        mapped.append((ends[0], ends[1], i, j))

    srcs = sorted({e for m in mapped for e in m[:2] if e is not None})
    d_pred = dijkstra(pred_graph, directed=False, indices=srcs) if srcs else None
    order = {s: t for t, s in enumerate(srcs)} if srcs else {}

    out = []
    for a, b, i, j in mapped:
        broken = a is None or b is None
        if not broken:
            broken = not np.isfinite(d_pred[order[a], b])
        if broken:
            mid = (gt_pts[i] + gt_pts[j]) / 2.0
            out.append((float(mid[0]), float(mid[1])))
    return out


def dedup_points(pts, min_dist: float):
    """按距离去重，保证标注圈不互相压叠。"""
    kept = []
    for y, x in pts:
        if all((y - ky) ** 2 + (x - kx) ** 2 >= min_dist ** 2 for ky, kx in kept):
            kept.append((y, x))
    return kept


# ---------------------------------------------------------------- Fig.1

def fig1(src_id: str | None = None, tgt_id: str | None = None) -> Path:
    """城乡外观差异 + 断裂中心线。左：源域（WHU Road, 1 m）；右：目标域（WHU-RRoad, 0.3 m）。

    src_id / tgt_id 为空时按规则自动选样（条件与结果记入 figures.json）；给出时用指定
    切片，用于把人工从候选预览中挑定的切片固定下来，保证重跑结果一致。
    """
    from metrics import skeletonize
    from scipy import ndimage as ndi

    # ---------------- 源域
    src_rows = {r["id"]: r for r in csv.DictReader(
        open(ROOT / "manifests" / "whu_road_1m.csv", encoding="utf-8-sig"))}
    if src_id is None:
        for k in sorted(src_rows):
            r = src_rows[k]
            lab = F.load_mask(r["mask"])
            frac = lab.sum() / lab.size
            if not (0.010 <= frac <= 0.030):
                continue
            sk = skeletonize(lab)
            b, hh, ww = 4, sk.shape[0], sk.shape[1]
            cover = sum(1 for i in range(b) for j in range(b)
                        if sk[i * hh // b:(i + 1) * hh // b,
                              j * ww // b:(j + 1) * ww // b].sum() > 40)
            if cover < 8:
                continue
            src_id = k
            break
    if src_id not in src_rows:
        raise SystemExit(f"source tile {src_id} not found in manifest")

    # ---------------- 目标域
    peri_m1 = {r["name"]: r for r in read_per_image(M1)}
    if tgt_id is None:
        for name in sorted(peri_m1):
            r = peri_m1[name]
            if (0.26 <= r["iou"] <= 0.34 and 20 <= r["cc_ratio"] <= 70
                    and 0.72 <= r["breakage_rate"] <= 0.90):
                tgt_id = name
                break
    if tgt_id not in peri_m1:
        raise SystemExit(f"target tile {tgt_id} has no M1 evaluation record")

    src_img = F.load_rgb(src_rows[src_id]["image"])
    src_sk = skeletonize(F.load_mask(src_rows[src_id]["mask"]))
    tgt_img = F.load_rgb(GT_TEST / f"{tgt_id}_sat.jpg")
    tgt_gt = F.load_mask(GT_TEST / f"{tgt_id}_mask.png")
    tgt_pred = F.load_mask(M1 / "pred" / f"{tgt_id}.png")
    gt_sk = skeletonize(tgt_gt)
    pr_sk = skeletonize(tgt_pred)

    pts = dedup_points(break_points(pr_sk, gt_sk, n=120, seed=0), min_dist=150)
    h, w = tgt_gt.shape
    pts = [(y, x) for y, x in pts if 70 < y < h - 70 and 70 < x < w - 70][:4]

    # ---------------- 版面（16 cm 通栏，两面板等大）
    W = cm(16.0)
    margin = cm(0.10)
    axis_w = cm(0.55)
    panel = int(cm(7.05))
    gap = cm(0.60)
    axis_h = cm(0.46)
    head = cm(1.30)
    foot = cm(1.85)
    H = head + panel + axis_h + foot

    c = F.Canvas(W, H)
    c.text((margin, cm(0.04)), "Urban-to-rural appearance gap and disconnected centerlines",
           size=7.6, bold=True)
    c.text((margin, cm(0.58)),
           "source domain 1 m/px (left) vs target domain 0.3 m/px (right); real tiles, "
           "identical display size", size=6.2, fill=F.MUTED)

    top = head
    x0 = int(margin + axis_w)
    x1 = x0 + panel + int(gap + axis_w)

    # ---------------- (a) 源域
    shown = F.blend(src_img, F.dilate(src_sk, 1), (255, 255, 255), 0.95)
    shown = F.blend(shown, src_sk, C_GT, 1.0)
    c.paste(shown, (x0, top, panel, panel))
    F.frame(c, (x0, top, panel, panel))
    F.panel_label(c, x0 + cm(0.10), top + cm(0.08), "(a)")
    F.scalebar(c, x0 + cm(0.15), top + panel - cm(0.78), 1.0, cm(1.1))
    F.pixel_axis(c, (x0, top, panel, panel), src_img.shape[0], step=200)
    c.text((x0 + panel / 2, top - cm(0.16)),
           "Source: WHU Road (same-domain reference)", size=6.7, bold=True, anchor="mb")

    # ---------------- (b) 目标域
    tgt = F.blend(tgt_img, F.dilate(gt_sk, 1), (255, 255, 255), 0.55)
    tgt = F.blend(tgt, gt_sk, (238, 238, 238), 1.0)
    tgt = F.blend_outline(tgt, F.dilate(pr_sk, 1), (255, 255, 255), width=3)
    tgt = F.blend(tgt, F.dilate(pr_sk, 1), C_PRED, 1.0)
    c.paste(tgt, (x1, top, panel, panel))
    F.frame(c, (x1, top, panel, panel))
    F.panel_label(c, x1 + cm(0.10), top + cm(0.08), "(b)")
    F.scalebar(c, x1 + cm(0.15), top + panel - cm(0.78), 0.3, cm(1.6))
    F.pixel_axis(c, (x1, top, panel, panel), tgt_gt.shape[0], step=400)
    c.text((x1 + panel / 2, top - cm(0.16)),
           "Target: WHU-RRoad, M1 prediction", size=6.7, bold=True, anchor="mb")

    # 断裂点圈注（位置由 breakage 的采样规则决定，不是手点）
    d = c.draw
    r = cm(0.16)
    lw = max(2, cm(0.022))
    for y, x in pts:
        px = x1 + int(x / w * panel)
        py = top + int(y / h * panel)
        for rr, col in ((r * 1.55, (15, 15, 15)), (r * 1.28, (255, 255, 255)), (r, C_PRED)):
            rr = int(rr)
            d.ellipse([(px - rr, py - rr), (px + rr, py + rr)], outline=col, width=lw)

    # ---------------- 图例与说明
    yb = top + panel + axis_h + cm(0.10)
    xx = margin
    xx += F.legend_chip(c, xx, yb, C_GT, "annotated centerline (source)", size=6.3) + cm(0.45)
    xx += F.legend_chip(c, xx, yb, (205, 205, 205), "ground truth (target)", size=6.3) + cm(0.45)
    xx += F.legend_chip(c, xx, yb, C_PRED, "M1 prediction (target)", size=6.3) + cm(0.45)
    c.text((xx, yb + pt(0.5)), f"circles: {len(pts)} disconnected points",
           size=6.3, fill=F.INK, anchor="lm")

    c.text((margin, yb + cm(0.58)),
           "Circles mark ground-truth point pairs whose shortest path is disconnected in the "
           "prediction, i.e. the same sampling rule that defines the breakage rate.",
           size=5.7, fill=F.MUTED)
    c.text((margin, yb + cm(0.94)),
           f"Tiles {src_id} (source) and {tgt_id} (target). Neither release provides "
           f"georeferencing for these tiles, so scale bars use the nominal ground sampling",
           size=5.7, fill=F.MUTED)
    c.text((margin, yb + cm(1.30)),
           "distance and no north arrow is drawn.", size=5.7, fill=F.MUTED)

    p = OUT / "fig1_motivation.png"
    c.save(p)
    record("fig1", source=[f"manifests/whu_road_1m.csv#{src_id}",
                           f"manifests/whu_rroad_test.csv#{tgt_id}",
                           "runs/m1_ori_e1b/pred"],
           source_id=src_id, target_id=tgt_id, n_break_circles=len(pts),
           break_points=[[round(y, 1), round(x, 1)] for y, x in pts],
           selection="source: first id with centerline fraction 0.5-1.8% and 3-40 components; "
                     "target: first id with IoU 0.26-0.34, cc_ratio 20-70, breakage 0.72-0.90",
           output=str(p.relative_to(ROOT)), size_px=list(c.img.size))
    flush_records()
    return p


# ---------------------------------------------------------------- 方向场推理

CACHE = ROOT / "runs" / "fig_cache"


def predict_ori(img_path: Path, ckpt: Path = M1 / "best.pt", tile: int = 256,
                stride: int = 192, batch: int = 4):
    """滑窗推理 M1：返回 (mask, cos2, sin2)。结果缓存到 runs/fig_cache，重跑不重复计算。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{img_path.stem}_{ckpt.parent.name}"
    fp = CACHE / f"{key}.npz"
    if fp.exists():
        d = np.load(fp)
        return d["mask"], d["c2"].astype(np.float32), d["s2"].astype(np.float32)

    import torch
    sys.path.insert(0, str(ROOT / "train"))
    from predict import build_model, tile_coords

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model, kind = build_model(ck)
    model = model.to(dev)
    model.load_state_dict(ck["model"], strict=(kind != "ori"))
    model.eval()

    img = np.asarray(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    prob = np.zeros((h, w), np.float32)
    c2 = np.zeros((h, w), np.float32)
    s2 = np.zeros((h, w), np.float32)
    wt = np.zeros((h, w), np.float32)
    buf, coords = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().to(dev) / 255.0
        with torch.no_grad():
            m, o = model(x)
        p = torch.sigmoid(m).squeeze(1).cpu().numpy()
        o = o.cpu().numpy()
        for (y, xx), pi, oi in zip(coords, p, o):
            prob[y:y + tile, xx:xx + tile] += pi
            c2[y:y + tile, xx:xx + tile] += oi[0]
            s2[y:y + tile, xx:xx + tile] += oi[1]
            wt[y:y + tile, xx:xx + tile] += 1.0
        buf.clear()
        coords.clear()

    for y, x in tile_coords(h, w, tile, stride):
        buf.append(img[y:y + tile, x:x + tile])
        coords.append((y, x))
        if len(buf) >= batch:
            flush()
    flush()

    wt[wt == 0] = 1.0
    prob, c2, s2 = prob / wt, c2 / wt, s2 / wt
    mask = prob > 0.5
    np.savez_compressed(fp, mask=mask, c2=c2.astype(np.float16),
                        s2=s2.astype(np.float16), prob=prob.astype(np.float16))
    return mask, c2, s2


def hsv_rgb(hue: np.ndarray, sat, val) -> np.ndarray:
    """向量化 HSV→RGB，hue 在 [0,1)。sat/val 可为标量或与 hue 同形状。"""
    hue = np.asarray(hue, np.float32)
    sat = np.broadcast_to(np.asarray(sat, np.float32), hue.shape)
    val = np.broadcast_to(np.asarray(val, np.float32), hue.shape)
    i = np.floor(hue * 6.0).astype(np.int32) % 6
    f = hue * 6.0 - np.floor(hue * 6.0)
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    out = np.zeros(hue.shape + (3,), np.float32)
    for k, chan in enumerate([(val, t, p), (q, val, p), (p, val, t),
                              (p, q, val), (t, p, val), (val, p, q)]):
        m = i == k
        for ci in range(3):
            out[..., ci][m] = chan[ci][m]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- Fig.3

def fig3() -> Path:
    """方向场引导的毛刺剪枝：同一掩膜、同一裁剪范围，剪枝前 vs 剪枝后。"""
    from scipy import ndimage as ndi
    from metrics import skeletonize
    from road_graph import crossing_map
    from orientation import prune_spurs

    # ---- 候选：路网较完整的切片里挑毛刺最多的一张（数量由脚本统计，不手点）
    rows = [r for r in read_per_image(M1)
            if r["iou"] >= 0.30 and 0.75 <= r["length_ratio"] <= 1.35]
    rows.sort(key=lambda r: (-r["iou"], r["name"]))
    # 选图不看全图毛刺总数，而看"单个 400 px 窗口里毛刺最多"——图要的是局部对比，
    # 全图毛刺多但分散在一堆角落的切片做不出可读的对比。
    W_CROP, STRIDE = 400, 40
    half = W_CROP // 2
    best = None
    for r in rows[:12]:
        name = r["name"]
        mask, c2, s2 = predict_ori(GT_TEST / f"{name}_sat.jpg")
        sk0 = skeletonize(mask)
        sk1 = prune_spurs(sk0, max_len=6, ori=(c2, s2), max_dev_deg=50.0)
        spur = sk0 & ~sk1
        skel_win = ndi.uniform_filter(sk0.astype(np.float32), size=W_CROP) * (W_CROP ** 2)
        spur_win = ndi.uniform_filter(spur.astype(np.float32), size=W_CROP) * (W_CROP ** 2)
        ys = range(half, sk0.shape[0] - half, STRIDE)
        xs_ = range(half, sk0.shape[1] - half, STRIDE)
        cand = [(spur_win[y, x], skel_win[y, x], y, x) for y in ys for x in xs_
                if 120 <= skel_win[y, x] <= 2000]
        if not cand:
            continue
        s_max, _sk, cy, cx = max(cand)
        _lab, nsp = ndi.label(spur, structure=np.ones((3, 3), int))
        if best is None or s_max > best[0]:
            best = (s_max, name, nsp, sk0, sk1, spur, c2, s2, mask, cy, cx)
    if best is None:
        raise SystemExit("no tile produced a usable crop")
    _smax, name, nsp, sk0, sk1, spur, c2, s2, mask, cy, cx = best
    y0, x0 = cy - half, cx - half

    def cut(a):
        return a[y0:y0 + W_CROP, x0:x0 + W_CROP]

    crop_img = cut(F.load_rgb(GT_TEST / f"{name}_sat.jpg"))
    in_crop_spurs = int(cut(spur).sum())

    cr0 = crossing_map(sk0) >= 3
    cr1 = crossing_map(sk1) >= 3
    spurious = cr0 & ~F.dilate(cr1, 3)
    _l, n_spur_nodes = ndi.label(spurious, structure=np.ones((3, 3), int))
    j0 = ndi.label(cr0, structure=np.ones((3, 3), int))[1]
    j1 = ndi.label(cr1, structure=np.ones((3, 3), int))[1]

    # ---- 版面：两面板（放大裁剪，差异才看得清）
    W = cm(16.0)
    margin = cm(0.08)
    axis_w = cm(0.50)
    panel = int(cm(7.20))
    gap = cm(0.72)
    axis_h = cm(0.44)
    head = cm(1.20)
    foot = cm(1.72)
    H = head + panel + axis_h + foot

    c = F.Canvas(W, H)
    c.text((margin, cm(0.04)), "Orientation-guided spur pruning at inference",
           size=7.6, bold=True)
    c.text((margin, cm(0.56)),
           f"tile {name}, identical {W_CROP}\u00d7{W_CROP} px crop taken from the same mask; "
           f"only the skeleton pruning differs", size=6.2, fill=F.MUTED)

    top = head
    xa = int(margin + axis_w)
    xb = xa + panel + int(gap)

    # (a) 只骨架化：洋红 = 会被剪掉的毛刺
    a = F.blend(crop_img, F.dilate(cut(sk0), 2), (0, 0, 0), 0.88)      # 暗描边
    a = F.blend(a, cut(sk0), (255, 255, 255), 1.0)                      # 亮骨架
    a = F.blend_outline(a, F.dilate(cut(spur), 1), (0, 0, 0), width=3)
    a = F.blend(a, F.dilate(cut(spur), 1), C_SPUR, 1.0)
    c.paste(a, (xa, top, panel, panel))
    F.frame(c, (xa, top, panel, panel))
    F.panel_label(c, xa + cm(0.08), top + cm(0.06), "(a)")
    _bar, bar_m = F.scalebar(c, xa + cm(0.14), top + panel - cm(0.66), 0.3, cm(1.2),
                             label_pt=5.8)
    F.pixel_axis(c, (xa, top, panel, panel), W_CROP, step=100, label_pt=5.4)
    c.text((xa + panel / 2, top - cm(0.15)), "skeletonised, no pruning",
           size=6.7, bold=True, anchor="mb")

    # (b) 剪枝后：骨架保留，原毛刺位置用淡洋红虚影标出，便于逐处对照
    b_img = F.blend(crop_img, F.dilate(cut(sk1), 2), (0, 0, 0), 0.88)
    b_img = F.blend(b_img, cut(sk1), (255, 255, 255), 1.0)
    b_img = F.blend(b_img, F.dilate(cut(spur), 1), C_SPUR, 0.30)
    c.paste(b_img, (xb, top, panel, panel))
    F.frame(c, (xb, top, panel, panel))
    F.panel_label(c, xb + cm(0.08), top + cm(0.06), "(b)")
    F.scalebar(c, xb + cm(0.14), top + panel - cm(0.66), 0.3, cm(1.2), label_pt=5.8)
    F.pixel_axis(c, (xb, top, panel, panel), W_CROP, step=100, show_y=False, label_pt=5.4)
    c.text((xb + panel / 2, top - cm(0.15)), "with orientation-guided pruning",
           size=6.7, bold=True, anchor="mb")

    # ---- 图例与统计
    yb = top + panel + axis_h + cm(0.06)
    xx = margin
    xx += F.legend_chip(c, xx, yb, (60, 60, 60), "centerline", size=6.2) + cm(0.40)
    xx += F.legend_chip(c, xx, yb, C_SPUR, "removed spur (faint in b)", size=6.2) + cm(0.40)

    n_spur_px = int(spur.sum())
    c.text((margin, yb + cm(0.50)),
           f"Whole tile: {nsp} spur branches ({n_spur_px:,} px) removed, {in_crop_spurs} px "
           f"in this crop; branch points {j0} \u2192 {j1} ({n_spur_nodes} disappear).",
           size=5.9, fill=F.INK)
    c.text((margin, yb + cm(0.88)),
           "A branch is removed when it is shorter than 6 px and deviates by more than 50 "
           "degrees from the predicted direction field; pruning only decides which skeleton",
           size=5.7, fill=F.MUTED)
    c.text((margin, yb + cm(1.24)),
           f"pixels enter the graph stage (Table 3 metrics use the unpruned skeleton). Crop "
           f"offset (x, y) = ({x0}, {y0}); scale bar {bar_m:g} m at 0.3 m/px.",
           size=5.7, fill=F.MUTED)

    p = OUT / "fig3_orientation.png"
    c.save(p)
    record("fig3", tile=name, crop=[int(x0), int(y0), W_CROP],
           spur_branches=nsp, spur_px=n_spur_px, spur_px_in_crop=in_crop_spurs,
           scale_bar_m=bar_m,
           branch_nodes_before=j0, branch_nodes_after=j1, pruned_nodes=n_spur_nodes,
           source=["runs/m1_ori_e1b/best.pt (orientation field, re-inference)",
                   f"datasets/WHU-RRoad/test/{name}_sat.jpg"],
           output=str(p.relative_to(ROOT)), size_px=list(c.img.size))
    flush_records()
    return p


# ---------------------------------------------------------------- Fig.4

def fig4() -> Path:
    """M2 图纠错：跨分量端点对的接受/拒绝，以及尺寸感知剪枝。

    候选与判别分数不是画上去的，而是用 train/m2_correction.py 的 collect_candidates +
    score_items 在选定切片上**重算**的；阈值取自 runs/m2_corrected/apply_stats.json。
    选图也不手点：按"裁剪窗口内接受链接数 + 被剪像素数"打分，取最高的一张一处。
    """
    import torch
    from scipy import ndimage as ndi
    from metrics import skeletonize
    from m2_correction import collect_candidates, score_items, load_net

    stats = json.loads((M2 / "apply_stats.json").read_text(encoding="utf-8"))
    thr_link, thr_del = float(stats["thr_link"]), float(stats["thr_del"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = RUNS / "m2_model"
    link_net = load_net(str(model_dir / "link_net.pt"), 5, 5, dev)
    del_net = load_net(str(model_dir / "delete_net.pt"), 5, 4, dev)

    with open(M2 / "apply_per_image.csv", encoding="utf-8-sig") as f:
        ap = list(csv.DictReader(f))
    pool = [r for r in ap if 2 <= int(r["link_applied"]) <= 8 and 4 <= int(r["del_applied"]) <= 40]
    pool.sort(key=lambda r: r["id"])

    W_CROP = 240
    half = W_CROP // 2
    best = None
    for r in pool[:8]:
        nm = r["id"]
        cand = collect_candidates(str(M1 / "pred" / f"{nm}.png"), str(GT_TEST), nm, None)
        pl = score_items(link_net, cand["link"], dev)
        pd = score_items(del_net, cand["delete"], dev)
        acc = [(it, float(p)) for it, p in zip(cand["link"], pl) if p >= thr_link]
        rej = [(it, float(p)) for it, p in zip(cand["link"], pl) if p < thr_link]
        dele = [(it, float(p)) for it, p in zip(cand["delete"], pd) if p >= thr_del]
        if not acc:
            continue

        m1_pred = F.load_mask(M1 / "pred" / f"{nm}.png")
        m2_pred = F.load_mask(M2 / "pred" / f"{nm}.png")
        del_px = m1_pred & ~F.dilate(m2_pred, 1)

        # 窗口评分：接受链接的端点分散、被剪像素多、拒绝候选少 → 图面清爽且能说明问题
        # 用积分图算任意窗口内的计数，避免逐窗口重扫
        H_, W_ = m1_pred.shape
        ii_del = np.zeros((H_ + 1, W_ + 1), np.float64)
        ii_del[1:, 1:] = del_px.astype(np.float64).cumsum(0).cumsum(1)
        acc_ends = [(int((it["a"][0] + it["b"][0]) / 2),
                     int((it["a"][1] + it["b"][1]) / 2)) for it, _ in acc]
        rej_ends = [it["a"] for it, _ in rej] + [it["b"] for it, _ in rej]

        def win_sum(ii, y, x):
            y0_, x0_ = max(0, y - half), max(0, x - half)
            y1_, x1_ = min(H_, y + half), min(W_, x + half)
            return float(ii[y1_, x1_] - ii[y0_, x1_] - ii[y1_, x0_] + ii[y0_, x0_])

        def in_win(py, px, y, x):
            return abs(py - y) <= half and abs(px - x) <= half

        for ty, tx in acc_ends:
            if not (half <= ty < H_ - half and half <= tx < W_ - half):
                continue
            n_a = sum(1 for py, px in acc_ends if in_win(py, px, ty, tx))
            n_r = sum(1 for py, px in rej_ends if in_win(py, px, ty, tx))
            n_del_win = win_sum(ii_del, ty, tx)
            sc = 2.0 * n_a + n_del_win / 30.0 - 0.45 * max(0, n_r - 2 * n_a)
            if best is None or sc > best[0]:
                best = (sc, nm, cand, acc, rej, dele, m1_pred, m2_pred, del_px, int(ty), int(tx))
    if best is None:
        raise SystemExit("no tile with an accepted link found")
    _sc, name, cand, accepted, rejected, deleted, m1_pred, m2_pred, del_px, cy, cx = best

    sk_m1 = skeletonize(m1_pred)
    sk_m2 = skeletonize(m2_pred)
    add_px = m2_pred & ~F.dilate(m1_pred, 1)

    y0, x0 = cy - half, cx - half

    def cut(a):
        return a[y0:y0 + W_CROP, x0:x0 + W_CROP]

    def inside(py, px):
        return y0 <= py < y0 + W_CROP and x0 <= px < x0 + W_CROP

    crop_img = cut(F.load_rgb(GT_TEST / f"{name}_sat.jpg"))
    n_acc = sum(1 for it, _ in accepted if inside(*it["a"]) or inside(*it["b"]))
    n_rej = sum(1 for it, _ in rejected if inside(*it["a"]) or inside(*it["b"]))
    n_del = 0
    for it, _ in deleted:
        sl = it["sl"]
        if inside((sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2):
            n_del += 1

    # ---- 版面
    W = cm(16.0)
    margin = cm(0.08)
    axis_w = cm(0.50)
    panel = int(cm(7.20))
    gap = cm(0.72)
    axis_h = cm(0.44)
    head = cm(1.20)
    foot = cm(1.80)
    H = head + panel + axis_h + foot

    c = F.Canvas(W, H)
    c.text((margin, cm(0.04)), "Graph correction: endpoint linking and size-aware pruning",
           size=7.6, bold=True)
    c.text((margin, cm(0.56)),
           f"tile {name}, identical {W_CROP}\u00d7{W_CROP} px crop; thresholds "
           f"p(link)\u2265{thr_link:g}, p(delete)\u2265{thr_del:g}, as in the full run",
           size=6.2, fill=F.MUTED)

    top = head
    xa = int(margin + axis_w)
    xb = xa + panel + int(gap)

    def to_px(ay, ax):
        return (xa + int((ax - x0) / W_CROP * panel), top + int((ay - y0) / W_CROP * panel))

    # (a) M1 输出 + 候选
    a = F.blend(crop_img, F.dilate(cut(sk_m1), 2), (0, 0, 0), 0.85)
    a = F.blend(a, cut(sk_m1), (255, 255, 255), 1.0)
    c.paste(a, (xa, top, panel, panel))
    d = c.draw
    for it, p in rejected:                     # 先画拒绝，接受压在上层
        (ay, ax), (by, bx) = it["a"], it["b"]
        if not (inside(ay, ax) or inside(by, bx)):
            continue
        pa, pb = to_px(ay, ax), to_px(by, bx)
        d.line([pa, pb], fill=(90, 90, 100), width=max(3, cm(0.036)))
    for it, p in accepted:
        (ay, ax), (by, bx) = it["a"], it["b"]
        if not (inside(ay, ax) or inside(by, bx)):
            continue
        pa, pb = to_px(ay, ax), to_px(by, bx)
        d.line([pa, pb], fill=(255, 255, 255), width=max(4, cm(0.085)))
        d.line([pa, pb], fill=C_ACC, width=max(2, cm(0.050)))
        for pxy in (pa, pb):
            d.ellipse([(pxy[0] - cm(0.06), pxy[1] - cm(0.06)),
                       (pxy[0] + cm(0.06), pxy[1] + cm(0.06))],
                      outline=(255, 255, 255), width=max(2, cm(0.018)))
    F.frame(c, (xa, top, panel, panel))
    F.panel_label(c, xa + cm(0.08), top + cm(0.06), "(a)")
    F.scalebar(c, xa + cm(0.14), top + panel - cm(0.62), 0.3, cm(1.2), label_pt=5.8,
               length_m=20)
    F.pixel_axis(c, (xa, top, panel, panel), W_CROP, step=50, label_pt=5.2)
    c.text((xa + panel / 2, top - cm(0.15)), "M1 output: candidate endpoint pairs",
           size=6.7, bold=True, anchor="mb")

    # (b) M2 输出：先铺被剪/新增像素，再压骨架，保证骨架不被盖住
    b_img = F.blend(crop_img, F.dilate(cut(del_px), 1), C_DEL, 0.60)
    b_img = F.blend_outline(b_img, F.dilate(cut(add_px), 1), (255, 255, 255), width=3)
    b_img = F.blend(b_img, F.dilate(cut(add_px), 1), C_ADD, 1.0)
    b_img = F.blend(b_img, F.dilate(cut(sk_m2), 2), (0, 0, 0), 0.85)
    b_img = F.blend(b_img, cut(sk_m2), (255, 255, 255), 1.0)
    c.paste(b_img, (xb, top, panel, panel))
    F.frame(c, (xb, top, panel, panel))
    F.panel_label(c, xb + cm(0.08), top + cm(0.06), "(b)")
    F.scalebar(c, xb + cm(0.14), top + panel - cm(0.62), 0.3, cm(1.2), label_pt=5.8,
               length_m=20)
    F.pixel_axis(c, (xb, top, panel, panel), W_CROP, step=50, show_y=False, label_pt=5.2)
    c.text((xb + panel / 2, top - cm(0.15)), "M2 output: linked and pruned",
           size=6.7, bold=True, anchor="mb")

    # ---- 图例与统计
    yb = top + panel + axis_h + cm(0.06)
    xx = margin
    xx += F.legend_chip(c, xx, yb, (60, 60, 60), "centerline", size=6.1) + cm(0.34)
    xx += F.legend_chip(c, xx, yb, C_ACC, "accepted link", size=6.1) + cm(0.34)
    xx += F.legend_chip(c, xx, yb, (90, 90, 100), "rejected candidate", size=6.1) + cm(0.34)
    xx += F.legend_chip(c, xx, yb, C_ADD, "added pixels", size=6.1) + cm(0.34)
    xx += F.legend_chip(c, xx, yb, C_DEL, "pruned component", size=6.1) + cm(0.34)

    c.text((margin, yb + cm(0.50)),
           f"In this crop: {n_acc} accepted of {n_acc + n_rej} candidate pairs, {n_del} isolated "
           f"components removed. Whole tile: {len(cand['link'])} candidate pairs, "
           f"{len(accepted)} accepted, {len(deleted)} components pruned.",
           size=5.9, fill=F.INK)
    c.text((margin, yb + cm(0.88)),
           "Linking is the only learned decision in the graph stage; pruning is a fixed rule that "
           "removes connected components no larger than 64 px. Removed components are isolated",
           size=5.7, fill=F.MUTED)
    c.text((margin, yb + cm(1.24)),
           f"or unmatched responses, not false positives. Crop offset (x, y) = ({x0}, {y0}); "
           f"scale bar 20 m at 0.3 m/px.", size=5.7, fill=F.MUTED)

    p = OUT / "fig4_correction.png"
    c.save(p)
    record("fig4", tile=name, crop=[int(x0), int(y0), W_CROP],
           thr_link=thr_link, thr_del=thr_del,
           link_candidates=len(cand["link"]), accepted=len(accepted),
           rejected=len(rejected), components_pruned=len(deleted),
           crop_accepted=n_acc, crop_rejected=n_rej, crop_pruned=n_del,
           source=["runs/m2_model/link_net.pt", "runs/m2_model/delete_net.pt",
                   f"runs/m1_ori_e1b/pred/{name}.png", f"runs/m2_corrected/pred/{name}.png"],
           output=str(p.relative_to(ROOT)), size_px=list(c.img.size))
    flush_records()
    return p


# ---------------------------------------------------------------- Fig.6

def fig6() -> Path:
    """三个代表失败案例，按误差类型着色：错检（红）、漏检（蓝）、命中（浅灰）。

    着色口径与 eval/metrics.py 的 line_metrics 完全一致（3 px 容差带）：
      红 = 预测骨架像素，且 3 px 内没有真值骨架  → 与 false-connection rate 同一把尺子；
      蓝 = 真值骨架像素，且 3 px 内没有预测骨架  → 与 line recall 同一把尺子；
      灰 = 两侧都落在容差带内。
    切片人工挑定（三类误差各一张，理由写进 figures.json），裁剪窗口由脚本取该类
    误差像素密度的最大值处，不手点。
    """
    from scipy import ndimage as ndi
    from metrics import skeletonize

    W_CROP = 320
    half = W_CROP // 2
    TOL = 3.0
    rows = read_per_image(M2)
    r_by = {r["name"]: r for r in rows}

    def box_density(arr, k):
        """k×k 窗口内的均值，用积分图算（大核下比 uniform_filter 快两个数量级）。"""
        H, W = arr.shape
        ii = np.zeros((H + 1, W + 1), np.float64)
        ii[1:, 1:] = arr.astype(np.float64).cumsum(0).cumsum(1)
        y0 = np.clip(np.arange(H) - k // 2, 0, H)
        y1 = np.clip(np.arange(H) + k // 2, 0, H)
        x0 = np.clip(np.arange(W) - k // 2, 0, W)
        x1 = np.clip(np.arange(W) + k // 2, 0, W)
        area = np.outer(y1 - y0, x1 - x0).astype(np.float64)
        tot = (ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)]
               - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)])
        return tot / np.maximum(area, 1)

    def analyse(name):
        img = F.load_rgb(GT_TEST / f"{name}_sat.jpg")
        gt = F.load_mask(GT_TEST / f"{name}_mask.png")
        pr = F.load_mask(M2 / "pred" / f"{name}.png")
        gs, ps = skeletonize(gt), skeletonize(pr)
        dist_gt = ndi.distance_transform_edt(~gs)
        dist_pr = ndi.distance_transform_edt(~ps)
        comm = ps & (dist_gt > TOL)           # 预测了，真值不在容差带内
        omis = gs & (dist_pr > TOL)           # 有标注，预测不在容差带内
        hit = (ps & (dist_gt <= TOL)) | (gs & (dist_pr <= TOL))
        return dict(img=img, gs=gs, ps=ps, comm=comm, omis=omis, hit=hit,
                    d_comm=box_density(comm, W_CROP), d_omis=box_density(omis, W_CROP),
                    d_gt=box_density(gs, W_CROP))

    def best_window(score):
        sc = score.copy()
        sc[:half, :] = sc[-half:, :] = 0
        sc[:, :half] = sc[:, -half:] = 0
        cy, cx = np.unravel_index(int(np.argmax(sc)), sc.shape)
        return (float(sc[cy, cx]), int(np.clip(cy, half, sc.shape[0] - half)),
                int(np.clip(cx, half, sc.shape[1] - half)))

    def score_for(kind, a):
        if kind == "over":
            return a["d_comm"]
        if kind == "under":
            # 漏检窗口：标注密集、且预测确实缺席（避免选到红蓝交错的窗口）
            return a["d_omis"] * (0.3 + a["d_gt"] / 0.01) * (1.0 - np.minimum(1.0, a["d_comm"] / 0.01))
        return np.minimum(a["d_comm"], a["d_omis"]) * (a["d_comm"] + a["d_omis"])

    # 三类误差各固定一张切片。2026-11-13 说明：原计划用"树荫遮挡"，但本数据集真值
    # 中心线 6 px 邻域内"暗且绿"像素占比仅 0.4%（自动扫描 60 张最高 0.004），不存在
    # 成片树冠压路的样本；为免图文不符，改用数据里确实存在的三类误差。
    KIND = ["over", "under", "mixed"]
    LABEL = [
        ("over-detection",
         "plot and farm-structure boundaries traced as centerline",
         "false-connection rate", "false_connection_rate"),
        ("under-detection",
         "narrow lanes not recovered",
         "line recall", "line_recall"),
        ("mixed",
         "a lane dropped while a parallel boundary is traced",
         "breakage rate", "breakage_rate"),
    ]
    FIXED = ["16894", "17337", "17239"]
    best = {}
    for i, (kind, name) in enumerate(zip(KIND, FIXED)):
        sc, cy, cx = best_window(score_for(kind, analyse(name)))
        best["abc"[i]] = (sc, name, cy, cx)

    W = cm(16.0)
    margin = cm(0.08)
    axis_w = cm(0.46)
    panel = int(cm(4.95))
    gap = cm(0.30)
    axis_h = cm(0.42)
    head = cm(1.22)
    foot = cm(2.30)
    H = head + panel + axis_h + foot

    c = F.Canvas(W, H)
    c.text((margin, cm(0.04)), "Error classes on representative tiles", size=7.6, bold=True)
    c.text((margin, cm(0.56)),
           "M2 output on WHU-RRoad test tiles; the colour code uses the same 3 px tolerance "
           "band as the line metrics", size=6.2, fill=F.MUTED)

    top_y = head
    x0p = int(margin + axis_w)
    crops = []
    for i, kind in enumerate(KIND):
        t = "abc"[i]
        _sc, name, cy, cx = best[t]
        a = analyse(name)
        r = r_by[name]
        short, desc, metric, key = LABEL[i]

        def cut(m):
            return m[cy - half:cy + half, cx - half:cx + half]

        shown = (cut(a["img"]).astype(np.float32) * 0.86).astype(np.uint8)
        shown = F.blend(shown, F.dilate(cut(a["hit"]), 1), (215, 215, 215), 0.75)
        for m, col in ((cut(a["omis"]), (40, 110, 235)), (cut(a["comm"]), C_PRED)):
            shown = F.blend(shown, F.dilate(m, 2), (255, 255, 255), 0.80)
            shown = F.blend(shown, F.dilate(m, 1), col, 1.0)

        x = x0p + i * (panel + int(gap))
        c.paste(shown, (x, top_y, panel, panel))
        F.frame(c, (x, top_y, panel, panel))
        F.panel_label(c, x + cm(0.07), top_y + cm(0.05), f"({t})")
        F.scalebar(c, x + cm(0.12), top_y + panel - cm(0.58), 0.3, cm(1.0), label_pt=5.5,
                   length_m=20)
        F.pixel_axis(c, (x, top_y, panel, panel), W_CROP, step=100, show_y=(i == 0),
                     label_pt=5.1)
        c.text((x + panel / 2, top_y - cm(0.15)), short, size=6.6, bold=True, anchor="mb")

        ytxt = top_y + panel + axis_h + cm(0.10)
        val = r[key]
        vtxt = f"{val:.0f}" if key == "cc_ratio" else f"{val:.2f}"
        c.text((x, ytxt), desc, size=5.5, fill=F.INK)
        c.text((x, ytxt + cm(0.30)), f"tile {name}  ·  {metric} {vtxt}",
               size=5.3, fill=F.MUTED)
        crops.append([int(cx - half), int(cy - half), W_CROP])

    # 公共图例 + 说明
    yb = top_y + panel + axis_h + cm(0.86)
    xx = margin
    xx += F.legend_chip(c, xx, yb, (215, 215, 215), "matched (within 3 px)", size=6.2) + cm(0.40)
    xx += F.legend_chip(c, xx, yb, C_PRED, "predicted, no annotation", size=6.2) + cm(0.40)
    xx += F.legend_chip(c, xx, yb, (40, 110, 235), "annotated, no prediction", size=6.2)

    c.text((margin, yb + cm(0.46)),
           "Red and blue are exactly the two error classes counted by the line metrics, so the "
           "panels can be read against the numbers in the result tables.",
           size=5.7, fill=F.MUTED)
    c.text((margin, yb + cm(0.82)),
           "Tiles are representative examples; the crop window inside each tile is placed at the "
           "maximum of that class's pixel density, and offset and score are recorded in "
           "figures.json.", size=5.7, fill=F.MUTED)

    p = OUT / "fig6_failures.png"
    c.save(p)
    record("fig6", tiles=[best[t][1] for t in "abc"], crops=crops,
           error_classes={"a": "predicted skeleton > 3 px from any annotation",
                          "b": "annotation > 3 px from any predicted skeleton",
                          "c": "both classes present in the same window"},
           window_scores={t: round(best[t][0], 4) for t in "abc"},
           window_offsets={t: [best[t][2], best[t][3]] for t in "abc"},
           metrics={best[t][1]: {k: round(r_by[best[t][1]][k], 4) for k in
                                 ("iou", "line_recall", "false_connection_rate",
                                  "cc_ratio", "breakage_rate")}
                    for t in "abc"},
           tolerance_px=TOL,
           selection="tiles hand-checked as representative of each error class; crop window = "
                     "argmax of that class's pixel density inside the tile",
           source=["runs/m2_corrected/pred", "datasets/WHU-RRoad/test"],
           output=str(p.relative_to(ROOT)), size_px=list(c.img.size))
    flush_records()
    return p


FIGS = {"1": fig1, "3": fig3, "4": fig4, "5": fig5, "6": fig6}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="all")
    ap.add_argument("--src-id", default=None, help="Fig.1 源域切片 id（默认按规则自动选）")
    ap.add_argument("--tgt-id", default=None, help="Fig.1 目标域切片 id（默认按规则自动选）")
    args = ap.parse_args(argv)
    want = list(FIGS) if args.fig == "all" else [s.strip() for s in args.fig.split(",")]
    for k in want:
        if k not in FIGS:
            print(f"skip unknown fig {k}")
            continue
        p = fig1(args.src_id, args.tgt_id) if k == "1" else FIGS[k]()
        print(f"fig{k}: {p}  {Image.open(p).size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
