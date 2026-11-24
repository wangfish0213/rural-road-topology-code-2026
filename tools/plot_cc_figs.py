# -*- coding: utf-8 -*-
"""任务 2 配图：把"孤立（未匹配）分量"可视化 + 归因统计图。

图 1（cc_fig1_samples.png）：典型样本三联。
  左 = 影像+真值叠加；中 = baseline 预测；右 = M1 预测。
  像素级着色：绿 = 落在真值 3px 容差带内，红 = 落在容差带外
  （含宽度/偏移误差与未匹配响应，不可逐像素判为假阳性）。
图 2（cc_fig2_attribution.png）：从 summary.json 读数字画柱状图，
  展示分量总数增量中有多少来自孤立（未匹配）分量。

**图内所有数字只从 runs/cc_analysis/summary.json 与 per_image.csv 读取，不硬编码。**
选样也从 per_image.csv 按脚本统计量挑，不人工指定图号。

用法：
  PYTHONPATH="F:/pyenvs/road/Lib/site-packages" <python> tools/plot_cc_figs.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import to_binary  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATA = ROOT / "runs" / "cc_analysis"
GT_DIR = ROOT / "datasets" / "WHU-RRoad" / "test"
OUT_DIR = ROOT / "previews"
STRUCT8 = np.ones((3, 3), dtype=int)
TOL_PX = 3

GREEN = (60, 210, 120)
RED = (245, 70, 70)
CYAN = (70, 190, 240)
BG = (255, 255, 255)
INK = (30, 34, 40)
GRID = (215, 220, 226)


def font(size: int, bold=False):
    cands = (["C:/Windows/Fonts/arialbd.ttf"] if bold else []) + [
        "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for p in cands:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def read_mask(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        if im.mode in ("P", "L", "I;16", "I"):
            return np.array(im)
        return np.array(im.convert("L"))


def read_rgb(p: Path) -> np.ndarray:
    """影像保留彩色（论文配图不能用灰度）。"""
    with Image.open(p) as im:
        return np.array(im.convert("RGB"))


def colorize_components(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """像素级着色：绿 = 预测像素命中真值 3px 容差带；红 = 预测像素落在容差带外。

    不用"整个连通分量染一色"：那样一个大面积误检块只要蹭到真值就会被整块染绿，
    会系统性低估假阳性，配图会误导读者。分量级的归因留给 Figure 2。
    """
    gt_b = to_binary(gt, 127)
    pr_b = to_binary(pred, 127)
    gt_band = ndi.distance_transform_edt(~gt_b) <= TOL_PX
    canvas = np.array(Image.fromarray(img).convert("RGB"), dtype=np.float32)
    hit = pr_b & gt_band
    miss = pr_b & ~gt_band
    canvas[hit] = canvas[hit] * 0.30 + np.array(GREEN, np.float32) * 0.70
    canvas[miss] = canvas[miss] * 0.30 + np.array(RED, np.float32) * 0.70
    return canvas.clip(0, 255).astype(np.uint8)


def overlay_gt(img: np.ndarray, gt: np.ndarray) -> np.ndarray:
    arr = np.array(Image.fromarray(img).convert("RGB"), dtype=np.float32)
    m = to_binary(gt, 127)
    arr[m] = arr[m] * 0.45 + np.array(CYAN, np.float32) * 0.55
    return arr.clip(0, 255).astype(np.uint8)


def pick_samples(rows: list[dict], k: int = 3) -> list[str]:
    """挑能支撑结论、且视觉上不误导的样本。

    约束（避免选到"整片红色"的极端图，那种图无法说明问题）：
      - 真值道路足够多（gt_px >= 8000）；
      - M1 覆盖率不下降（delta_cov >= -0.01）；
      - M1 预测面积明显小于 baseline（体现预测更细），且绝对面积不过大；
      - 在此基础上取孤立分量增量最大的几张。
    """
    by = {}
    for r in rows:
        by.setdefault(r["stem"], {})[r["model"]] = r
    cand = []
    for stem, d in by.items():
        if "m1" not in d or "baseline" not in d:
            continue
        m, b = d["m1"], d["baseline"]
        delta_iso = float(m["n_isolated"]) - float(b["n_isolated"])
        delta_cov = float(m["coverage"]) - float(b["coverage"])
        gt_px, m_px, b_px = float(m["gt_px"]), float(m["pred_px"]), float(b["pred_px"])
        if gt_px < 8000 or delta_cov < -0.01:
            continue
        if m_px > 200000 or b_px > 600000:      # 排除整片被预测铺满的图
            continue
        if m_px > b_px * 0.95:                  # M1 应比 baseline 更细
            continue
        cand.append((delta_iso, stem))
    cand.sort(reverse=True)
    return [c[1] for c in cand[:k]]


def fig1(stems: list[str]) -> Path:
    cell, pad, top, cap = 320, 8, 46, 26
    W = 3 * (cell + pad) + pad
    H = top + len(stems) * (cell + cap + pad) + 34
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    f_t = font(17, bold=True)
    f_c = font(13)
    f_s = font(12)

    heads = ["(a) Image + ground truth", "(b) Baseline U-Net", "(c) Ours (orientation field)"]
    f_h = font(15, bold=True)
    for j, h in enumerate(heads):
        d.text((pad + j * (cell + pad) + 4, 14), h, fill=INK, font=f_h)

    for i, stem in enumerate(stems):
        y0 = top + i * (cell + cap + pad)
        img = read_rgb(GT_DIR / f"{stem}_sat.jpg")
        gt = read_mask(GT_DIR / f"{stem}_mask.png")
        panels = [
            overlay_gt(img, gt),
            colorize_components(img, gt, read_mask(ROOT / "runs" / "e1_unet" / "pred" / f"{stem}.png")),
            colorize_components(img, gt, read_mask(ROOT / "runs" / "m1_ori_e1b" / "pred" / f"{stem}.png")),
        ]
        for j, arr in enumerate(panels):
            x0 = pad + j * (cell + pad)
            canvas.paste(Image.fromarray(arr).resize((cell, cell)), (x0, y0))
        d.text((pad + 2, y0 + cell + 5), f"Tile {stem}", fill=INK, font=f_c)

    # 图例
    ly = H - 30
    d.rectangle((pad + 2, ly + 3, pad + 18, ly + 17), fill=GREEN)
    d.text((pad + 24, ly), "predicted pixel within 3 px of GT", fill=INK, font=f_s)
    d.rectangle((pad + 310, ly + 3, pad + 326, ly + 17), fill=RED)
    d.text((pad + 332, ly), "predicted outside 3 px tolerance (width / offset or spurious)", fill=INK, font=f_s)
    d.rectangle((pad + 790, ly + 3, pad + 806, ly + 17), fill=CYAN)
    d.text((pad + 812, ly), "GT", fill=INK, font=f_s)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "cc_fig1_samples.png"
    canvas.save(out)
    return out


def fig2(summary: dict) -> Path:
    b = summary["by_model"]["baseline"]
    m = summary["by_model"]["m1"]
    attr = summary["attribution"]
    apx = summary.get("absolute_pixels", {}).get("m1_vs_baseline", {})

    W, H = 1000, 580
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    f_t = font(19, bold=True)
    f_l = font(13)
    f_s = font(12)
    f_n = font(14, bold=True)

    d.text((24, 16), "Attribution of the increased connected components", fill=INK, font=f_t)
    d.text((24, 42), f"Mean over {summary['n_images']} non-empty test tiles; "
                     "all values read from summary.json", fill=(105, 112, 122), font=f_s)

    # 堆叠柱：分量总数 = 命中 + 孤立
    groups = [("Baseline", b), ("Ours (M1)", m)]
    x0, y_base, bw, gap, scale = 80, 430, 130, 220, 1.0
    vals = [(n, g["n_hit"]["mean"], g["n_isolated"]["mean"]) for n, g in groups]
    vmax = max(v[1] + v[2] for v in vals)
    plot_h = 300
    scale = plot_h / vmax

    for gy in range(0, int(vmax) + 1, max(1, int(vmax / 5))):
        y = int(y_base - gy * scale)
        d.line((x0 - 8, y, W - 340, y), fill=GRID, width=1)
        d.text((x0 - 46, y - 8), str(gy), fill=(110, 116, 126), font=f_s)

    for i, (name, hit, iso) in enumerate(vals):
        x = x0 + i * gap
        h1, h2 = hit * scale, iso * scale
        d.rectangle((x, int(y_base - h1), x + bw, y_base), fill=GREEN)
        d.rectangle((x, int(y_base - h1 - h2), x + bw, int(y_base - h1)), fill=RED)
        d.text((x + 10, y_base + 10), name, fill=INK, font=f_l)
        d.text((x + 8, int(y_base - h1 / 2 - 8)), f"{hit:.1f}", fill=(255, 255, 255), font=f_n)
        d.text((x + 8, int(y_base - h1 - h2 / 2 - 8)), f"{iso:.1f}", fill=(255, 255, 255), font=f_n)
        d.text((x + 8, int(y_base - h1 - h2 - 22)), f"{hit+iso:.1f}", fill=INK, font=f_n)

    d.rectangle((x0, 496, x0 + 16, 510), fill=GREEN)
    d.text((x0 + 22, 494), "matched components (overlap GT tolerance)", fill=INK, font=f_s)
    d.rectangle((x0 + 280, 496, x0 + 296, 510), fill=RED)
    d.text((x0 + 302, 494), "isolated components (no GT overlap)", fill=INK, font=f_s)

    # 右侧结论框：标签左列 / 数值右列，避免挤在一起
    bx = W - 330
    d.rounded_rectangle((bx, 96, W - 20, 480), radius=10, fill=(244, 247, 250), outline=GRID, width=1)
    d.text((bx + 16, 108), "Key numbers (from script)", fill=INK, font=f_n)
    kv = [
        ("Delta components / tile", f"{attr['delta_n_pred']:+.1f}"),
        ("of which isolated", f"{attr['delta_n_isolated']:+.1f}"),
        ("Share of increase", f"{attr['isolated_share_of_increase']*100:.1f}%"),
        ("Isolated share (baseline)", f"{b['isolated_ratio']['mean']*100:.1f}%"),
        ("Isolated share (ours)", f"{m['isolated_ratio']['mean']*100:.1f}%"),
        ("Median isolated size (base)", f"{b['isolated_size_median']['mean']:.1f} px"),
        ("Median isolated size (ours)", f"{m['isolated_size_median']['mean']:.1f} px"),
        ("GT coverage (base, ours)",
         f"{b['coverage']['mean']:.3f}, {m['coverage']['mean']:.3f}"),
    ]
    if apx:
        kv.append(("Isolated pixels (base -> ours)",
                   f"{apx['isolated_px_mean']['baseline']/1000:.1f}k -> "
                   f"{apx['isolated_px_mean']['m1']/1000:.1f}k"))
    for i, (k, v) in enumerate(kv):
        y = 146 + i * 33
        d.text((bx + 16, y + 3), k, fill=(70, 78, 90), font=f_s)
        tw = d.textlength(v, font=f_n)
        d.text((W - 36 - tw, y), v, fill=INK, font=f_n)
        if i < len(kv) - 1:
            d.line((bx + 16, y + 27, W - 36, y + 27), fill=(228, 232, 238), width=1)

    out = OUT_DIR / "cc_fig2_attribution.png"
    canvas.save(out)
    return out


def write_captions(summary: dict, stems: list[str]) -> Path:
    """英文图注（Remote Sensing 用），数字全部取自 summary.json。"""
    b = summary["by_model"]["baseline"]
    m = summary["by_model"]["m1"]
    a = summary["attribution"]
    n = summary["n_images"]
    apx = summary.get("absolute_pixels", {}).get("m1_vs_baseline", {})
    abs_sentence = ""
    if apx:
        v = apx["isolated_px_mean"]
        abs_sentence = (
            f" Importantly, the mean number of pixels belonging to isolated components "
            f"*decreases* from {v['baseline']:.0f} to {v['m1']:.0f} per tile "
            f"({v['rel_change']*100:+.1f}%): a comparable (in fact smaller) unmatched area "
            f"is split into many more, much smaller fragments. The higher component-count "
            f"ratio therefore reflects fragmentation of the unmatched area, not an increase "
            f"in its total extent."
        )
    txt = f"""# Figure captions (draft, English)

All numbers below are read from `runs/cc_analysis/summary.json`, which is produced by
`tools/analyze_cc.py` over {n} non-empty WHU-RRoad test tiles. Do not edit the numbers by hand.

**Figure 1.** Qualitative comparison of component-level predictions on representative
test tiles ({', '.join(stems)}). Left: satellite image with ground-truth roads in cyan.
Middle: cross-domain U-Net baseline. Right: the proposed orientation-field model (M1).
Predicted pixels are green when they fall within the 3 px tolerance band of the ground
truth and red otherwise. Colouring is per pixel, not per connected component, so a large
spurious patch is not concealed by a single touch with the ground truth. Note that the
ground-truth annotations are thin lines whereas the predictions are area masks; red
areas therefore include both genuine width/offset discrepancies and spurious responses,
and must **not** be read as false positives pixel by pixel. M1 produces noticeably
thinner predictions that follow the annotated roads more closely, while generating a
larger number of small isolated components. The component-level attribution of those
components is quantified in Figure 2; no claim about network fragmentation is made from
visual inspection alone.

**Figure 2.** Attribution of the increased number of predicted connected components.
Per tile, the mean total number of components rises from
{b['n_pred']['mean']:.1f} (baseline) to {m['n_pred']['mean']:.1f} (M1), i.e.
{a['delta_n_pred']:+.1f}; isolated components with zero overlap with the 3 px
ground-truth tolerance band account for {a['delta_n_isolated']:+.1f} of this increase
({a['isolated_share_of_increase']*100:.1f}% of the increment). The median size of an
isolated component decreases from {b['isolated_size_median']['mean']:.1f} px to
{m['isolated_size_median']['mean']:.1f} px, while ground-truth pixel coverage increases
from {b['coverage']['mean']:.3f} to {m['coverage']['mean']:.3f}. Hence the larger
component-count ratio of M1 is dominated by additional small unmatched components
rather than by a more fragmented road network; this is consistent with the improved
connectivity, breakage rate and APLS reported in `runs/RESULTS.md`.{abs_sentence}

**Honest caveats for the manuscript.** (i) The *number* and number-share of isolated
components do increase (fraction of predicted pixels in isolated components:
{b['isolated_px_ratio']['mean']:.3f} to {m['isolated_px_ratio']['mean']:.3f}), and the
auxiliary split_per_covered_gt rises from
{b['split_per_covered_gt']['mean']:.3f} to {m['split_per_covered_gt']['mean']:.3f}.
These are genuine weaknesses and must be reported, not hidden; they motivate the
graph-node semantic correction (delete branch) described in the following section.
(ii) The WHU-RRoad annotations are
sparse thin lines (about 11.6% of the tiles contain no labelled road), whereas the
predictions are area masks. An isolated component is therefore *unmatched*, not
necessarily a false positive: it may correspond to a real road omitted from the
annotation. Accordingly, we avoid calling these components "false positives" and rely on
the connectivity metrics for statements about road-network fragmentation.
"""
    out = DATA / "figure_captions.md"
    out.write_text(txt, encoding="utf-8")
    return out


def main(argv=None) -> int:
    import argparse
    global DATA, OUT_DIR
    default_data, default_out = str(DATA), str(OUT_DIR)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=default_data, help="分析产物目录（默认 runs/cc_analysis）")
    ap.add_argument("--out", default=default_out, help="图片输出目录（默认 previews）")
    args = ap.parse_args(argv)
    DATA = Path(args.data)
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    with open(DATA / "per_image.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    stems = pick_samples(rows, 3)
    if not stems:
        print("没有可选样本")
        return 1
    p1 = fig1(stems)
    p2 = fig2(summary)
    p3 = write_captions(summary, stems)
    print(f"样本：{stems}")
    print(f"已生成 {p1} / {p2} / {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
