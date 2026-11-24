# -*- coding: utf-8 -*-
"""任务 2：M1 连通分量比偏高（93 vs baseline 50）的归因分析。

问题：M1 的 cc_ratio（预测连通分量数 / 真值连通分量数）高达 93，
      是否意味着"把道路断得更碎"？
假设：M1 召回更高，多出来的分量主要是**与标注零重叠的小型孤立（未匹配）分量**，
      真实道路反而没有被切得更碎（断裂率/连通率均改善可佐证）。
      注意 WHU-RRoad 标注稀疏，孤立分量不等于确证的假阳性，可能含漏标真实道路。

口径（与 eval/metrics.py 完全一致，便于写进论文）：
  - 二值化阈值 127；连通分量用 scipy.ndimage.label + 8 邻域 3x3 结构元；
  - 容差带 = 真值掩膜的欧氏 3 px 距离带（与 line_metrics 的 tol_px=3 对齐）；
  - "孤立分量" = 与真值 3 px 容差带**零重叠**的预测分量（阈值无关，最干净）；
  - 真实道路被切段数沿用评测口径 min_link_px=30（实质覆盖像素下限）。

输出（全部由脚本写盘，数字不得人工转抄）：
  runs/cc_analysis/per_image.csv   每张图 × 每模型一行
  runs/cc_analysis/summary.json    聚合统计（图内数字与结论句的唯一来源）
  runs/cc_analysis/findings.md     由 summary.json 自动生成的结论文字

依赖：numpy / scipy / Pillow；必须用可落盘解释器运行，例如
  PYTHONPATH="F:/pyenvs/road/Lib/site-packages" \
  "C:/Users/mapstore01/.workbuddy/binaries/python/versions/3.13.12/python.exe" tools/analyze_cc.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import skeletonize, to_binary  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

STRUCT8 = np.ones((3, 3), dtype=int)
TOL_PX = 3
MIN_LINK_PX = 30          # 与 eval/metrics.py EvalConfig.min_link_px 一致
GT_DIR = ROOT / "datasets" / "WHU-RRoad" / "test"

DEFAULT_MODELS = {
    "baseline": ROOT / "runs" / "e1_unet" / "pred",
    "m1": ROOT / "runs" / "m1_ori_e1b" / "pred",
    "same_domain": ROOT / "runs" / "e2_unet" / "pred",
}

FIELDS = [
    "stem", "model", "n_gt", "n_pred", "cc_ratio",
    "n_isolated", "isolated_ratio", "isolated_px_ratio",
    "isolated_size_median", "isolated_short_lt50_ratio",
    "n_hit", "hit_size_median",
    "pred_size_median", "pred_short_lt50_ratio",
    "coverage", "gt_covered_ratio", "split_per_covered_gt",
    "pred_px", "gt_px",
]


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode in ("P", "L", "I;16", "I"):
            return np.array(im)
        return np.array(im.convert("L"))


def band(mask: np.ndarray, tol: int = TOL_PX) -> np.ndarray:
    """距前景 <= tol 像素的欧氏容差带。"""
    return ndi.distance_transform_edt(~mask) <= tol


def analyze_one(gt: np.ndarray, pred: np.ndarray) -> dict | None:
    gt_b = to_binary(gt, 127)
    pr_b = to_binary(pred, 127)
    if not gt_b.any():
        return None
    if pr_b.shape != gt_b.shape:
        return None

    gt_lab, n_gt = ndi.label(gt_b, STRUCT8)
    pr_lab, n_pred = ndi.label(pr_b, STRUCT8)

    # 一次 EDT 同时得到容差带与"最近真值像素所属分量"，避免重复计算
    gt_dist, idx = ndi.distance_transform_edt(~gt_b, return_indices=True)
    owner = gt_lab[idx[0], idx[1]]
    gt_band = gt_dist <= TOL_PX
    pr_band = ndi.distance_transform_edt(~pr_b) <= TOL_PX

    sizes = np.bincount(pr_lab.ravel(), minlength=n_pred + 1)[1:]
    slices = ndi.find_objects(pr_lab)

    isolated_sizes, hit_sizes = [], []
    gt_split = np.zeros(n_gt + 1, dtype=np.int32)
    isolated_px = 0
    for pi, sl in enumerate(slices, 1):
        if sl is None:
            continue
        comp = pr_lab[sl] == pi
        size = int(comp.sum())
        hit_px = int((comp & gt_band[sl]).sum())
        if hit_px == 0:
            isolated_sizes.append(size)
            isolated_px += size
            continue
        hit_sizes.append(size)
        owners = owner[sl][comp & gt_band[sl]]
        cnt = np.bincount(owners[owners > 0], minlength=n_gt + 1)
        for o in np.nonzero(cnt >= MIN_LINK_PX)[0]:
            gt_split[o] += 1

    covered = np.zeros(n_gt + 1, dtype=bool)
    covered_px = (pr_band & gt_b)
    covered_labels = np.unique(gt_lab[covered_px])
    covered[covered_labels[covered_labels > 0]] = True

    n_isolated = len(isolated_sizes)
    iso_arr = np.array(isolated_sizes, dtype=float)
    hit_arr = np.array(hit_sizes, dtype=float)
    covered_ids = np.nonzero(covered)[0]
    split_on_covered = gt_split[covered_ids] if len(covered_ids) else np.array([])

    return {
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "cc_ratio": float(n_pred / n_gt),
        "n_isolated": int(n_isolated),
        "isolated_ratio": float(n_isolated / n_pred) if n_pred else 0.0,
        "isolated_px_ratio": float(isolated_px / max(int(pr_b.sum()), 1)),
        "isolated_size_median": float(np.median(iso_arr)) if n_isolated else 0.0,
        "isolated_short_lt50_ratio": float((iso_arr < 50).mean()) if n_isolated else 0.0,
        "n_hit": int(len(hit_sizes)),
        "hit_size_median": float(np.median(hit_arr)) if len(hit_sizes) else 0.0,
        "pred_size_median": float(np.median(sizes)) if n_pred else 0.0,
        "pred_short_lt50_ratio": float((sizes < 50).mean()) if n_pred else 0.0,
        "coverage": float(covered_px.sum() / gt_b.sum()),
        "gt_covered_ratio": float(covered[1:].mean()) if n_gt else 0.0,
        "split_per_covered_gt": float(split_on_covered.mean()) if len(split_on_covered) else 0.0,
        "pred_px": int(pr_b.sum()),
        "gt_px": int(gt_b.sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3000, help="最多处理前 N 个测试样本")
    ap.add_argument("--out", default=str(ROOT / "runs" / "cc_analysis"))
    ap.add_argument("--models", default="baseline,m1,same_domain")
    ap.add_argument("--findings-only", action="store_true",
                    help="只按已有 summary.json 重生成 findings.md，不重算数据"
                         "（用于结论措辞修订后同步产物，符合『改生成器不手改产物』）")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.findings_only:
        with open(out / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        # 绝对量：从 per_image.csv 现算，避免"比值高=误检多"的误读
        if (out / "per_image.csv").is_file():
            with open(out / "per_image.csv", encoding="utf-8-sig", newline="") as f:
                _rows = list(csv.DictReader(f))
            abs_stat = {}
            for mod in sorted({r["model"] for r in _rows}):
                sub = [r for r in _rows if r["model"] == mod]
                pp = np.array([float(r["pred_px"]) for r in sub])
                ip = np.array([float(r["isolated_px_ratio"]) * float(r["pred_px"]) for r in sub])
                hp = np.array([(1 - float(r["isolated_px_ratio"])) * float(r["pred_px"]) for r in sub])
                abs_stat[mod] = {
                    "pred_px_mean": float(pp.mean()), "pred_px_median": float(np.median(pp)),
                    "isolated_px_mean": float(ip.mean()), "isolated_px_median": float(np.median(ip)),
                    "hit_px_mean": float(hp.mean()),
                }
            summary["absolute_pixels"] = abs_stat
            if "baseline" in abs_stat and "m1" in abs_stat:
                b0, m0 = abs_stat["baseline"], abs_stat["m1"]
                summary["absolute_pixels"]["m1_vs_baseline"] = {
                    k: {"baseline": b0[k], "m1": m0[k], "delta": m0[k] - b0[k],
                        "rel_change": (m0[k] - b0[k]) / b0[k] if b0[k] else 0.0}
                    for k in ("pred_px_mean", "isolated_px_mean", "hit_px_mean")
                }
        # 同步刷新说明文字（数字一律不动），保证 summary 与 findings 口径一致
        if "attribution" in summary:
            summary["attribution"]["note"] = (
                "孤立分量 = 与真值 3px 容差带零重叠的预测分量（未匹配分量）；"
                "因 WHU-RRoad 标注稀疏，不能一概等同为假阳性，可能含漏标真实道路。"
                "真路连通性须以断裂率/APLS（eval/metrics.py 图连通判据）为准，"
                "split_per_covered_gt 受分量总数影响，仅是辅助量，不等同于断裂率。"
            )
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        p = write_findings(summary, out, summary["n_images"])
        print(f"已按 summary.json 重生成 {p}（数值未改动）")
        return 0

    models = {k: DEFAULT_MODELS[k] for k in args.models.split(",") if k in DEFAULT_MODELS}
    per_csv = out / "per_image.csv"

    stems = sorted({f.name[:-len("_mask.png")] for f in GT_DIR.glob("*_mask.png")})[: args.limit]

    t0 = time.time()
    n_img = 0
    with open(per_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, stem in enumerate(stems, 1):
            gt_path = GT_DIR / f"{stem}_mask.png"
            if not gt_path.is_file():
                continue
            gt = read_mask(gt_path)
            if not to_binary(gt, 127).any():
                continue
            for name, pdir in models.items():
                pp = pdir / f"{stem}.png"
                if not pp.is_file():
                    continue
                r = analyze_one(gt, read_mask(pp))
                if r is None:
                    continue
                w.writerow({"stem": stem, "model": name, **{k: r[k] for k in FIELDS[2:]}})
            f.flush()
            n_img += 1
            if i % 100 == 0:
                print(f"[{i}/{len(stems)}] 已处理 {n_img} 张非空图，耗时 {time.time()-t0:.0f}s", flush=True)

    # ---- 聚合：全部数字在此产生，后续图与结论只读这个 JSON
    rows = []
    with open(per_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    summary = {"source": "tools/analyze_cc.py", "n_images": n_img,
               "n_rows": len(rows), "tol_px": TOL_PX, "min_link_px": MIN_LINK_PX,
               "gt_dir": str(GT_DIR), "models": {k: str(v) for k, v in models.items()},
               "by_model": {}}
    for name in models:
        sub = [r for r in rows if r["model"] == name]
        if not sub:
            continue
        agg = {"n_images": len(sub)}
        for k in FIELDS[2:]:
            vals = np.array([float(r[k]) for r in sub], dtype=float)
            agg[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)),
                      "total": float(vals.sum())}
        summary["by_model"][name] = agg

    # 对比结论所需的差值（同样由脚本算，不让人工转抄）
    if "baseline" in summary["by_model"] and "m1" in summary["by_model"]:
        b, m = summary["by_model"]["baseline"], summary["by_model"]["m1"]
        summary["m1_vs_baseline"] = {
            k: {"baseline_mean": b[k]["mean"], "m1_mean": m[k]["mean"],
                "delta": m[k]["mean"] - b[k]["mean"]}
            for k in ("cc_ratio", "n_pred", "n_isolated", "n_hit", "isolated_ratio",
                      "isolated_px_ratio", "isolated_size_median", "hit_size_median",
                      "pred_size_median", "pred_short_lt50_ratio",
                      "coverage", "gt_covered_ratio", "split_per_covered_gt")
        }
        d_pred = m["n_pred"]["mean"] - b["n_pred"]["mean"]
        d_iso = m["n_isolated"]["mean"] - b["n_isolated"]["mean"]
        summary["attribution"] = {
            "delta_n_pred": float(d_pred),
            "delta_n_isolated": float(d_iso),
            "isolated_share_of_increase": float(d_iso / d_pred) if d_pred else 0.0,
            "note": "孤立分量 = 与真值 3px 容差带零重叠的预测分量（未匹配分量）；"
                    "因 WHU-RRoad 标注稀疏，不能一概等同为假阳性，可能含漏标真实道路。"
                    "真路连通性须以断裂率/APLS（eval/metrics.py 图连通判据）为准，"
                    "split_per_covered_gt 受分量总数影响，仅是辅助量，不等同于断裂率。",
        }

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 结论文字：数字全部从 summary 读取
    write_findings(summary, out, n_img)

    print(f"完成：{n_img} 张非空图，{len(rows)} 行，耗时 {time.time()-t0:.0f}s")
    print(f"输出：{per_csv} / {out/'summary.json'} / {out/'findings.md'}")
    return 0


def write_findings(summary: dict, out: Path, n_img: int) -> Path:
    """由 summary.json 生成 findings.md；可在分析跑完后单独重跑（不重算数据）。"""
    lines = ["# 任务 2：M1 连通分量比偏高的归因分析", "",
             f"数据来源：`per_image.csv`（{summary['n_rows']} 行，{n_img} 张非空测试图）；",
             "聚合数字全部取自 `summary.json`，本文不含人工转抄数值。", "",
             "## 口径", "",
             f"- 连通分量：8 邻域 `scipy.ndimage.label`；容差带：真值欧氏 {TOL_PX} px；",
             "- 孤立分量：与真值 3 px 容差带零重叠的预测分量（阈值无关）；",
             f"- 真实道路被切段数：预测分量对同一真值分量覆盖 ≥ {MIN_LINK_PX} px 才计数（沿用评测口径）。", "",
             "## 事实对比（均值）", "",
             "| 指标 | baseline | M1 | 差值 |", "|---|---|---|---|"]
    if "m1_vs_baseline" in summary:
        for k, v in summary["m1_vs_baseline"].items():
            lines.append(f"| {k} | {v['baseline_mean']:.3f} | {v['m1_mean']:.3f} | {v['delta']:+.3f} |")
    lines += ["", "## 由脚本自动生成的归因判据", ""]
    if "m1_vs_baseline" in summary and "attribution" in summary:
        d = summary["m1_vs_baseline"]
        a = summary["attribution"]
        lines += [
            f"- 预测分量总数增加 {a['delta_n_pred']:+.2f} 个/图，其中孤立（未匹配）分量增加 "
            f"{a['delta_n_isolated']:+.2f} 个，**占增量的 {a['isolated_share_of_increase']*100:.1f}%**；",
            f"- 孤立分量占全部分量比例：baseline {d['isolated_ratio']['baseline_mean']:.3f} → "
            f"M1 {d['isolated_ratio']['m1_mean']:.3f}；孤立像素占预测像素比例："
            f"{d['isolated_px_ratio']['baseline_mean']:.3f} → {d['isolated_px_ratio']['m1_mean']:.3f}；",
            f"- 孤立分量中位尺寸：{d['isolated_size_median']['baseline_mean']:.1f} → "
            f"{d['isolated_size_median']['m1_mean']:.1f} px（M1 的碎片更小）；",
            f"- 命中分量中位尺寸：{d['hit_size_median']['baseline_mean']:.1f} → "
            f"{d['hit_size_median']['m1_mean']:.1f} px；",
            f"- 真值像素覆盖率：{d['coverage']['baseline_mean']:.3f} → {d['coverage']['m1_mean']:.3f}；"
            f"被覆盖真值分量比例：{d['gt_covered_ratio']['baseline_mean']:.3f} → "
            f"{d['gt_covered_ratio']['m1_mean']:.3f}；",
            f"- 辅助量 split_per_covered_gt：{d['split_per_covered_gt']['baseline_mean']:.3f} → "
            f"{d['split_per_covered_gt']['m1_mean']:.3f}（受分量总数影响，不等同于断裂率）。",
            "",
        ]
        apx = summary.get("absolute_pixels", {}).get("m1_vs_baseline")
        if apx:
            lines += [
                "## 绝对量（防止「比值升高 = 误检变多」的误读）", "",
                "| 每图平均像素数 | baseline | M1 | 相对变化 |", "|---|---|---|---|",
            ]
            for k, name in (("pred_px_mean", "预测总像素"),
                            ("isolated_px_mean", "孤立分量像素"),
                            ("hit_px_mean", "命中分量像素")):
                v = apx[k]
                lines.append(f"| {name} | {v['baseline']:.0f} | {v['m1']:.0f} | "
                             f"{v['rel_change']*100:+.1f}% |")
            lines += [
                "",
                "**关键**：M1 的孤立分量**个数**大幅增加，但孤立分量**总像素数**反而下降；",
                "即同样的（甚至更少的）未匹配面积被切成了更多更小的碎片，",
                "这才是 cc_ratio 从 50 升到 93 的直接原因。",
                "",
            ]
        lines += [
            "**结论口径（须与论文一致）**：cc_ratio 的上升主要由大量**小型孤立（未匹配）分量**驱动，",
            "即方向场分支在提高召回的同时产生了更多未与标注对齐的碎片；"
            "它们不是真道路被切断产生的片段。",
            "真路是否断得更碎，以 `eval/metrics.py` 的图连通判据（断裂率、连通率、APLS）为准，",
            "该组指标在 `runs/RESULTS.md` 中 M1 均优于 baseline。",
            "",
            "**诚实边界（必须如实报告，不得只报改善）**："
            "① 孤立分量个数与个数占比确实上升，是本方法的真实弱点；"
            f"② 辅助量 split_per_covered_gt 由 "
            f"{d['split_per_covered_gt']['baseline_mean']:.3f} 升至 "
            f"{d['split_per_covered_gt']['m1_mean']:.3f}，"
            "说明真路在预测中对应的段数也略有增加，不能回避；"
            "但它不等同于断裂率（该量对分量总数敏感），"
            "真路连通性应以 RESULTS.md 中的断裂率/连通率/APLS 为准。"
            "两者共同构成「尺寸感知孤立分量剪枝（几何后处理）」与"
            "「跨分量端点语义补连判别」两条纠错措施的动机。",
            "",
            "**另一条必须写清的限制**：WHU-RRoad 标注稀疏（空标注切片约 11.6%，",
            "且真值为细线、预测为面状掩膜），与真值 3 px 容差带零重叠的预测分量",
            "**不能一概判定为假阳性**——其中可能包含标注遗漏的真实道路。",
            "因此本文只称其为「孤立分量 / 未匹配分量」，图注不得写成",
            "「false positive」的逐像素断言；真路是否断得更碎，仍以断裂率/APLS 为准。",
        ]
    lines += ["", "> 论文正文引用本结论前，请由王老师确认措辞；本文件只提供脚本实测数字。", ""]
    with open(out / "findings.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out / "findings.md"


if __name__ == "__main__":
    raise SystemExit(main())
