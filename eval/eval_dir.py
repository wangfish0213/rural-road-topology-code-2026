# -*- coding: utf-8 -*-
"""批量评测：逐张计算指标并汇总，支持多进程与 Markdown 输出。

用法示例
--------
    python eval_dir.py --gt ../datasets/WHU_RRoad/test/label --pred ../runs/unet/pred \
        --gsd 0.3 --workers 8 --out results/unet_rr.csv

    # 只看前 200 张（调参阶段用）
    python eval_dir.py --gt ... --pred ... --limit 200

输出
----
  -per-image CSV（每张一行，全指标）
  -汇总 CSV（均值/中位数/标准差）
  -终端打印可直接粘进论文的 Markdown 表
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

from metrics import (METRIC_KEYS, METRIC_NAMES, EvalConfig, evaluate)

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

_CFG = EvalConfig()


def read_mask(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("P", "L", "I;16", "I"):
        return np.array(img)
    return np.array(img.convert("L"))


def _work(job):
    stem, gt_path, pred_path, cfg = job
    try:
        gt = read_mask(gt_path)
        if cfg.skip_empty_gt and not (gt > cfg.mask_threshold).any():
            return stem, None, None, True
        pred = np.zeros_like(gt)
        if pred_path is not None:
            p = read_mask(pred_path)
            if p.shape != gt.shape:
                return stem, None, f"尺寸不一致 gt={gt.shape} pred={p.shape}", False
            pred = p
        m = evaluate(pred, gt, cfg)
        return stem, m, None, False
    except Exception as exc:  # 单张失败不应中断整批
        return stem, None, f"{type(exc).__name__}: {exc}", False


def collect(gt_dir: str, pred_dir: str, limit: int | None, gt_suffix: str = "", pred_suffix: str = ""):
    """gt_suffix/pred_suffix 用于真值目录里混有影像时只挑标注，例如 _mask.png。"""
    stems = []
    for root, _, files in os.walk(gt_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            if gt_suffix and not f.endswith(gt_suffix):
                continue
            bases = os.path.splitext(f)[0]
            stems.append(bases[: -len(os.path.splitext(gt_suffix)[0])] if gt_suffix else bases)
    stems = sorted(set(stems))
    if limit:
        stems = stems[:limit]

    def find(base: str, stem: str, suffix: str):
        cand = stem + suffix if suffix else None
        if cand:
            for ext in IMG_EXT:
                p = os.path.join(base, cand + ext)
                if os.path.exists(p):
                    return p
        for ext in IMG_EXT:
            p = os.path.join(base, stem + ext)
            if os.path.exists(p):
                return p
        # 允许预测放在子目录中（如 pred/test/xxx.png）
        for root, _, files in os.walk(base):
            for f in files:
                if os.path.splitext(f)[1].lower() not in IMG_EXT:
                    continue
                name = f[: -len(suffix)] if (suffix and f.endswith(suffix)) else os.path.splitext(f)[0]
                if name == stem:
                    return os.path.join(root, f)
        return None

    jobs = []
    for s in stems:
        gt_path = find(gt_dir, s, gt_suffix)
        if gt_path is None:
            continue
        jobs.append((s, gt_path, find(pred_dir, s, pred_suffix), None))
    return jobs


def summarize(rows):
    out = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows if np.isfinite(r[k])], dtype=float)
        if len(vals) == 0:
            continue
        out[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)),
                  "std": float(vals.std()), "n": int(len(vals))}
    return out


def markdown_table(summary):
    head = "| 指标 | 均值 | 中位数 | 标准差 | 张数 |"
    sep = "|---|---|---|---|---|"
    lines = [head, sep]
    for k in METRIC_KEYS:
        if k not in summary:
            continue
        s = summary[k]
        lines.append(f"| {METRIC_NAMES.get(k, k)} | {s['mean']:.4f} | {s['median']:.4f} | {s['std']:.4f} | {s['n']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="道路中心线/掩膜批量评测")
    ap.add_argument("--gt", required=True, help="真值目录")
    ap.add_argument("--pred", required=True, help="预测目录")
    ap.add_argument("--out", default="results", help="输出目录")
    ap.add_argument("--gsd", type=float, default=1.0, help="地面采样距离（米/像素）")
    ap.add_argument("--tol", type=float, default=3.0, help="中心线匹配容差（像素）")
    ap.add_argument("--samples", type=int, default=60, help="APLS 采样点对数")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=None, help="只评测前 N 张")
    ap.add_argument("--threshold", type=int, default=127, help="二值化阈值")
    ap.add_argument("--keep-empty", action="store_true",
                    help="保留真值为空（无道路）的切片。默认剔除，避免空图把均值抬高")
    ap.add_argument("--gt-suffix", default="",
                    help="真值文件名后缀，例如 _mask.png（目录里混有影像时必填）")
    ap.add_argument("--pred-suffix", default="",
                    help="预测文件名后缀，不填则按同名匹配")
    args = ap.parse_args(argv)

    cfg = EvalConfig(gsd=args.gsd, tol_px=args.tol, apls_samples=args.samples,
                     mask_threshold=args.threshold, skip_empty_gt=not args.keep_empty)
    jobs = collect(args.gt, args.pred, args.limit, args.gt_suffix, args.pred_suffix)
    if not jobs:
        print("没有找到可评测的样本，请检查 --gt / --pred 路径与文件名是否对应")
        return 1

    missing = sum(1 for j in jobs if j[2] is None)
    print(f"待评测 {len(jobs)} 张；其中 {missing} 张缺少预测（按全空计分）")

    payload = [(s, g, p, cfg) for s, g, p, _ in jobs]

    os.makedirs(args.out, exist_ok=True)
    rows, errors = [], []
    skipped = 0
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_work, j) for j in payload]
        for fut in as_completed(futures):
            stem, m, err, was_skipped = fut.result()
            done += 1
            if err:
                errors.append((stem, err))
            elif was_skipped:
                skipped += 1
            else:
                m["name"] = stem
                rows.append(m)
            if done % 200 == 0 or done == len(futures):
                print(f"  进度 {done}/{len(futures)}  用时 {time.time()-t0:.0f}s")

    if not rows:
        print("全部样本评测失败：")
        for s, e in errors[:10]:
            print("  -", s, e)
        return 1

    per_image = os.path.join(args.out, "per_image.csv")
    with open(per_image, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["name"] + METRIC_KEYS)
        for r in rows:
            w.writerow([r["name"]] + [f"{r[k]:.6f}" for k in METRIC_KEYS])

    summary = summarize(rows)
    summ_path = os.path.join(args.out, "summary.csv")
    with open(summ_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median", "std", "n"])
        for k, s in summary.items():
            w.writerow([METRIC_NAMES.get(k, k), f"{s['mean']:.6f}",
                        f"{s['median']:.6f}", f"{s['std']:.6f}", s["n"]])

    print()
    print(markdown_table(summary))
    print()
    print(f"计入汇总 {len(rows)} 张，剔除空标注 {skipped} 张，失败 {len(errors)} 张，耗时 {time.time()-t0:.0f}s")
    if skipped:
        print("  注：空标注切片已剔除（--keep-empty 可保留）。论文里应单独报告这类切片占比。")
    print(f"逐张结果：{per_image}")
    print(f"汇总结果：{summ_path}")
    if errors:
        print("失败样例（前 5 条）：")
        for s, e in errors[:5]:
            print("  -", s, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
