# -*- coding: utf-8 -*-
"""路线② 后半：推理期线—面一致性后处理（不改动训练权重）。

立论：tools/e4_gt_consistency.py 实测真值中 99.3%~99.99% 的道路中心线落在
道路面内、建筑与路面几乎不重叠。因此可在**推理期**用另一任务的预测概率
对当前任务的预测做门控/抑制，而不依赖训练期真值、也不增加训练成本。

与训练期正则的关键区别：后处理直接改变输出掩膜，可测 IoU / APLS 是否真涨。

严格可比性（写死，不得绕过）：
  - 权重固定为已收敛基线 runs/e4_v3_base/last.pt（公共初始化微调 6 轮，λ=0）；
  - 同一批 120 张 test、同一 GT、同一 EvalConfig(gsd=1.0, tol_px=2.0,
    apls_samples=40, seed=0)、同一个 e4_metrics 函数；
  - pp_baseline 不做任何变换，用来验证重算链路与既有 summary.csv 逐指标一致；
  - _matched 变体把线带前景像素数强制恢复到基线（逐图分位数阈值），
    使 Δ 只反映「像素选得更准」，排除「预测得更少 → precision 虚涨」。

用法：python train/e4_postprocess.py [--size 176] [--test-limit 120]
产物：runs/pp_*/eval/{per_image.csv,summary.csv,config.json}, runs/pp_*/pred/*.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "eval"))
from e4_consistency import (  # noqa: E402
    UNet3Head, E4Tile, load_mask, load_image, e4_metrics, seed_all,
)
from metrics import EvalConfig  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RUNS = ROOT / "runs"
BASE = "e4_v3_base"

# 变换作用在 sigmoid 概率图上；返回 (pb', pr', pl_band')。
VARIANTS = {
    "pp_baseline": None,
    # 真值先验「线在路面内」：用路面概率给中心线加权
    "pp_line_gate_soft": lambda pb, pr, pl: (pb, pr, pl * (0.5 + 0.5 * pr)),
    "pp_line_gate_hard": lambda pb, pr, pl: (pb, pr, pl * (pr > 0.5)),
    # 真值先验「建筑与路面互斥」：互相抑制
    "pp_excl_b": lambda pb, pr, pl: (pb * (1 - pr), pr, pl),
    "pp_excl_r": lambda pb, pr, pl: (pb, pr * (1 - pb), pl),
    # 组合：三项同时用**原始**概率计算，避免链式依赖
    "pp_combo": lambda pb, pr, pl: (pb * (1 - pr), pr * (1 - pb),
                                    pl * (0.5 + 0.5 * pr)),
}
# 这些变体额外产出一个覆盖率保持版；源变体必须是改了线带的
MATCHED_FROM = {
    "pp_line_gate_soft_matched": "pp_line_gate_soft",
    "pp_line_gate_hard_matched": "pp_line_gate_hard",
    "pp_combo_matched": "pp_combo",
}
FORMULAS = {
    "pp_baseline": "无变换（阴性对照，应逐指标等于 e4_v3_base）",
    "pp_line_gate_soft": "pl' = pl * (0.5 + 0.5*pr)",
    "pp_line_gate_hard": "pl' = pl * (pr > 0.5)",
    "pp_excl_b": "pb' = pb * (1 - pr)",
    "pp_excl_r": "pr' = pr * (1 - pb)",
    "pp_combo": "pb'=pb*(1-pr), pr'=pr*(1-pb), pl'=pl*(0.5+0.5*pr)（同用原始概率）",
}
# 纯阈值敏感性诊断：不改概率、只改线带二值化阈值。
# 用途：排除「增益其实只是阈值没调好」的质疑；**不计入判据候选**。
THR_FROM = {
    "pp_thr030": 0.30, "pp_thr040": 0.40,
    "pp_thr060": 0.60, "pp_thr070": 0.70,
}


def binarize(pb, pr, pl, base_cov_line=None, matched=False, line_thr=None):
    """fixed: 三头统一 >0.5；matched: 线带按基线覆盖率取逐图分位数阈值；
    line_thr: 仅诊断用，线带改用指定固定阈值。"""
    bb, br = pb > 0.5, pr > 0.5
    if matched and base_cov_line is not None:
        # 精确 top-k：保证前景像素数与基线严格相等，分位数阈值在并列值上会漂移
        flat = pl.ravel()
        k = max(1, min(flat.size, int(round(base_cov_line * flat.size))))
        idx = np.argpartition(flat, flat.size - k)[flat.size - k:]
        bl = np.zeros(flat.size, dtype=bool)
        bl[idx] = True
        bl = bl.reshape(pl.shape)
    elif line_thr is not None:
        bl = pl > line_thr
    else:
        bl = pl > 0.5
    return bb, br, bl


def save_eval(name: str, rows: list, cfg: dict, pred_dir: Path):
    out = RUNS / name
    (out / "eval").mkdir(parents=True, exist_ok=True)
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
    with open(out / "eval" / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return out / "eval" / "summary.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=176)
    ap.add_argument("--test-limit", type=int, default=120)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    seed_all(7)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ck = RUNS / args.base / "last.pt"
    ckpt = torch.load(ck, map_location=dev, weights_only=False)
    model = UNet3Head(ckpt["args"]["base"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"权重 {ck}（epoch {ckpt['epoch']}），设备 {dev}")

    ds = E4Tile("test", args.size, args.test_limit, False, line_dilate=0, cache=False)
    cfg = EvalConfig(gsd=1.0, tol_px=2.0, apls_samples=40, seed=0)
    cache, t0 = [], time.time()
    with torch.no_grad():
        for i, r in enumerate(ds.rows):
            x = torch.from_numpy(load_image(r["image"], args.size))[None].to(dev)
            p = torch.sigmoid(model(x))[0].cpu().numpy()
            gt = tuple(load_mask(r[k], args.size) > 0
                       for k in ("building", "roadarea", "roadline"))
            cache.append((r["id"], p, gt))
            if (i + 1) % 40 == 0:
                print(f"  前向 {i+1}/{len(ds.rows)}，{time.time()-t0:.0f}s")
    print(f"概率图缓存完成：{len(cache)} 张，耗时 {time.time()-t0:.1f}s")

    # 每张图的基线线带覆盖率，供 matched 版逐图恢复
    base_cov = {rid: float((p[2] > 0.5).mean()) for rid, p, _ in cache}

    all_names = list(VARIANTS) + list(MATCHED_FROM) + list(THR_FROM)
    for name in all_names:
        matched = name.endswith("_matched")
        src = MATCHED_FROM.get(name, name)
        thr = THR_FROM.get(name)
        fn = VARIANTS["pp_baseline"] if thr is not None else VARIANTS[src]
        rows, pred_dir = [], RUNS / name / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        for rid, p, (gb, gr, gl) in cache:
            pb, pr, pl = p[0], p[1], p[2]
            if fn is not None:
                pb, pr, pl = fn(pb, pr, pl)
            bb, br, bl = binarize(pb, pr, pl,
                                  base_cov_line=base_cov[rid] if matched else None,
                                  matched=matched, line_thr=thr)
            m = {"id": rid, **e4_metrics(bb, br, bl, gb, gr, gl, args.radius, cfg)}
            rows.append(m)
            if len(rows) <= 20:
                Image.fromarray(np.stack([bb, br, bl], -1).astype(np.uint8) * 255).save(
                    pred_dir / f"{rid}.png")
        cfg_out = {
            "name": name,
            "kind": "threshold_diagnostic" if thr is not None else "inference_postprocess",
            "source_variant": src,
            "formula": f"线带阈值 {thr}" if thr is not None else FORMULAS.get(src, ""),
            "line_threshold": thr,
            "matched_coverage": matched, "base_weights": str(ck),
            "size": args.size, "test_limit": args.test_limit, "radius": args.radius,
            "seed": 7, "epoch": ckpt["epoch"], "n_images": len(rows),
            "lam": {"excl": None, "dist": None, "linein": None},
            "init": str(ck), "elapsed_s": round(time.time() - t1, 1),
        }
        save_eval(name, rows, cfg_out, pred_dir)
        apls = np.mean([r["line_apls"] for r in rows])
        biou = np.mean([r["building_iou"] for r in rows])
        cov = np.mean([r["pred_line_coverage"] for r in rows])
        print(f"  {name:30s} APLS={apls:.4f} 建筑IoU={biou:.4f} "
              f"线带覆盖={cov:.4f} ({time.time()-t1:.1f}s)")

    # 阴性对照：pp_baseline 必须逐指标复现 e4_v3_base，否则整条链路口径不一致
    def load_sum(p):
        d = {}
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                d[r["metric"]] = float(r["mean"])
        return d
    a = load_sum(RUNS / "pp_baseline" / "eval" / "summary.csv")
    b = load_sum(RUNS / args.base / "eval" / "summary.csv")
    diffs = {k: abs(a[k] - b[k]) for k in a if k in b}
    worst = max(diffs.values()) if diffs else float("nan")
    print(f"阴性对照最大逐指标差异 = {worst:.8f}（应≈0）")
    if worst > 1e-5:
        print("!! 口径不一致，后处理结果不可用于判据")
        return 2

    diag = ["# 推理期一致性后处理：全部变体与阈值敏感性诊断", "",
            "> 由 `train/e4_postprocess.py` 生成，数字只读各 `eval/summary.csv`。", "",
            "> `pp_thr*` 只改线带二值化阈值，属诊断，**不计入增益判据候选**；"
            "它用于回答「基线是否只是阈值没调好」。", "",
            "| 变体 | 线带阈值 | 覆盖率保持 | APLS | 线F1 | 建筑IoU | 路面IoU | 线带覆盖率 |",
            "|---|---|---|---|---|---|---|---|"]
    for name in all_names:
        s = load_sum(RUNS / name / "eval" / "summary.csv")
        with open(RUNS / name / "eval" / "config.json", encoding="utf-8") as f:
            c = json.load(f)
        thr_s = c.get("line_threshold")
        diag.append(f"| {name} | {thr_s if thr_s is not None else 0.5} | "
                    f"{'是' if name.endswith('_matched') else '否'} | "
                    f"{s['line_apls']:.4f} | {s['line_line_f1']:.4f} | "
                    f"{s['building_iou']:.4f} | {s['roadarea_iou']:.4f} | "
                    f"{s['pred_line_coverage']:.4f} |")
    diag += ["", "## 阴性对照", "",
             f"`pp_baseline`（不做任何变换）与 `e4_v3_base` 逐指标最大差异 "
             f"**{worst:.8f}**（容许阈值 1e-5），说明后处理链路与原评测严格同口径。", ""]
    with open(RUNS / "e4_postprocess_diagnostics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(diag))
    print(f"诊断表已写 {RUNS/'e4_postprocess_diagnostics.md'}")
    print("后处理全部完成，产物已写盘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
