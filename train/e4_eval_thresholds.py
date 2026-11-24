# -*- coding: utf-8 -*-
"""路线② 收口复核：在多个**统一**线带阈值口径下重评全部训练期变体。

动机（2026-11-11 阈值诊断发现）：基线在 0.5 阈值下 APLS 0.3364，
降到 0.3 后 APLS 0.3731，说明 0.5 不是最优工作点。若只在一个阈值下比较，
可能把「约束在某个工作点有效」误判为无效。因此对所有变体用**完全相同**的
阈值重评，再跑同一套事前判据。

严格性：
  - 所有变体共享同一批 120 张 test、同一 GT、同一 EvalConfig；
  - 阈值对所有变体一视同仁，不由任何加约束变体的结果挑选；
  - `*_t050` 必须逐指标复现原 `eval/summary.csv`（阴性对照，阈值 1e-5）；
  - 判据仍是 tools/e4_gain_criteria.py 里写死的那套，不放宽。

用法：python train/e4_eval_thresholds.py
产物：runs/{name}_t{030,040,050}/eval/{per_image.csv,summary.csv,config.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "eval"))
from e4_consistency import (  # noqa: E402
    UNet3Head, E4Tile, load_mask, load_image, e4_metrics, seed_all,
)
from e4_postprocess import save_eval  # noqa: E402
from metrics import EvalConfig  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RUNS = ROOT / "runs"
VARIANTS = [
    "e4_v3_base", "e4_v3_gt2", "e4_lam002", "e4_lam005",
    "e4_lam010", "e4_lam020", "e4_lam100", "e4_excl_only",
]
THRS = (0.30, 0.40, 0.50)


def load_sum(p: Path) -> dict:
    d = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                d[r["metric"]] = float(r["mean"])
            except ValueError:
                continue
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=176)
    ap.add_argument("--test-limit", type=int, default=120)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    seed_all(7)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = E4Tile("test", args.size, args.test_limit, False, line_dilate=0, cache=False)
    cfg = EvalConfig(gsd=1.0, tol_px=2.0, apls_samples=40, seed=0)
    t_all = time.time()

    for name in VARIANTS:
        ck = RUNS / name / "last.pt"
        ckpt = torch.load(ck, map_location=dev, weights_only=False)
        model = UNet3Head(ckpt["args"]["base"]).to(dev)
        model.load_state_dict(ckpt["model"])
        model.eval()
        cache, t0 = [], time.time()
        with torch.no_grad():
            for r in ds.rows:
                x = torch.from_numpy(load_image(r["image"], args.size))[None].to(dev)
                p = torch.sigmoid(model(x))[0].cpu().numpy()
                gt = tuple(load_mask(r[k], args.size) > 0
                           for k in ("building", "roadarea", "roadline"))
                cache.append((r["id"], p, gt))
        for thr in THRS:
            out_name = f"{name}_t{int(round(thr*100)):03d}"
            rows = []
            for rid, p, (gb, gr, gl) in cache:
                pb, pr, pl = p[0] > 0.5, p[1] > 0.5, p[2] > thr
                rows.append({"id": rid,
                             **e4_metrics(pb, pr, pl, gb, gr, gl, args.radius, cfg)})
            cfg_out = {
                "name": out_name, "kind": "threshold_recheck", "source": name,
                "line_threshold": thr, "size": args.size,
                "test_limit": args.test_limit, "radius": args.radius, "seed": 7,
                "epoch": ckpt["epoch"], "n_images": len(rows),
                "init": ckpt["args"].get("init"),
                "lam": {"excl": ckpt["args"].get("lam_excl"),
                        "dist": ckpt["args"].get("lam_dist"),
                        "linein": ckpt["args"].get("lam_linein")},
                "elapsed_s": round(time.time() - t0, 1),
            }
            save_eval(out_name, rows, cfg_out, RUNS / out_name / "pred")
            apls = float(np.mean([r["line_apls"] for r in rows]))
            lf1 = float(np.mean([r["line_line_f1"] for r in rows]))
            print(f"  {out_name:24s} APLS={apls:.4f} 线F1={lf1:.4f}")
        print(f"  {name} 完成，耗时 {time.time()-t0:.1f}s")

    # 阴性对照：t050 必须逐指标复现原口径，否则整轮复核不可信
    worst_all = 0.0
    for name in VARIANTS:
        a = load_sum(RUNS / f"{name}_t050" / "eval" / "summary.csv")
        b = load_sum(RUNS / name / "eval" / "summary.csv")
        d = max(abs(a[k] - b[k]) for k in a if k in b)
        worst_all = max(worst_all, d)
    print(f"t050 阴性对照最大逐指标差异 = {worst_all:.8f}（应≈0）")
    if worst_all > 1e-5:
        print("!! 口径不一致，阈值复核结果不可用于判据")
        return 2
    print(f"阈值口径复核完成，总耗时 {time.time()-t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
