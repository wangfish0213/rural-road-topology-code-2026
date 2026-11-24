# -*- coding: utf-8 -*-
"""创新点 3 / E4 立论前提：真值层面「线—面一致性」是否成立。

在动模型之前必须先回答：IRSAMap 的标注本身是否满足
  ① 建筑面与道路面互斥；
  ② 建筑边界不落在道路中心线邻域内；
  ③ 道路中心线应落在道路面内。
若真值本身大面积违反，则软约束没有依据，必须改写创新点 3。

所有数字由本脚本从掩膜直接统计，输出到 runs/e4_prep/gt_consistency.{json,md}。
用法：python tools/e4_gt_consistency.py --limit 200 [--split train,val,test]
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
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "e4_prep"
MANIFESTS = ROOT / "manifests"

sys.stdout.reconfigure(encoding="utf-8")


def load_bin(path: str, resize: int | None = None) -> np.ndarray:
    """IRSAMap 栅格标签用**类别值**（building=31、roadarea=32），不是 0/255。

    因此必须 >0 二值化；若沿用 eval/metrics.py 的默认 127 阈值会把整层判成背景。
    roadline 由 tools/irsamap_prep.py 生成，是 0/255，同样兼容。
    """
    im = Image.open(path).convert("L")
    if resize:
        im = im.resize((resize, resize), Image.NEAREST)
    return (np.asarray(im) > 0).astype(bool)


def boundary(mask: np.ndarray) -> np.ndarray:
    """单像素内边界：原图减去 3x3 腐蚀。"""
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))


def per_image(r: dict, resize: int, radius: int) -> dict:
    b = load_bin(r["building"], resize)
    ra = load_bin(r["roadarea"], resize)
    ln = load_bin(r["roadline"], resize)
    n_b, n_ra, n_ln = b.sum(), ra.sum(), ln.sum()

    inter = (b & ra).sum()
    # 建筑边界落在中心线 radius px 邻域内的比例
    bd = boundary(b)
    near_line = ndi.distance_transform_edt(~ln) <= radius
    bd_violation = (bd & near_line).sum()
    # 中心线落在建筑膨胀带内的比例（另一视角）
    near_b = ndi.distance_transform_edt(~b) <= radius
    return {
        "building_px": int(n_b), "roadarea_px": int(n_ra), "line_px": int(n_ln),
        "overlap_px": int(inter),
        "overlap_over_roadarea": float(inter / max(1, n_ra)),
        "overlap_over_union": float(inter / max(1, n_b + n_ra - inter)),
        "boundary_px": int(bd.sum()),
        "boundary_line_hit_px": int(bd_violation),
        "boundary_line_violation": float(bd_violation / max(1, bd.sum())),
        "line_in_roadarea": float((ln & ra).sum() / max(1, n_ln)),
        "line_near_building": float((ln & near_b).sum() / max(1, n_ln)),
    }


def summarise(rows: list[dict]) -> dict:
    keys = ["overlap_over_roadarea", "overlap_over_union", "boundary_line_violation",
            "line_in_roadarea", "line_near_building"]
    out = {"n_images": len(rows)}
    for k in keys:
        v = np.asarray([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)), "max": float(v.max())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train,val,test")
    ap.add_argument("--limit", type=int, default=200, help="每个 split 最多统计 N 张，0=全量")
    ap.add_argument("--resize", type=int, default=512, help="统计分辨率（缩半以加速，比例量不变）")
    ap.add_argument("--radius", type=int, default=3, help="边界—中心线邻域半径（像素，按 resize 后尺度）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result, lines = {}, ["# E4 立论前提：IRSAMap 真值的线—面一致性统计", "",
                         f"> 由 `tools/e4_gt_consistency.py` 生成；分辨率 {args.resize}×{args.resize}，"
                         f"邻域半径 {args.radius} px；数字不得人工转抄。", ""]
    for sp in args.split.split(","):
        src = MANIFESTS / f"irsamap_e4_{sp}.csv"
        if not src.is_file():
            print(f"{sp}: 缺 {src}")
            continue
        rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
        if args.limit:
            rows = rows[:args.limit]
        per, bad = [], 0
        for r in rows:
            try:
                per.append(per_image(r, args.resize, args.radius))
            except Exception as e:
                bad += 1
                print(f"  ! {sp}/{r['id']}: {e}")
        s = summarise(per)
        s["n_failed"] = bad
        result[sp] = s
        lines += [f"## {sp}（{s['n_images']} 张，失败 {bad}）", "",
                  "| 量 | 均值 | 中位 | P90 | 最大 |", "|---|---|---|---|---|"]
        names = {
            "overlap_over_roadarea": "GT 建筑∩路面 / 路面像素（互斥违规率）",
            "overlap_over_union": "GT 建筑∩路面 / 并集",
            "boundary_line_violation": "GT 建筑边界落在中心线 3px 邻域的比例",
            "line_in_roadarea": "GT 中心线落在路面内的比例",
            "line_near_building": "GT 中心线落在建筑 3px 邻域的比例",
        }
        for k, name in names.items():
            v = s[k]
            lines.append(f"| {name} | {v['mean']:.4f} | {v['median']:.4f} | "
                         f"{v['p90']:.4f} | {v['max']:.4f} |")
        lines.append("")
        print(f"{sp}: {s['n_images']} 张，互斥违规率均值 "
              f"{s['overlap_over_roadarea']['mean']:.4f}，"
              f"边界-中心线违规 {s['boundary_line_violation']['mean']:.4f}，"
              f"中心线在路面内 {s['line_in_roadarea']['mean']:.4f}")

    lines += ["## 判据（写进论文前须由王老师确认措辞）", "",
              "- 前两项越接近 0，说明「建筑/路面互斥」与「建筑边界远离中心线」在真值中确实成立，"
              "软约束才是有依据的先验；",
              "- 第三项（中心线在路面内）越高，说明中心线头可由路面头提供结构先验；",
              "- 以上均为**真值**统计，不是模型结果；模型侧的同类指标见 `runs/e4_*/eval/`。", ""]
    with open(out / "gt_consistency.json", "w", encoding="utf-8") as f:
        json.dump({"generated_by": "tools/e4_gt_consistency.py",
                   "resize": args.resize, "radius": args.radius,
                   "elapsed_s": round(time.time() - t0, 1),
                   "splits": result}, f, ensure_ascii=False, indent=2)
    with open(out / "gt_consistency.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"输出：{out/'gt_consistency.json'} / {out/'gt_consistency.md'}，耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
