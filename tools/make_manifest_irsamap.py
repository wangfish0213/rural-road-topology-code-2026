# -*- coding: utf-8 -*-
"""为 IRSAMap 生成索引清单（多图层版）。

IRSAMap 与其它数据集不同：它的标注是**按图层分目录**的，同一景可以同时有
building / roadarea / roadline / vege / water / sport 若干层的栅格标签，
缺哪层就说明该景没有该类地物（不是漏标，是本来就没有）。

所以清单里每一层是"有则给路径、无则留空"，下游按需取用。

用法：
    python make_manifest_irsamap.py --root ../datasets/IRSAMap/IRSAMap_V2 --out ../manifests
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# 有栅格标签的图层（roadline 只提供 GeoJSON 矢量，没有栅格）
LAYERS = ["building", "roadarea", "vege", "water", "sport"]
VEC_LAYERS = ["building", "roadarea", "roadline", "vege", "water", "sport"]


def collect_split(split_dir: str, split: str, require: list[str]):
    img_dir = os.path.join(split_dir, "image")
    if not os.path.isdir(img_dir):
        return [], {}
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(".png")}

    layer_files, vec_files = {}, {}
    for cls in LAYERS:
        d = os.path.join(split_dir, f"label_{cls}")
        if os.path.isdir(d):
            layer_files[cls] = {os.path.splitext(f)[0]: f for f in os.listdir(d)
                                if f.lower().endswith(".png")}
    for cls in VEC_LAYERS:
        d = os.path.join(split_dir, f"label_geojson_{cls}")
        if os.path.isdir(d):
            vec_files[cls] = {os.path.splitext(f)[0]: f for f in os.listdir(d)
                              if f.lower().endswith(".geojson")}

    rows = []
    for key in sorted(imgs):
        # require 里的图层既可以是栅格层也可以是矢量层
        ok = True
        for c in require:
            if c in layer_files:
                if key not in layer_files[c]:
                    ok = False; break
            elif c in vec_files:
                if key not in vec_files[c]:
                    ok = False; break
            else:
                ok = False; break
        if not ok:
            continue
        row = {"id": key, "split": split, "image": os.path.abspath(os.path.join(img_dir, imgs[key]))}
        for cls in LAYERS:
            f = layer_files.get(cls, {}).get(key)
            row[cls] = os.path.abspath(os.path.join(split_dir, f"label_{cls}", f)) if f else ""
        for cls in VEC_LAYERS:
            f = vec_files.get(cls, {}).get(key)
            row[f"{cls}_geojson"] = os.path.abspath(
                os.path.join(split_dir, f"label_geojson_{cls}", f)) if f else ""
        rows.append(row)
    stats = {cls: sum(1 for r in rows if r[cls]) for cls in LAYERS}
    stats.update({f"{cls}(vec)": sum(1 for r in rows if r[f"{cls}_geojson"]) for cls in VEC_LAYERS})
    return rows, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="IRSAMap 根目录（其下有 train/val/test）")
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--require", default="building,roadarea",
                    help="只保留同时具备这些图层的景（逗号分隔），默认建筑+路面")
    args = ap.parse_args(argv)

    require = [c.strip() for c in args.require.split(",") if c.strip()]
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for split in ("train", "val", "test"):
        d = os.path.join(args.root, split)
        if not os.path.isdir(d):
            continue
        rows, stats = collect_split(d, split, require)
        if not rows:
            print(f"{split}: 无满足条件的样本")
            continue
        path = os.path.join(args.out, f"irsamap_{split}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            fields = ["id", "split", "image"] + LAYERS + [f"{c}_geojson" for c in VEC_LAYERS]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        total += len(rows)
        detail = "，".join(f"{k} {v}" for k, v in stats.items() if v)
        print(f"{split:5s} {len(rows):5d} 条（要求 {require}）→ {path}\n        各层非空：{detail}")
    print(f"合计 {total} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
