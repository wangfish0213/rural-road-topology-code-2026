# -*- coding: utf-8 -*-
"""数据集体检：配对完整性、掩膜编码、前景占比、连通分量、骨架长度。

用途
----
1. 确认影像与标注是否一一配对（少了哪张一眼能看见）；
2. 确认掩膜是 0/255 还是 0/1、有没有灰度杂值；
3. 统计道路像素占比与连通分量——这两个数字决定损失函数要不要加权、
   以及为什么像素准确率在这类数据上不可信（前景常低于 2%）。

用法
----
    python dataset_stats.py --root ../datasets --images-suffix _sat.jpg --masks-suffix _mask.png

    # 只抽 500 对快速看分布
    python dataset_stats.py --root ../datasets --limit 500
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from metrics import skeletonize  # noqa: E402  复用同一套骨架化实现，避免两处口径不一致

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def scan(root: str, img_suffix: str, mask_suffix: str, limit: int | None):
    """按"文件名去掉后缀"配对，例如 00001_sat.jpg 与 00001_mask.png → 键 00001。"""
    pairs, orphan_masks, orphan_imgs = [], [], []
    for dirpath, _, files in os.walk(root):
        imgs, masks = {}, {}
        for f in files:
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            if f.endswith(img_suffix):
                imgs[f[: -len(img_suffix)]] = f
            elif f.endswith(mask_suffix):
                masks[f[: -len(mask_suffix)]] = f
        for key in sorted(set(imgs) | set(masks)):
            if key in imgs and key in masks:
                pairs.append((os.path.join(dirpath, imgs[key]), os.path.join(dirpath, masks[key])))
            elif key in masks:
                orphan_masks.append(os.path.join(dirpath, masks[key]))
            else:
                orphan_imgs.append(os.path.join(dirpath, imgs[key]))
    pairs.sort()
    if limit:
        pairs = pairs[:limit]
    return pairs, orphan_masks, orphan_imgs


def scan_paired_dirs(images_dir: str, masks_dir: str, limit: int | None):
    """影像与标注分两目录、同名配对（WHU Road 布局）。"""
    def listing(d):
        return {os.path.splitext(f)[0]: f for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() in IMG_EXT}
    imgs, masks = listing(images_dir), listing(masks_dir)
    pairs = [(os.path.join(images_dir, imgs[k]), os.path.join(masks_dir, masks[k]))
             for k in sorted(set(imgs) & set(masks))]
    orphan_masks = [os.path.join(masks_dir, masks[k]) for k in sorted(set(masks) - set(imgs))]
    orphan_imgs = [os.path.join(images_dir, imgs[k]) for k in sorted(set(imgs) - set(masks))]
    if limit:
        pairs = pairs[:limit]
    return pairs, orphan_masks, orphan_imgs


def inspect(job):
    img_path, mask_path, with_skel = job
    try:
        with Image.open(mask_path) as mi:
            mask = np.array(mi)
            mode, size = mi.mode, mi.size
        with Image.open(img_path) as ii:
            isize = ii.size
        if mask.ndim == 3:
            mask = mask[..., 0]
        values = np.unique(mask)
        binary = mask > 0
        fg = float(binary.mean())
        _, ncomp = ndi.label(binary, structure=np.ones((3, 3), int))
        if with_skel and binary.any():
            skel_px = int(skeletonize(binary).sum())
        else:
            skel_px = -1
        return {
            "name": os.path.basename(img_path),
            "img_w": isize[0], "img_h": isize[1],
            "mask_mode": mode, "mask_size_match": int(isize == size),
            "n_unique": int(len(values)),
            "values": ",".join(str(int(v)) for v in values[:4]) if len(values) <= 4 else "多值",
            "fg_ratio": fg,
            "components": int(ncomp),
            "skel_px": skel_px,
            "empty": int(not binary.any()),
        }
    except Exception as exc:
        return {"name": os.path.basename(img_path), "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="遥感分割数据集体检")
    ap.add_argument("--root", help="数据集目录（同目录 / 同名后缀布局）")
    ap.add_argument("--images-dir", help="影像目录（与 --masks-dir 配对使用，适用于 WHU Road 布局）")
    ap.add_argument("--masks-dir", help="标注目录")
    ap.add_argument("--images-suffix", default="_sat.jpg")
    ap.add_argument("--masks-suffix", default="_mask.png")
    ap.add_argument("--out", default="dataset_reports")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with-skeleton", action="store_true",
                    help="额外计算每张骨架长度（1024 掩膜约 1.7s/张，全量会慢很多，建议抽样用）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args(argv)

    if args.images_dir and args.masks_dir:
        pairs, orphan_masks, orphan_imgs = scan_paired_dirs(args.images_dir, args.masks_dir, args.limit)
    else:
        if not args.root:
            print("请提供 --root，或同时提供 --images-dir 与 --masks-dir")
            return 1
        pairs, orphan_masks, orphan_imgs = scan(args.root, args.images_suffix, args.masks_suffix, args.limit)
    print(f"配对成功 {len(pairs)} 对 | 只有标注 {len(orphan_masks)} | 只有影像 {len(orphan_imgs)}")
    if not pairs:
        print("未找到配对样本，请检查 --images-suffix / --masks-suffix")
        return 1

    jobs = [(a, b, args.with_skeleton) for a, b in pairs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(inspect, jobs, chunksize=32))

    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "per_tile.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fg = np.array([r["fg_ratio"] for r in rows])
    comp = np.array([r["components"] for r in rows])
    skel = np.array([r["skel_px"] for r in rows if r["skel_px"] >= 0])
    empty = int(sum(r["empty"] for r in rows))
    sizes = Counter((r["img_w"], r["img_h"]) for r in rows)
    modes = Counter(r["mask_mode"] for r in rows)
    uniq = Counter(r["values"] for r in rows)
    mismatch = sum(1 for r in rows if not r["mask_size_match"])

    lines = [
        "# 数据集体检报告",
        "",
        f"- 样本数：{len(rows)}" + (f"（另有 {len(bad)} 张读取失败）" if bad else ""),
        f"- 影像尺寸分布：{dict(sizes.most_common(5))}",
        f"- 标注模式：{dict(modes)}；尺寸不一致 {mismatch} 张",
        f"- 标注取值：{dict(uniq.most_common(5))}",
        f"- 空标注（无前景目标）：{empty} 张，占 {empty/len(rows)*100:.2f}%",
        "",
        "## 前景像素占比",
        "",
        f"- 均值 {fg.mean()*100:.2f}% ｜ 中位数 {np.median(fg)*100:.2f}%",
        f"- P10 {np.percentile(fg,10)*100:.2f}% ｜ P90 {np.percentile(fg,90)*100:.2f}% ｜ 最大 {fg.max()*100:.2f}%",
        "",
        "## 拓扑规模（每张切片内）",
        "",
        f"- 连通分量：均值 {comp.mean():.1f} ｜ 中位数 {np.median(comp):.0f} ｜ 最大 {comp.max()}",
    ]
    if len(skel):
        lines.append(f"- 骨架长度：均值 {skel.mean():.0f} px ｜ 中位数 {np.median(skel):.0f} px（{len(skel)} 张）")
    else:
        lines.append("- 骨架长度：本次未计算（加 --with-skeleton 才算，1024 掩膜约 1.7s/张）")
    report = "\n".join(lines)
    md_path = os.path.join(args.out, "stats_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print()
    print(report)
    print()
    print(f"逐张明细：{csv_path}")
    print(f"汇总报告：{md_path}")
    if bad:
        print("读取失败样例：", bad[:3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
