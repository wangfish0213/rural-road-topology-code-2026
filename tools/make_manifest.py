# -*- coding: utf-8 -*-
"""生成数据集索引清单（manifest）：id / 影像路径 / 标注路径 / 划分。

训练与评测都只读这份 CSV，避免每次递归扫目录，也便于记录"这一版实验用的是哪份划分"。

用法：
    python make_manifest.py --root ../datasets --out ../manifests
    # 生成 manifests/train.csv、manifests/test.csv、manifests/all.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def scan(root: str, img_suffix: str, mask_suffix: str, split_name: str):
    rows = []
    for dirpath, _, files in os.walk(root):
        imgs = {f[: -len(img_suffix)]: f for f in files if f.endswith(img_suffix)}
        masks = {f[: -len(mask_suffix)]: f for f in files if f.endswith(mask_suffix)}
        for key in sorted(set(imgs) & set(masks)):
            rows.append({
                "id": key,
                "split": split_name,
                "image": os.path.abspath(os.path.join(dirpath, imgs[key])),
                "mask": os.path.abspath(os.path.join(dirpath, masks[key])),
            })
    return rows


def scan_paired_dirs(images_dir: str, masks_dir: str, split_name: str):
    """影像与标注分两个目录、按同名配对的布局（WHU Road 就是这种）。"""
    def listing(d):
        return {os.path.splitext(f)[0]: f for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() in IMG_EXT}
    imgs, masks = listing(images_dir), listing(masks_dir)
    rows = []
    for key in sorted(set(imgs) & set(masks)):
        rows.append({
            "id": key,
            "split": split_name,
            "image": os.path.abspath(os.path.join(images_dir, imgs[key])),
            "mask": os.path.abspath(os.path.join(masks_dir, masks[key])),
        })
    return rows, len(set(imgs) - set(masks)), len(set(masks) - set(imgs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="数据集根目录，其下按 train/ test/ 分子目录")
    ap.add_argument("--images-dir", help="影像目录（与 --masks-dir 配对使用）")
    ap.add_argument("--masks-dir", help="标注目录")
    ap.add_argument("--split", default="train", help="分目录模式下的划分名")
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--images-suffix", default="_sat.jpg", help="同目录布局下的影像后缀")
    ap.add_argument("--masks-suffix", default="_mask.png", help="同目录布局下的标注后缀")
    args = ap.parse_args(argv)

    # 模式一：影像/标注分目录，同名配对
    if args.images_dir and args.masks_dir:
        rows, only_img, only_mask = scan_paired_dirs(args.images_dir, args.masks_dir, args.split)
        if not rows:
            print("两个目录里没有同名文件，检查路径是否正确")
            return 1
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{args.split}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["id", "split", "image", "mask"])
            w.writeheader()
            w.writerows(rows)
        print(f"{args.split:5s} {len(rows):6d} 条 -> {path}"
              f"（仅影像无标注 {only_img}，仅标注无影像 {only_mask}）")
        return 0

    if not args.root:
        print("请提供 --root，或同时提供 --images-dir 与 --masks-dir")
        return 1

    all_rows = []
    for split in ("train", "test", "val"):
        d = os.path.join(args.root, split)
        if not os.path.isdir(d):
            continue
        rows = scan(d, args.images_suffix, args.masks_suffix, split)
        if not rows:
            continue
        all_rows.extend(rows)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{split}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["id", "split", "image", "mask"])
            w.writeheader()
            w.writerows(rows)
        print(f"{split:5s} {len(rows):6d} 条 -> {path}")

    if not all_rows:
        print("没有找到任何配对样本")
        return 1

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "all.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "image", "mask"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"{'all':5s} {len(all_rows):6d} 条 -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
