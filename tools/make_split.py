# -*- coding: utf-8 -*-
"""把一份 manifest 切成 train/val（按 id 哈希，可复现，换机重跑结果一致）。

老实说清楚局限：WHU Road 的文件名是随机串（如 01lfZ.png）、又没有地理坐标，
因此**做不到按区域划分**；哈希切分只保证"可复现、无人工挑选"，不解决空间相邻泄漏。
论文里必须写明这一点，别让审稿人替我们发现。

用法：
    python make_split.py --manifest ../manifests/whu_road_1m.csv --ratio 0.85 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys


def bucket(key: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ratio", type=float, default=0.85, help="训练集比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default=None, help="输出文件名前缀，默认取输入文件名")
    args = ap.parse_args(argv)

    with open(args.manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("清单为空")
        return 1

    train, val = [], []
    for r in rows:
        (train if bucket(r["id"], args.seed) < args.ratio else val).append(r)

    base = args.prefix or os.path.splitext(os.path.basename(args.manifest))[0]
    out_dir = os.path.dirname(os.path.abspath(args.manifest))
    for name, subset in ((f"{base}_train", train), (f"{base}_val", val)):
        path = os.path.join(out_dir, name + ".csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["id", "split", "image", "mask"])
            w.writeheader()
            w.writerows(subset)
        print(f"{name:32s} {len(subset):6d} 条 -> {path}")
    print(f"训练/验证 = {len(train)}/{len(val)}（ratio={args.ratio}, seed={args.seed}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
