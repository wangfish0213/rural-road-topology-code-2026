# -*- coding: utf-8 -*-
"""产出回读校验器（保险①）：确认脚本声称生成的文件真的落盘且内容完整。

用法：
    python verify_outputs.py --files a.png,b.csv,c.json --out runs/verify.txt
    python verify_outputs.py --spec runs/verify_spec.json --out runs/verify.txt

校验规则（按扩展名）：
  - 图片：PIL 打开并 load()，尺寸 >= 16px，像素不能是纯色（防"保存了空图"）
  - CSV ：能被 csv 读取，至少 1 行数据
  - JSON：能被 json.load，且非空
  - 文本：UTF-8 可解码且非空
  - 其他：文件存在且 size > 0

退出码 0 = 全部通过；1 = 有失败项（报告里逐条列出）。
注意：必须用**另一个进程**调用本脚本，不能由生成文件的那个脚本自己校验自己。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")


def check(path: str) -> tuple[bool, str]:
    if not os.path.isfile(path):
        return False, "文件不存在"
    size = os.path.getsize(path)
    if size <= 0:
        return False, "文件大小为 0"
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            from PIL import Image, ImageStat
            with Image.open(path) as im:
                im.load()
                w, h = im.size
                if min(w, h) < 16:
                    return False, f"尺寸过小 {w}x{h}"
                stat = ImageStat.Stat(im.convert("RGB"))
                if max(stat.stddev) < 0.5:
                    return False, f"纯色图（stddev={max(stat.stddev):.3f}），疑似空图"
                return True, f"OK {w}x{h}, {size:,} B, stddev={max(stat.stddev):.1f}"
        if ext == ".csv":
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                return False, "CSV 无数据行"
            return True, f"OK {len(rows)-1} 行数据, {len(rows[0])} 列, {size:,} B"
        if ext == ".json":
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
            if not obj:
                return False, "JSON 为空"
            return True, f"OK {size:,} B, 顶层类型 {type(obj).__name__}"
        if ext in (".md", ".txt", ".log"):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if not text.strip():
                return False, "文本为空"
            return True, f"OK {len(text):,} 字符, {size:,} B"
        return True, f"OK {size:,} B"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="", help="逗号分隔的文件路径")
    ap.add_argument("--spec", default="", help="JSON 文件，内容可为路径数组或 {\"files\": [...]}")
    ap.add_argument("--out", default="", help="校验报告写出路径（可选）")
    args = ap.parse_args(argv)

    paths = [p.strip() for p in args.files.split(",") if p.strip()]
    if args.spec:
        with open(args.spec, encoding="utf-8") as f:
            obj = json.load(f)
        extra = obj if isinstance(obj, list) else obj.get("files", [])
        paths += [p.strip() for p in extra if p.strip()]
    if not paths:
        print("没有指定待校验文件")
        return 1

    lines = ["# 产出回读校验报告", ""]
    n_bad = 0
    for p in paths:
        ok, msg = check(p)
        n_bad += 0 if ok else 1
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {p} — {msg}")
        print(f"[{'PASS' if ok else 'FAIL'}] {p} — {msg}")
    lines += ["", f"共 {len(paths)} 项，通过 {len(paths)-n_bad}，失败 {n_bad}。"]
    report = "\n".join(lines) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
