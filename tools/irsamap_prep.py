# -*- coding: utf-8 -*-
"""创新点 3 / E4 数据准备：IRSAMap 矢量道路中心线 -> 栅格掩膜 + E4 索引。

背景（2026-11-09 实测）：IRSAMap V2 只给 roadline 的 GeoJSON，没有栅格标签；
坐标是切片局部像素坐标（0..1024，虽然 crs 字段写 EPSG:3857），因此可直接栅格化。

产物（全部由本脚本写盘，数字不得人工转抄）：
  datasets/IRSAMap/IRSAMap_V2/{split}/label_roadline/{id}.png   0/255 中心线掩膜
  manifests/irsamap_e4_{split}.csv                               三类齐备切片 + roadline 列
  runs/e4_prep/stats.json                                        逐 split 统计
  previews/e4_prep_samples.png                                   影像/建筑/路面/中心线叠加预览

用法：
  PYTHONPATH="F:/pyenvs/road/Lib/site-packages" python tools/irsamap_prep.py --limit 20
  PYTHONPATH="F:/pyenvs/road/Lib/site-packages" python tools/irsamap_prep.py --skip-existing
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "IRSAMap" / "IRSAMap_V2"
MANIFESTS = ROOT / "manifests"
RUNS = ROOT / "runs" / "e4_prep"
PREVIEWS = ROOT / "previews"

SIZE = (1024, 1024)


def rasterize(geo_path: Path, width: int = 3):
    """把 GeoJSON 的 LineString / MultiLineString 画成 0/255 掩膜。

    返回 (mask_uint8, stat_dict)。坐标裁剪到 [0, 1023]；不支持的几何类型计数跳过。
    """
    with open(geo_path, encoding="utf-8") as f:
        gj = json.load(f)
    img = Image.new("L", SIZE, 0)
    draw = ImageDraw.Draw(img)
    feats = gj.get("features", [])
    n_lines = n_pts = n_skip = 0
    for feat in feats:
        geom = feat.get("geometry") or {}
        typ, coords = geom.get("type"), geom.get("coordinates") or []
        if typ == "LineString":
            lines = [coords]
        elif typ == "MultiLineString":
            lines = coords
        else:
            n_skip += 1
            continue
        for line in lines:
            pts = []
            for xy in line:
                if not xy or len(xy) < 2:
                    continue
                x = max(0, min(SIZE[1] - 1, int(round(float(xy[0])))))
                y = max(0, min(SIZE[0] - 1, int(round(float(xy[1])))))
                pts.append((x, y))
            if len(pts) < 2:
                continue
            draw.line(pts, fill=255, width=width, joint="curve")
            n_lines += 1
            n_pts += len(pts)
    mask = np.asarray(img, dtype=np.uint8)
    stat = {
        "n_features": len(feats),
        "n_lines": n_lines,
        "n_points": n_pts,
        "n_geom_skipped": n_skip,
        "line_px": int((mask > 0).sum()),
    }
    return mask, stat


def to_binary(path: Path, resize: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if resize:
        im = im.resize((resize, resize), Image.NEAREST)
    return (np.asarray(im) > 0).astype(np.uint8)


def make_preview(samples: list[dict], out_path: Path, tile: int = 256):
    """3 行样例 × 4 列（影像 / 建筑 / 路面 / 中心线）彩色叠加图。"""
    if not samples:
        return None
    canvas = Image.new("RGB", (tile * 4, tile * len(samples)), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    colours = {"building": (255, 60, 60), "roadarea": (60, 120, 255),
               "roadline": (0, 220, 60)}
    for i, s in enumerate(samples):
        base = np.asarray(Image.open(s["image"]).convert("RGB").resize(
            (tile, tile), Image.BILINEAR)).astype(np.float32)
        panels = [base]
        for key in ("building", "roadarea", "roadline"):
            m = to_binary(Path(s[key]), tile).astype(bool)
            v = base.copy()
            v[m] = 0.45 * v[m] + 0.55 * np.array(colours[key], dtype=np.float32)
            panels.append(v)
        for j, v in enumerate(panels):
            canvas.paste(Image.fromarray(np.clip(v, 0, 255).astype(np.uint8)),
                         (j * tile, i * tile))
        d.text((4, i * tile + 4), f"{s['id']}  image | building | roadarea | roadline",
               fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA))
    ap.add_argument("--manifest-dir", default=str(MANIFESTS))
    ap.add_argument("--out", default=str(RUNS))
    ap.add_argument("--width", type=int, default=3, help="中心线栅格线宽（像素）")
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只处理前 N 张（冒烟用）")
    ap.add_argument("--skip-existing", action="store_true", help="断点续跑，跳过已有 PNG")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_stats, preview_samples = {}, []

    for split in ("train", "val", "test"):
        src = Path(args.manifest_dir) / f"irsamap_{split}.csv"
        if not src.is_file():
            print(f"{split}: 缺索引 {src}")
            continue
        rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
        if args.limit:
            rows = rows[:args.limit]
        dst_dir = root / split / "label_roadline"
        dst_dir.mkdir(parents=True, exist_ok=True)

        e4_rows, st = [], {
            "n_input": len(rows), "n_written": 0, "n_skipped_existing": 0,
            "n_missing_geojson": 0, "n_empty_line": 0, "n_failed": 0,
            "n_features": 0, "n_lines": 0, "n_geom_skipped": 0,
            "sum_line_px": 0,
        }
        for r in rows:
            geo = r.get("roadline_geojson", "")
            if not geo or not os.path.isfile(geo):
                st["n_missing_geojson"] += 1
                continue
            dst = dst_dir / f"{r['id']}.png"
            if args.skip_existing and dst.is_file() and dst.stat().st_size > 0:
                st["n_skipped_existing"] += 1
                line_px = int((np.asarray(Image.open(dst).convert("L")) > 0).sum())
                st["sum_line_px"] += line_px
                st["n_written"] += 1
            else:
                try:
                    mask, s = rasterize(Path(geo), args.width)
                    Image.fromarray(mask).save(dst)
                    st["n_written"] += 1
                    st["n_features"] += s["n_features"]
                    st["n_lines"] += s["n_lines"]
                    st["n_geom_skipped"] += s["n_geom_skipped"]
                    st["sum_line_px"] += s["line_px"]
                    if s["line_px"] == 0:
                        st["n_empty_line"] += 1
                except Exception as e:  # 单张失败不中断，计数上报
                    st["n_failed"] += 1
                    print(f"  ! {split}/{r['id']} 栅格化失败: {e}")
                    continue
            row = dict(r)
            row["roadline"] = str(dst)
            e4_rows.append(row)
            if len(preview_samples) < 3 and split == "train":
                preview_samples.append({"id": r["id"], "image": r["image"],
                                        "building": r["building"],
                                        "roadarea": r["roadarea"],
                                        "roadline": str(dst)})

        fields = list(rows[0].keys()) + ["roadline"] if rows else []
        dst_csv = Path(args.manifest_dir) / f"irsamap_e4_{split}.csv"
        with open(dst_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(e4_rows)
        st["n_e4_rows"] = len(e4_rows)
        st["mean_line_px"] = round(st["sum_line_px"] / max(1, st["n_written"]), 1)
        st["line_width"] = args.width
        all_stats[split] = st
        print(f"{split}: 写入 {st['n_written']}（跳过 {st['n_skipped_existing']}），"
              f"E4 索引 {len(e4_rows)} 条，空线 {st['n_empty_line']}，失败 {st['n_failed']}，"
              f"均线像素 {st['mean_line_px']} → {dst_csv}")

    stats_path = out / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"generated_by": "tools/irsamap_prep.py",
                   "elapsed_s": round(time.time() - t0, 1),
                   "splits": all_stats}, f, ensure_ascii=False, indent=2)
    pv = make_preview(preview_samples, PREVIEWS / "e4_prep_samples.png")
    print(f"统计：{stats_path}")
    print(f"预览：{pv}")
    print(f"耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
