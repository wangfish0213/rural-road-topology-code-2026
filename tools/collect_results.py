# -*- coding: utf-8 -*-
"""汇总所有实验的评测结果，生成一张可直接贴进论文的对比表。

扫描 runs/*/eval/summary.csv，按实验名归并，输出 Markdown 与 CSV。
用法：python tools/collect_results.py
"""

from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
KEYS = ["IoU", "F1", "Precision", "Recall", "线F1", "APLS", "连通率", "断裂率", "伪连接率", "连通分量比"]


def read_summary(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                out[r["metric"]] = float(r["mean"])
            except (KeyError, ValueError):
                continue
    return out


def main() -> int:
    rows = []
    for name in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, name, "eval", "summary.csv")
        if not os.path.exists(p):
            continue
        m = read_summary(p)
        if not m:
            continue
        rows.append((name, m))

    if not rows:
        print("还没有任何评测结果")
        return 1

    # 表头：只看实际存在的指标
    keys = [k for k in KEYS if any(k in m for _, m in rows)]
    out_md = os.path.join(RUNS, "RESULTS.md")
    out_csv = os.path.join(RUNS, "results_table.csv")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["experiment"] + keys)
        for name, m in rows:
            w.writerow([name] + [f"{m.get(k, float('nan')):.4f}" for k in keys])

    lines = ["# 实验结果汇总", "",
             "注意：标 `旧` 的实验是在断裂率定义修订（2026-10-29）之前跑的，指标口径不同，**不可与新实验直接比较**。", "",
             f"共 {len(rows)} 个已评测实验；指标为 WHU-RRoad 测试集（非空切片）上的均值。", "",
             "| 实验 | " + " | ".join(keys) + " |",
             "|---" * (len(keys) + 1) + "|"]
    for name, m in rows:
        cells = [f"{m.get(k, float('nan')):.3f}" if k not in ("连通分量比",) else f"{m.get(k, float('nan')):.1f}"
                 for k in keys]
        tag = "" if "连通率" in m else " `旧`"
        lines.append(f"| {name}{tag} | " + " | ".join(cells) + " |")

    # 按 APLS 排序便于看趋势
    ranked = sorted([t for t in rows if "连通率" in t[1]], key=lambda t: -t[1].get("APLS", -1))
    lines += ["", "## 按 APLS 排序（拓扑完整性）", "",
              "仅统计新口径实验。", "",
              "| 排名 | 实验 | APLS | IoU | 连通率 |", "|---|---|---|---|---|"]
    for i, (name, m) in enumerate(ranked, 1):
        lines.append(f"| {i} | {name} | {m.get('APLS', float('nan')):.3f} | "
                     f"{m.get('IoU', float('nan')):.3f} | {m.get('连通率', float('nan')):.3f} |")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n已写入 {out_md} 与 {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
