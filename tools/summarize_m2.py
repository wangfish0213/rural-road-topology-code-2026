# -*- coding: utf-8 -*-
"""M2 结果汇总：所有数字只读脚本产物（eval/summary.csv、train_log.json、apply_stats.json）。

绝不硬编码任何性能数字。缺文件时标记「未就绪」而非报错，便于链路未跑完时先看部分结果。
用法：python tools/summarize_m2.py
输出：runs/m2_report.md
"""

from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(RUNS, "m2_report.md")

KEYS = ["IoU", "F1", "Precision", "Recall", "线F1", "APLS", "连通率",
        "断裂率", "伪连接率", "连通分量比", "长度比"]

FULL = [
    ("M1 方向场（跨域）", "m1_ori_e1b/eval/summary.csv"),
    ("M2 图结构纠错（跨域）", "m2_corrected/eval/summary.csv"),
]
ABLATION = [
    ("M1 基线", "m2_probe/eval_m1/summary.csv"),
    ("仅语义补连 link-only", "m2_ablation/link_only/eval/summary.csv"),
    ("仅尺寸剪枝 delete-only", "m2_ablation/delete_only/eval/summary.csv"),
    ("纯几何全删（启发式）", "m2_ablation/heuristic/eval/summary.csv"),
    ("M2 完整（补连 + 剪枝）", "m2_probe/eval_m2/summary.csv"),
]


def rd(rel: str):
    p = os.path.join(RUNS, rel)
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                out[r["metric"]] = float(r["mean"])
            except (KeyError, ValueError):
                continue
    return out


def md_table(rows: list[tuple[str, dict | None]]) -> str:
    avail = [k for k in KEYS if any(d and k in d for _, d in rows)]
    lines = ["| 实验 | " + " | ".join(avail) + " |",
             "|---" * (len(avail) + 1) + "|"]
    for name, d in rows:
        if d is None:
            lines.append("| " + name + " | " + " | ".join(["未就绪"] * len(avail)) + " |")
        else:
            lines.append("| " + name + " | " +
                         " | ".join(f"{d[k]:.4f}" if k in d else "—" for k in avail) + " |")
    return "\n".join(lines)


def main() -> int:
    full = [(n, rd(p)) for n, p in FULL]
    abl = [(n, rd(p)) for n, p in ABLATION]

    lines = ["# M2（创新点 2：图结构纠错）结果汇总", "",
             "> 本文件由 `tools/summarize_m2.py` 生成，所有数字只读 `runs/*/eval/summary.csv`，"
             "不得手工修改。", "",
             "> **表述口径（2026-11-09 王老师拍板）**：唯一的语义学习贡献是「跨分量端点补连判别」；"
             "「删除」一律写作非学习的几何后处理规则「尺寸感知孤立分量剪枝」，不得宣称其有学习增量。", "",
             "## 1. 全量测试集主结果（WHU-RRoad test，与 M1 同 2608 张非空切片口径）", "",
             md_table(full), ""]

    a, b = full[0][1], full[1][1]
    if a and b:
        lines += ["| 指标 | M1 | M2 | Δ | 相对 |", "|---|---|---|---|---|"]
        for k in KEYS:
            if k in a and k in b:
                d = b[k] - a[k]
                rel = 100 * d / max(abs(a[k]), 1e-9)
                lines.append(f"| {k} | {a[k]:.4f} | {b[k]:.4f} | {d:+.4f} | {rel:+.1f}% |")
        lines.append("")

    lines += ["## 2. 消融（200 张子集，167 张非空；同一批图、同一评测口径）", "",
              md_table(abl), "",
              "> `纯几何全删` = 不做补连、删除阈值置 0（删除所有 ≤64 px 分量），"
              "用于检验删除分支的 CNN 相对固定规则是否有增量。", "",
              "> **消融判据（决定论文表述）**：`仅尺寸剪枝` 与 `纯几何全删` 在全部指标上一致，"
              "说明该尺度下 CNN 删除判别器与固定尺寸规则无可测差异；"
              "因此删除分支只作为几何后处理规则报告，语义学习增量全部归于补连分支。", ""]

    tl = os.path.join(RUNS, "m2_model", "train_log.json")
    if os.path.exists(tl):
        with open(tl, encoding="utf-8") as f:
            t = json.load(f)
        lines += ["## 3. 纠错器工作点（阈值在 train 划分的独立 val 上选定，未用测试集调参）", "",
                  "| 分支 | 目标精度 | 阈值 | 模式 | 验证集 P | 验证集 R | 候选应用率 |",
                  "|---|---|---|---|---|---|---|"]
        for task in ("link", "delete"):
            v = t["tasks"][task]
            x = v["val"]
            lines.append(f"| {task} | {v['target_precision']:.2f} | {x['thr']} | "
                         f"{x['mode']} | {x['precision']:.3f} | {x['recall']:.3f} | "
                         f"{x['applied_rate']:.3f} |")
        lines.append("")

    ap = os.path.join(RUNS, "m2_corrected", "apply_stats.json")
    if os.path.exists(ap):
        with open(ap, encoding="utf-8") as f:
            s = json.load(f)
        m = s.get("mean_per_tile", {})
        lines += ["## 4. 纠错操作量（全量每图均值）", "",
                  "| 补连候选 | 实际补连 | 删除候选 | 实际删除 | 纠错前像素 | 纠错后像素 |",
                  "|---|---|---|---|---|---|",
                  f"| {m.get('link_cand', float('nan')):.1f} | {m.get('link_applied', float('nan')):.2f} | "
                  f"{m.get('del_cand', float('nan')):.1f} | {m.get('del_applied', float('nan')):.1f} | "
                  f"{m.get('px_before', float('nan')):.0f} | {m.get('px_after', float('nan')):.0f} |", ""]

    lines += ["## 5. 口径与诚实边界", "",
              "- 纠错器训练数据来自 M1 在 WHU-RRoad **train 划分**上的预测 + train GT；"
              "最终评测在 **test** 划分，与 M1 同口径，无测试集泄漏。",
              "- 阈值由 train 划分内独立 val 的候选级精度约束选定，未用 test 调参；"
              "消融与全量评测均只是报告结果。",
              "- 评测：`eval_dir.py --gsd 0.3 --tol 3 --samples 60`，默认剔除空标注切片。",
              "- 补连会引入少量伪连接（代价，见第 1 节）；删除分支虽训练了 CNN，"
              "但与「删掉所有 ≤64 px 分量」的固定规则在第 2 节消融中全部指标 Δ=0，"
              "故论文把它表述为非学习的几何后处理规则「尺寸感知孤立分量剪枝」，不宣称学习增量。",
              "- WHU-RRoad 标注为稀疏细线，与标注零重叠的分量可能是漏标真实道路，"
              "不得断言为假阳性（见 `runs/cc_analysis/findings.md`）。", ""]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成 {OUT}")
    for name, d in full + abl:
        print(f"  {name}: {'就绪' if d else '未就绪'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
