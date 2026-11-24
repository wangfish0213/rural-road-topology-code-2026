# -*- coding: utf-8 -*-
"""创新点 3 / 路线② 的增益判据：在看到结果之前写死，避免事后择数。

用法：
  python tools/e4_gain_criteria.py --candidates e4_v3_gt2,e4_v3_gt3,e4_lam_002,...

判据（常量在本文件顶部，任何改动都会写进产物报告，便于复核）：
  P1 主指标增益：line_apls 或 building_iou 的配对均值 Delta >= MIN_GAIN。
  P2 不许拆东墙补西墙：其余两个主指标 Delta >= -MAX_HARM。
  P3 统计稳健：该指标配对 t 检验 p < ALPHA / 候选数（Bonferroni，控制多重比较）。
  P4 非阈值效应：若变体名以 _matched 结尾或有对应 matched 变体，
     只有 fixed 达标而 matched 不达标时，标「疑似阈值效应」，不计入 PASS。
  结论：任一候选通过 P1+P2+P3+P4 -> 找到增益，可保创新点 3；
        全部未通过 -> 按研究方案预案降级为讨论 + 初步实验。

所有数字只读 runs/*/eval/per_image.csv 与 summary.csv，不硬编码任何性能值。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "e4_gain_report.md"
BASE = "e4_v3_base"

# ---- 事前判据常量（改这里必须同步改报告口径） ----
PRIMARY = ("line_apls", "building_iou", "roadarea_iou")
GAIN_CANDIDATES = ("line_apls", "building_iou")
MIN_GAIN = 0.005
MAX_HARM = 0.005
ALPHA = 0.05
LABELS = {"line_apls": "APLS", "building_iou": "建筑 IoU",
          "roadarea_iou": "路面 IoU"}


def read_per_image(name: str) -> dict[str, dict[str, float]]:
    p = RUNS / name / "eval" / "per_image.csv"
    if not p.is_file():
        return {}
    out = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k == "id":
                    continue
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x):
                    row[k] = x
            out[r["id"]] = row
    return out


def read_config(name: str) -> dict:
    p = RUNS / name / "eval" / "config.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def paired_delta(base: dict, cand: dict, key: str):
    """返回 (配对差值数组, 均值, 双尾 p)。按 id 严格配对，缺失样本丢弃。"""
    ids = sorted(set(base) & set(cand))
    d = np.asarray([cand[i][key] - base[i][key] for i in ids
                    if key in base[i] and key in cand[i]], dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return d, float("nan"), float("nan")
    p = float(stats.ttest_rel(
        np.asarray([cand[i][key] for i in ids if key in base[i] and key in cand[i]]),
        np.asarray([base[i][key] for i in ids if key in base[i] and key in cand[i]]),
    ).pvalue)
    return d, float(d.mean()), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="逗号分隔的候选变体名（含基线以外的全部 λ 扫描与后处理变体）")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--label", default="t050",
                    help="口径标签，用于汇总 JSON，如 t030 / t040 / t050")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    names = [s.strip() for s in args.candidates.split(",") if s.strip() and s.strip() != args.base]
    base = read_per_image(args.base)
    if not base:
        print(f"缺基线 {args.base}/eval/per_image.csv")
        return 1

    # 独立工作点数：_matched 是覆盖率诊断、pp_baseline 是阴性对照，
    # 都不是新的工作点，不计入 Bonferroni 的多重比较数。
    independent = [n for n in names
                   if not n.endswith("_matched") and n != "pp_baseline"]
    n_tests = max(1, len(independent))
    p_thresh = ALPHA / n_tests
    results = []
    for n in names:
        cand = read_per_image(n)
        if not cand:
            results.append({"name": n, "ready": False})
            continue
        cfg = read_config(n)
        deltas, ps = {}, {}
        for k in PRIMARY:
            d, mean, p = paired_delta(base, cand, k)
            deltas[k] = (mean, int(d.size))
            ps[k] = p
        p1_metric = [k for k in GAIN_CANDIDATES
                     if deltas[k][0] >= MIN_GAIN and ps[k] < p_thresh]
        p2 = all(deltas[k][0] >= -MAX_HARM for k in PRIMARY)
        results.append({"name": n, "ready": True, "cfg": cfg,
                        "deltas": deltas, "ps": ps, "p1": p1_metric, "p2": p2,
                        "pass": bool(p1_metric) and p2})

    # P4：matched 覆盖率保持版必须同样达标，防止「少预测 = 涨 precision」的阈值假象
    by_name = {r["name"]: r for r in results}
    for r in results:
        if not r.get("ready"):
            continue
        n = r["name"]
        if n.endswith("_matched"):
            continue
        m = by_name.get(n + "_matched")
        if m and m.get("ready") and r["pass"] and not m["pass"]:
            r["pass"] = False
            r["threshold_artifact"] = True

    ready = [r for r in results if r.get("ready")]
    passed = [r for r in ready if r.get("pass")]
    verdict = "找到增益，可保创新点 3" if passed else "未找到增益，按预案降级为讨论 + 初步实验"

    L = ["# 路线② 增益判据与结果", "",
         "> 本文件由 `tools/e4_gain_criteria.py` 生成；判据在运行实验前写死，"
         "数字只读 `runs/*/eval/per_image.csv`，不得手工修改。", "",
         "## 0. 事前判据（写在代码常量里）", "",
         f"- 主指标：{', '.join(LABELS[k] for k in PRIMARY)}；"
         f"增益只认 {', '.join(LABELS[k] for k in GAIN_CANDIDATES)}。",
         f"- P1 配对均值 Δ ≥ **+{MIN_GAIN:.3f}**；P2 其余主指标 Δ ≥ **−{MAX_HARM:.3f}**；"
         f"P3 配对 t 检验 p < {ALPHA}/{n_tests} = **{p_thresh:.5f}**（Bonferroni）；"
         "P4 覆盖率保持版（`_matched`）须同样达标，否则判为阈值效应。",
         f"- **口径 {args.label}**，基线 `{args.base}`；候选 {len(names)} 个"
         f"（独立工作点 {n_tests} 个；"
         "`_matched` 为覆盖率诊断、`pp_baseline` 为阴性对照，不计入多重比较），"
         "按测试图 id 严格配对（不配对样本丢弃）。",
         "- 任一候选同时通过 P1–P4 → 保创新点 3；全部未通过 → 按研究方案预案降级。"
         "**判据在看到任何结果之前确定，不得事后放宽。**", "",
         f"## 1. 结论：{verdict}", ""]

    if ready:
        L += ["## 2. 全部候选的配对 Δ 与显著性（不挑数字，全部列出）", "",
              "| 候选 | 配对张数 | Δ APLS | p(APLS) | Δ 建筑IoU | p(建筑IoU) | "
              "Δ 路面IoU | p(路面IoU) | 判定 |", "|---|---|---|---|---|---|---|---|---|"]
        for r in ready:
            d, p = r["deltas"], r["ps"]
            def f(key, val):
                v, n = d[key]
                return "—" if not math.isfinite(v) else f"{v:+.4f}"
            def fp(key):
                v = p[key]
                return "—" if not math.isfinite(v) else f"{v:.4g}"
            tag = "PASS" if r["pass"] else ("阈值效应" if r.get("threshold_artifact") else "未达标")
            n_pair = min(v[1] for v in d.values())
            L.append(f"| {r['name']} | {n_pair} | {f('line_apls',None)} | {fp('line_apls')} | "
                     f"{f('building_iou',None)} | {fp('building_iou')} | "
                     f"{f('roadarea_iou',None)} | {fp('roadarea_iou')} | {tag} |")
        L.append("")

        L += ["## 3. 实验配置（由各自 config.json 读出）", "",
              "| 候选 | 微调轮数 | 测试张数 | 分辨率 | λ互斥 | λ距离 | λ线在面内 | 公共初始化 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in ready:
            c = r["cfg"]
            lam = c.get("lam") or {}
            L.append(f"| {r['name']} | {c.get('epoch','—')} | {c.get('n_images','—')} | "
                     f"{c.get('size','—')} | {lam.get('excl','—')} | {lam.get('dist','—')} | "
                     f"{lam.get('linein','—')} | {c.get('init','—')} |")
        L.append("")

    missing = [r["name"] for r in results if not r.get("ready")]
    if missing:
        L += ["> 未就绪候选：" + "、".join(missing) + "。", ""]

    L += ["## 4. 判据解读", "",
          "- 配对 t 检验用同一批测试图，消除图间难度差异；Bonferroni 校正是因为 "
          "λ 扫描与后处理变体构成多重比较，不校正时很容易撞出一个偶然显著的点。",
          "- `_matched` 变体把后处理后的线带前景像素数强制恢复到基线水平"
          "（按每张图的分位数取阈值），因此它的 Δ 只反映「像素选得更准」，"
          "不反映「预测得更少」。只有 fixed 版涨、matched 版不涨，即为阈值效应。",
          "- 门槛 +0.005 是相对主链贡献量级（M1 APLS +0.089、M2 分量比 −74.9%）"
          "取的一个保守下限：低于它的变化既无法说服审稿人，也无法排除随机波动。", ""]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # 供 tools/summarize_e4.py 第 7 节自动汇总；数字全部来自本脚本现算
    vdir = RUNS / "e4_gain_verdicts"
    vdir.mkdir(parents=True, exist_ok=True)
    verdict_json = {
        "label": args.label, "base": args.base,
        "n_candidates": len(names), "n_independent": n_tests,
        "p_threshold": p_thresh, "n_ready": len(ready),
        "n_pass": len(passed), "pass_names": [r["name"] for r in passed],
        "verdict": verdict, "report": str(out_path.name),
        "deltas": {r["name"]: {k: r["deltas"][k][0] for k in PRIMARY}
                   for r in ready},
    }
    with open(vdir / f"{args.label}.json", "w", encoding="utf-8") as f:
        json.dump(verdict_json, f, ensure_ascii=False, indent=2)
    print(f"已生成 {out_path}；候选 {len(ready)}/{len(names)} 就绪，"
          f"PASS {len(passed)} → {verdict}")
    for r in passed:
        print(f"  PASS: {r['name']} ({', '.join(LABELS[k] for k in r['p1'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
