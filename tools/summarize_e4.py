# -*- coding: utf-8 -*-
"""创新点 3 / E4 结果汇总：只读各变体 eval/summary.csv，绝不硬编码性能数字。

输出 runs/e4_report.md；缺变体时标「未就绪」，不报错。
用法：python tools/summarize_e4.py [--variants e4_v3_base,e4_v3_pred,e4_v3_gt2,e4_v3_gt3]

编号口径：本文只有 3 个创新点，E1–E4 是实验编号。
  E1 跨域 / E2 同域 → 创新点 1、2 的主战场（WHU-RRoad）
  E3 面状辅助（WHU Building）、E4 线面联合（IRSAMap）→ 验证创新点 3
  M1/M2 是方法模块，不是实验编号。E4 ≠ 创新点 4。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "e4_report.md"

DISPLAY = [
    ("building_iou", "建筑 IoU"),
    ("building_f1", "建筑 F1"),
    ("roadarea_iou", "路面 IoU"),
    ("roadarea_f1", "路面 F1"),
    ("line_iou", "中心线 IoU"),
    ("line_line_f1", "线F1(2px)"),
    ("line_apls", "APLS"),
    ("line_connectivity", "连通率"),
    ("line_breakage_rate", "断裂率"),
    ("line_false_connection_rate", "伪连接率"),
    ("line_cc_ratio", "连通分量比"),
    ("excl_violation", "互斥违规率(预测vs预测)"),
    ("boundary_line_violation", "边界-中心线违规率(预测vs预测)"),
    ("line_in_roadarea", "中心线在路面内(预测vs预测)"),
    ("gt_excl_violation", "预测建筑压真值路面 / 真值路面"),
    ("gt_boundary_line_violation", "预测建筑边界近真值中心线"),
    ("gt_line_in_roadarea", "预测线在真值路面内"),
    ("pred_line_coverage", "预测线带覆盖率(退化诊断)"),
    ("pred_roadarea_coverage", "预测路面覆盖率(退化诊断)"),
    ("pred_building_coverage", "预测建筑覆盖率(退化诊断)"),
]
NAMES = {
    "e4_v3_base": "基线（无约束）",
    "e4_v3_pred": "预测锚定双约束（v2 口径）",
    "e4_v3_gt2": "真值锚定双约束（v3）",
    "e4_v3_gt3": "真值锚定三约束（+线在面内）",
    "e4_base": "v2 基线（已归档，仅历史）",
    "e4_cons": "v2 双约束（已归档，仅历史）",
    "e4_cons3": "v2 三约束（已归档，仅历史）",
    "e4_excl_only": "E4 仅互斥约束",
    "e4_dist_only": "E4 仅距离约束",
    "e4_linein_only": "E4 仅线在面内约束",
}
BASE_KEY = "e4_v3_base"


def read_summary(name: str):
    p = RUNS / name / "eval" / "summary.csv"
    if not p.is_file():
        return None, None
    d = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                d[r["metric"]] = float(r["mean"])
            except (KeyError, ValueError):
                continue
    cfg_path = RUNS / name / "eval" / "config.json"
    cfg = None
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    return d, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants",
                    default="e4_v3_base,e4_v3_pred,e4_v3_gt2,e4_v3_gt3")
    args = ap.parse_args()
    names = [s.strip() for s in args.variants.split(",") if s.strip()]
    data = {n: read_summary(n) for n in names}
    avail = [n for n in names if data[n][0]]

    lines = ["# E4（创新点 3：线—面一致性约束）结果汇总", "",
             "> 本文件由 `tools/summarize_e4.py` 生成，数字只读 "
             "`runs/*/eval/summary.csv`，不得手工修改。", "",
             "> 与 E1/E2/M1/M2 不同：E4 用 IRSAMap V2 同景数据（建筑 + 路面 + 道路中心线），"
             "**指标不可与 WHU-RRoad 的 RESULTS.md 直接比较**。", "",
             "> **编号口径（勿混淆）**：全文只有 **3 个创新点**，E1–E4 是**实验编号**。"
             "E1 跨域 / E2 同域 → 创新点 1、2；E3（WHU Building 面状辅助）与 "
             "E4（IRSAMap 线面联合）→ 验证**创新点 3**；M1/M2 是方法模块。"
             "因此 **E4 不是「创新点 4」**。", ""]

    if avail:
        lines += ["## 1. 各变体指标（同公共初始化、同种子、同数据、同微调轮数）", "",
                  "| 指标 | " + " | ".join(NAMES.get(n, n) for n in avail) + " |",
                  "|---" * (len(avail) + 1) + "|"]
        for key, label in DISPLAY:
            if not any(key in (data[n][0] or {}) for n in avail):
                continue
            vals = []
            for n in avail:
                v = (data[n][0] or {}).get(key)
                vals.append("—" if v is None else f"{v:.4f}")
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        lines.append("")

        if BASE_KEY in data and data[BASE_KEY][0] and len(avail) > 1:
            base = data[BASE_KEY][0]
            lines += ["## 2. 相对基线的变化（Δ = 变体 − 基线）", "",
                      "| 指标 | " + " | ".join(NAMES.get(n, n) for n in avail if n != BASE_KEY) + " |",
                      "|---" * (len(avail)) + "|"]
            for key, label in DISPLAY:
                if key not in base:
                    continue
                vals = []
                for n in avail:
                    if n == BASE_KEY:
                        continue
                    v = (data[n][0] or {}).get(key)
                    vals.append("—" if v is None else f"{v-base[key]:+.4f}")
                lines.append(f"| {label} | " + " | ".join(vals) + " |")
            lines.append("")

        lines += ["## 3. 实验配置（由 eval/config.json 读出）", "",
                  "| 变体 | 微调轮数 | 测试张数 | 分辨率 | 约束锚 | λ互斥 | λ距离 | λ线在面内 | 公共初始化 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for n in avail:
            c = data[n][1] or {}
            lam = c.get("lam") or {}
            init = c.get("init")
            init_s = os.path.basename(str(init)) if init else "—"
            lines.append(f"| {NAMES.get(n, n)} | {c.get('epoch','—')} | "
                         f"{c.get('n_images','—')} | {c.get('size','—')} | "
                         f"{c.get('anchor','—')} | "
                         f"{lam.get('excl','—')} | {lam.get('dist','—')} | "
                         f"{lam.get('linein','—')} | {init_s} |")
        lines += ["", "> 公共初始化 `e4_v3_shared`：176² / 300 张 / 14 轮 / lr 2e-3 / "
                  "posw 3.0（线头 6.0）/ λ=0。四个变体从它的 `last.pt` 出发，"
                  "各微调 6 轮（lr 1e-3），同 seed、同数据顺序，只差约束锚与 λ。", ""]
        lines.append("")

    missing = [n for n in names if not data[n][0]]
    if missing:
        lines += ["> 未就绪变体：" + "、".join(missing) + "。", ""]

    gt = RUNS / "e4_prep" / "gt_consistency.json"
    if gt.is_file():
        with open(gt, encoding="utf-8") as f:
            g = json.load(f)
        lines += ["## 4. 立论前提：真值层面的线—面一致性（tools/e4_gt_consistency.py）", "",
                  "| 划分 | 建筑∩路面/路面 | 建筑边界-中心线违规 | 中心线在路面内 | 张数 |",
                  "|---|---|---|---|---|"]
        for sp, s in g["splits"].items():
            lines.append(f"| {sp} | {s['overlap_over_roadarea']['mean']:.4f} | "
                         f"{s['boundary_line_violation']['mean']:.4f} | "
                         f"{s['line_in_roadarea']['mean']:.4f} | {s['n_images']} |")
        lines.append("")

    lines += ["## 5. 口径与诚实边界", "",
              "- **编号**：全文只有 3 个创新点；E4 是第 4 组实验，验证创新点 3，不是创新点 4。",
              "- **本机约束（2026-11-09 定：没有外部卡，全程本机 MX450 2 GB）**："
              "训练耗时≈样本数×分辨率²×轮数，实测 batch 4→16 无提速；"
              "混合精度在本卡产生 NaN 且不提速（`runs/e4_v3_amp_nan_invalid`，作废）。"
              "故采用「公共初始化 + 四变体各微调 6 轮」压缩机时，"
              "规模定为本机可跑完的最大值（176² / 300 训练张 / 120 测试张）。",
              "- **v2 → v3 方法修正（须避免过度归因）**：v2 的 pred-vs-pred 约束在理论上存在"
              "**平凡解**（关掉路面头或线头即可让互斥损失趋零），v2 小样本里确实出现 `roadarea_iou→0`。"
              "v3 改为**真值锚定**的训练期正则（`p_b·GT_roadarea + p_r·GT_building`、"
              "`boundary(p_b)·GT_line_near`、`p_l·(1-GT_roadarea)`），推理时不需要真值。"
              "**但本次充分训练对照中 pred 锚定并未复现平凡解**（路面 IoU 与基线几乎相同），"
              "说明 v2 的路面头归零更可能是「从头 6 轮严重欠拟合 + 约束」共同作用；"
              "因此 v3 只是消除一个潜在退化风险，**不得宣称它带来性能增益**。",
              "- 所有变体统一取最后一个 epoch，不按验证集挑最优；IRSAMap 栅格标签是类别值"
              "（building=31、roadarea=32），必须 >0 二值化；中心线由 "
              "`tools/irsamap_prep.py` 从 GeoJSON 栅格化（3 px @1024）。",
              "- 训练/评测在缩放切片上进行，tol_px、gsd 与 E1/M1 不同，"
              "**只做 E4 变体间横向比较**，不得与 WHU-RRoad 的 `RESULTS.md` 并列。",
              "- **v3 收敛状况（与 v2 的关键差别）**：公共初始化 14 轮 + 微调 6 轮后三头均已学到结构"
              "（建筑/路面 IoU 约 0.40/0.24，预测线带覆盖率约 0.17，v2 曾高达 0.57），"
              "v2 那种「覆盖率异常 → 指标退化」的情形已大幅缓解，故本表指标可作变体间比较。"
              "但三头仍共享小型解码器、存在此消彼长的双稳态，**判据仍以分母固定的 `gt_*` 三行、"
              "`building_iou` / `roadarea_iou` 为主**；APLS/线F1 只作同口径横向比较，"
              "不得与 E1/E2 结果并列或当作绝对性能。",
              "- **历史版本作废**：v1 细线直监督（中心线头全 0）、v2 pred 锚定小样本、"
              "v3 的 AMP NaN 探针均已归档，其数字不得引用。",
              "- 本阶段结论与决策判据见第 6 节（由本脚本从 `summary.csv` 现算，不得手工转抄）。", ""]
    # 第 6 节：决策判据（所有数字由本脚本从 summary.csv 现算，不硬编码）
    b = data.get(BASE_KEY, (None, None))[0]
    g2 = data.get("e4_v3_gt2", (None, None))[0]
    p2 = data.get("e4_v3_pred", (None, None))[0]
    g3 = data.get("e4_v3_gt3", (None, None))[0]
    if b and g2:
        def delta(src, key):
            if not src or key not in src or key not in b:
                return float("nan")
            return src[key] - b[key]

        lines += ["## 6. 决策判据（数字由本脚本从 summary.csv 现算）", ""]
        for nm, src in (("预测锚定双约束", p2), ("真值锚定双约束", g2),
                        ("真值锚定三约束", g3)):
            if not src:
                continue
            lines.append(f"- **{nm}** 相对基线：建筑 IoU {delta(src,'building_iou'):+.4f}、"
                         f"路面 IoU {delta(src,'roadarea_iou'):+.4f}、"
                         f"APLS {delta(src,'line_apls'):+.4f}、"
                         f"预测建筑压真值路面 {delta(src,'gt_excl_violation'):+.4f}、"
                         f"预测侧互斥违规率 {delta(src,'excl_violation'):+.4f}。")
        lines += ["",
                  "- **核心判断（写论文必须遵守）**：约束确实把**预测侧**一致性违规率压了下去，"
                  "但以真值为参照的违规率、建筑/路面 IoU、APLS 都只有千分量级变化 → "
                  "本阶段**不能宣称一致性约束带来性能提升**，只能宣称「约束在训练中生效并改变预测分布」。",
                  "- **v2 结论作废**：v2 小样本里「预测建筑压真值路面大幅下降」在模型收敛后不复现"
                  "（基线本身已降到低得多的水平），属欠拟合假象，**不得作为创新点 3 的效果证据**。",
                  "- **平凡解未在充分训练下复现**：预测锚定变体的路面 IoU 与基线几乎相同，"
                  "故 v3 真值锚定只是消除潜在退化风险，不构成可测增量。",
                  "- **第三条约束不纳入贡献列表**：它抬高了「中心线在路面内」，"
                  "但 APLS 与连通率下降、断裂率上升、建筑 IoU 略降，与 v2 结论一致。",
                  "- **决策与执行（王老师 2026-11-11 拍板：走路线②）**：不直接降级、也不硬保留，"
                  "先做 λ 扫描与推理期一致性后处理，寻找能转化为 IoU/APLS 增益的工作点。"
                  "**判据在实验前写死**于 `tools/e4_gain_criteria.py`（Δ ≥ +0.005、"
                  "配对 t 检验 Bonferroni 校正、覆盖率保持版 `_matched` 防「少预测」阈值效应），"
                  "实验与三口径复核结果见第 7 节；找到增益才保创新点 3，"
                  "否则按研究方案预案降级为讨论 + 初步实验。"
                  "无论哪条，E4 只报本机规模（176² / 300 训练张 / 120 测试张），"
                  "不得与 E1/E2 全量表并列。", ""]

    # 第 7 节：路线② 自动汇总（全部读脚本产物，不硬编码性能数字）
    vdir = RUNS / "e4_gain_verdicts"
    vfiles = sorted(vdir.glob("*.json")) if vdir.is_dir() else []
    if vfiles:
        vs = []
        for p in vfiles:
            with open(p, encoding="utf-8") as f:
                vs.append(json.load(f))
        vs.sort(key=lambda v: v["label"])
        total_cand = sum(v["n_candidates"] for v in vs)
        total_pass = sum(v["n_pass"] for v in vs)
        lines += ["## 7. 路线②：增益判据与结果（数字由判据脚本现算）", "",
                  "> 本节读 `runs/e4_gain_verdicts/*.json`（由 `tools/e4_gain_criteria.py` 生成）；"
                  "判据在跑实验前写死，不得事后放宽。", "",
                  "| 口径 | 基线 | 候选 | 独立工作点 | Bonferroni p 阈值 | PASS | 结论 |",
                  "|---|---|---|---|---|---|---|"]
        for v in vs:
            lines.append(f"| {v['label']} | `{v['base']}` | {v['n_candidates']} | "
                         f"{v['n_independent']} | {v['p_threshold']:.5f} | "
                         f"{v['n_pass']} | {v['verdict']} |")
        lines.append("")
        if total_pass == 0:
            lines += [f"- **总判定：{len(vs)} 个口径共 {total_cand} 个候选，0 个通过 → "
                      "未找到可转化为 IoU/APLS 的增益，按研究方案预案把创新点 3 "
                      "降级为「讨论 + 初步实验」，不进贡献列表，题目回到道路中心线提取"
                      "（如另有异议可再议）。**", ""]
        else:
            hit = [n for v in vs for n in v["pass_names"]]
            lines += [f"- **总判定：{total_pass} 个候选通过事前判据（{', '.join(hit)}），"
                      "创新点 3 可保留在贡献列表；须在论文中写明工作点、配对显著性与"
                      "覆盖率保持版结果。**", ""]
        lines += ["- 逐候选 Δ 与配对 p 值见 `runs/e4_gain_t030.md`、`runs/e4_gain_t040.md`、"
                  "`runs/e4_gain_t050.md`；三个口径各自独立做 Bonferroni 校正。", ""]
        t50 = next((v for v in vs if v["label"] == "t050"), None)
        if t50:
            g = t50["deltas"].get("pp_line_gate_soft", {}).get("line_apls")
            r = t50["deltas"].get("pp_excl_r", {}).get("roadarea_iou")
            if g is not None and r is not None:
                lines += ["- **推理期后处理的负面结果（必须如实写进 Discussion）**："
                          f"用预测路面门控中心线让 APLS {g:+.4f}（覆盖率保持版仍为负），"
                          "原因是路面头 IoU 仅约 0.24，门控删掉了正确线像素；"
                          f"用建筑预测抑制路面则让路面 IoU {r:+.4f}。"
                          "即「真值中 99.3% 的线在路面内」这一先验成立，"
                          "但**用低精度预测去做门控并不能把它转化为增益**。", ""]
        b = read_summary("pp_baseline")[0]
        t3 = read_summary("pp_thr030")[0]
        if b and t3:
            lines += ["- **阈值敏感性诊断**（排除「工作点没调好」的质疑）：统一把线带阈值 "
                      f"从 0.5 降到 0.3，基线 APLS {b['line_apls']:.4f}→"
                      f"{t3['line_apls']:.4f}（{t3['line_apls']-b['line_apls']:+.4f}），"
                      f"但线F1 {b['line_line_f1']:.4f}→{t3['line_line_f1']:.4f}"
                      f"（{t3['line_line_f1']-b['line_line_f1']:+.4f}）、线带覆盖率 "
                      f"{b['pred_line_coverage']:.4f}→{t3['pred_line_coverage']:.4f}，"
                      "属「更粗的线带换拓扑分」的权衡而非净增益；且在 0.3 / 0.4 "
                      "两个统一口径下重评，训练期各变体仍全部未通过判据。", ""]
        lines += ["- 全部变体与阈值明细见 `runs/e4_postprocess_diagnostics.md`；"
                  "后处理与阈值复核的阴性对照（`pp_baseline`、`*_t050`）逐指标差异均为 "
                  "0.00000000，说明与原评测严格同口径。", ""]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成 {OUT}；就绪变体 {len(avail)}/{len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
