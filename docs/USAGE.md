# Command-line reference

Every script in `tools/`, `train/` and `eval/` is listed below with the options
declared in its argument parser. Run any script with `--help` for the full text.
Scripts that only expose functions (libraries) have no options.

## `eval/eval_dir.py`

批量评测：逐张计算指标并汇总，支持多进程与 Markdown 输出。

Options: `--gt`, `--pred`, `--out`, `--gsd`, `--tol`, `--samples`, `--workers`, `--limit`, `--threshold`, `--keep-empty`, `--gt-suffix`, `--pred-suffix`

## `eval/metrics.py`

道路中心线/掩膜评测核心指标。

Library module (no command-line options).

## `eval/selfcheck.py`

合成样例自测：不依赖任何真实数据，验证指标实现的正确性与方向性。

Library module (no command-line options).

## `tools/analyze_cc.py`

任务 2：M1 连通分量比偏高（93 vs baseline 50）的归因分析。

Options: `--limit`, `--out`, `--models`, `--findings-only`

## `tools/audit_manuscript.py`

Audit the manuscript for provenance violations.

Library module (no command-line options).

## `tools/build_manuscript_tables.py`

Generate all manuscript tables and number macros from evaluation products.

Library module (no command-line options).

## `tools/collect_results.py`

汇总所有实验的评测结果，生成一张可直接贴进论文的对比表。

Library module (no command-line options).

## `tools/dataset_stats.py`

数据集体检：配对完整性、掩膜编码、前景占比、连通分量、骨架长度。

Options: `--root`, `--images-dir`, `--masks-dir`, `--images-suffix`, `--masks-suffix`, `--out`, `--limit`, `--with-skeleton`, `--workers`

## `tools/e4_gain_criteria.py`

创新点 3 / 路线② 的增益判据：在看到结果之前写死，避免事后择数。

Options: `--candidates`, `--base`, `--label`, `--out`

## `tools/e4_gt_consistency.py`

创新点 3 / E4 立论前提：真值层面「线—面一致性」是否成立。

Options: `--split`, `--limit`, `--resize`, `--radius`, `--out`

## `tools/irsamap_prep.py`

创新点 3 / E4 数据准备：IRSAMap 矢量道路中心线 -> 栅格掩膜 + E4 索引。

Options: `--root`, `--manifest-dir`, `--out`, `--width`, `--limit`, `--skip-existing`

## `tools/make_m1_pred_bat.py`

生成 run_m1_pred_resume.bat（GBK 编码，供 schtasks 启动 M1 断点推理）。

Library module (no command-line options).

## `tools/make_manifest.py`

生成数据集索引清单（manifest）：id / 影像路径 / 标注路径 / 划分。

Options: `--root`, `--images-dir`, `--masks-dir`, `--split`, `--out`, `--images-suffix`, `--masks-suffix`

## `tools/make_manifest_irsamap.py`

为 IRSAMap 生成索引清单（多图层版）。

Options: `--root`, `--out`, `--require`

## `tools/make_pred_preview.py`

对比图：影像 / 真值 / 预测 三联，用于直观检查断裂与碎片。

Options: `--manifest`, `--pred`, `--out`, `--n`, `--size`, `--skip`

## `tools/make_figures.py`

Rebuild the six manuscript figures at 600 dpi from the evaluation products.
Requires Pillow, numpy and scipy only (no matplotlib), and the rendered
predictions under `runs/`; `--fig` selects one figure or a comma separated list.

Options: `--fig {1,3,4,5,6,all}`, `--src-id`, `--tgt-id`

The vector pipeline diagram is built separately:
`pdflatex tools/fig2_pipeline.tex` (standalone TikZ class).

## `tools/figlib.py`

Shared drawing primitives for `tools/make_figures.py`: centimetre/pixel
conversion at a fixed dpi, scale bars, pixel axes, colour bars, translucent
overlays and a small text/label canvas. Library module (no command-line
options).

## `tools/make_preview.py`

生成样本预览图：影像 + 掩膜叠加，用于肉眼抽查标注质量。

Options: `--root`, `--out`, `--n`, `--cell`, `--images-suffix`, `--masks-suffix`, `--seed`

## `tools/make_split.py`

把一份 manifest 切成 train/val（按 id 哈希，可复现，换机重跑结果一致）。

Options: `--manifest`, `--ratio`, `--seed`, `--prefix`

## `tools/plot_cc_figs.py`

任务 2 配图：把"孤立（未匹配）分量"可视化 + 归因统计图。

Options: `--data`, `--out`

## `tools/summarize_e4.py`

创新点 3 / E4 结果汇总：只读各变体 eval/summary.csv，绝不硬编码性能数字。

Options: `--variants`

## `tools/summarize_m2.py`

M2 结果汇总：所有数字只读脚本产物（eval/summary.csv、train_log.json、apply_stats.json）。

Library module (no command-line options).

## `tools/verify_outputs.py`

产出回读校验器（保险①）：确认脚本声称生成的文件真的落盘且内容完整。

Options: `--files`, `--spec`, `--out`

## `tools/wordcount.py`

Count the words that Elsevier counts for Computers & Geosciences.

Library module (no command-line options).

## `train/correction.py`

创新点 2：图节点的语义纠错（补连 / 删除 / 保留）。

Options: `--pred`, `--gt`, `--limit`, `--holdout`, `--epochs`, `--lr`, `--out`

## `train/e4_consistency.py`

创新点 3 / E4：线—面跨任务一致性软约束（IRSAMap V2，同景建筑+路面+中心线）。

Options: `--name`, `--size`, `--base`, `--batch`, `--epochs`, `--lr`, `--train-limit`, `--test-limit`, `--radius`, `--line-dilate`, `--line-pos-weight`, `--posw`, `--posw-line`, `--anchor`, `--no-cache`, `--lam-excl`, `--lam-dist`, `--lam-linein`, `--seed`, `--cpu`, `--amp`, `--init`

## `train/e4_eval_thresholds.py`

路线② 收口复核：在多个**统一**线带阈值口径下重评全部训练期变体。

Options: `--size`, `--test-limit`, `--radius`, `--cpu`

## `train/e4_postprocess.py`

路线② 后半：推理期线—面一致性后处理（不改动训练权重）。

Options: `--size`, `--test-limit`, `--radius`, `--base`, `--cpu`

## `train/m2_correction.py`

创新点 2 / M2：图结构纠错（语义补连判别 + 尺寸感知孤立分量剪枝）。

Options: `--pred`, `--gt`, `--image-dir`, `--out`, `--limit`, `--val-frac`, `--cap-link`, `--cap-del`, `--data`, `--out`, `--epochs`, `--lr`, `--batch`, `--device`, `--model`, `--pred`, `--image-dir`, `--out`, `--out-dir-stats`, `--limit`, `--thr-link`, `--thr-del`, `--overwrite`, `--device`

Subcommands: `build`, `train`, `apply`

## `train/orientation.py`

创新点 1：方向场（orientation field）辅助的中心线提取。

Library module (no command-line options).

## `train/post_m1_eval.py`

M1 训练+推理完成后的评测收尾（2026-10-30）。

Library module (no command-line options).

## `train/predict.py`

滑窗推理：对 1024 切片逐块预测并写回全分辨率概率图，输出二值掩膜 PNG。

Options: `--ckpt`, `--manifest`, `--out`, `--tile`, `--stride`, `--thr`, `--limit`, `--batch`, `--device`, `--save-prob`

## `train/road_graph.py`

骨架 → 道路图：节点、边链、候选连接的生成。

Library module (no command-line options).

## `train/run_ladder.py`

接力脚本：等当前队列跑完，再依次启动后续 stage。

Options: `--wait-log`, `--marker`, `--timeout-h`, `--stages`, `--extra-args`

## `train/run_queue.py`

串行实验队列：训练 → 推理 → 评测，一个作业接一个作业跑。

Options: `--stage`, `--epochs`, `--limit-train`, `--e1a-epochs`, `--e2-limit`, `--r50-epochs`, `--r50-limit`, `--eval-limit`, `--job-delay`, `--skip-existing`

## `train/train_ori.py`

训练带方向场分支的模型（创新点 1 的实现入口）。

Options: `--manifest`, `--val-manifest`, `--epochs`, `--crop`, `--batch`, `--lr`, `--base`, `--pos-weight`, `--ori-weight`, `--limit-train`, `--limit-val`, `--max-iters`, `--workers`, `--device`, `--out`, `--seed`, `--resume`

## `train/train_tv.py`

torchvision 分割基线（DeepLabv3+ / LRASPP / FCN），与 train_unet.py 同数据同设置。

Options: `--arch`, `--manifest`, `--val-manifest`, `--epochs`, `--crop`, `--batch`, `--lr`, `--pos-weight`, `--limit-train`, `--limit-val`, `--workers`, `--pretrained`, `--device`, `--out`, `--seed`, `--resume`

## `train/train_unet.py`

训练入口：本机预实验用，参数默认值刻意设小，跑通优先于刷分。

Options: `--manifest`, `--val-manifest`, `--epochs`, `--crop`, `--batch`, `--lr`, `--base`, `--pos-weight`, `--limit-train`, `--limit-val`, `--max-iters`, `--workers`, `--amp`, `--device`, `--out`, `--seed`, `--resume`

## `train/unet.py`

模型与数据：轻量 U-Net + 道路切片数据集 + Dice/BCE 组合损失。

Library module (no command-line options).

## `train/watch_m1.py`

守望 E2 队列完成后自动接力 M1（2026-10-30 补挂，替代已死掉的 run_ladder）。

Library module (no command-line options).
