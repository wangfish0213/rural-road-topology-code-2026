# -*- coding: utf-8 -*-
"""M1 训练+推理完成后的评测收尾（2026-10-30）。

背景：经 run_queue 的 run() 链条启动的 multiprocessing 评测在本机会挂死
（0 worker + system CPU 空转）；从会话直接启动则正常。因此：
  1. 等 M1 训练完成（train.log 12 个轮次标记）且推理完成（pred 3000 张）；
  2. 杀掉 M1 队列与任何挂死的 eval 进程（防止抢跑/重复）；
  3. 手动依次跑 deeplab 评测、M1 评测（两种结果文件落位后，RESULTS 汇总即可用）。
日志：runs/post_m1.log
"""
import os
import subprocess
import sys
import time

import psutil

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"F:\e\科研\论文发表\第六篇论文"
TRAIN_LOG = os.path.join(ROOT, r"runs\m1_ori_e1b\train.log")
PRED_DIR = os.path.join(ROOT, r"runs\m1_ori_e1b\pred")
PY = r"F:\pyenvs\road\Scripts\python.exe"
EVAL_CWD = os.path.join(ROOT, "eval")
DEADLINE = time.time() + 4 * 3600


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_done():
    try:
        with open(TRAIN_LOG, encoding="utf-8", errors="ignore") as f:
            return sum(1 for ln in f if ln.startswith("epoch ") and "完成" in ln)
    except OSError:
        return 0


def kill(needle):
    """杀掉命令行含 needle 的进程（排除自己）。"""
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            if needle in " ".join(p.info["cmdline"] or []):
                log(f"杀进程 {p.info['pid']} ({needle})")
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def run_eval(gt_tag, pred_sub, out_sub):
    cmd = [PY, "-u", "eval_dir.py",
           "--gt", os.path.join(ROOT, "datasets", "WHU-RRoad", "test"),
           "--pred", os.path.join(ROOT, "runs", pred_sub),
           "--gt-suffix", "_mask.png",
           "--out", os.path.join(ROOT, "runs", out_sub),
           "--gsd", "0.3", "--workers", "6", "--limit", "3000"]
    log(f"开始评测 {pred_sub}")
    rc = subprocess.run(cmd, cwd=EVAL_CWD).returncode
    log(f"评测 {pred_sub} 结束 rc={rc}")
    return rc


# ---- 阶段 1：等待 M1 训练+推理完成 ----
while time.time() < DEADLINE:
    d = train_done()
    n = len(os.listdir(PRED_DIR)) if os.path.isdir(PRED_DIR) else 0
    if d >= 12 and n >= 3000:
        log(f"M1 训练 {d}/12、推理 {n}/3000 已完成，进入收尾")
        break
    log(f"等待中… 训练 {d}/12，推理 {n}/3000")
    time.sleep(180)
else:
    log("超时 4h 退出")
    sys.exit(1)

# ---- 阶段 2：清理队列与可能挂死的评测 ----
kill("run_queue.py --stage M1")
kill("eval_dir.py")
time.sleep(10)

# ---- 阶段 3：依次评测 ----
run_eval("deeplab", r"e2_deeplab_mv3\pred", r"e2_deeplab_mv3\eval")
run_eval("m1", r"m1_ori_e1b\pred", r"m1_ori_e1b\eval")
log("全部评测收尾完成")
