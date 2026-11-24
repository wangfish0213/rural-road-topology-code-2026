# -*- coding: utf-8 -*-
"""串行实验队列：训练 → 推理 → 评测，一个作业接一个作业跑。

本机只有 2 GB 显存，并行必 OOM，所以严格串行。断点续跑靠"权重已存在则跳过"实现，
中断后重跑同一条命令即可接着往下走。

用法：
    python run_queue.py --stage A              # 三个基线（E1 跨域）
    python run_queue.py --stage A --eval-limit 3000
    python run_queue.py --stage R50 --eval-limit 3000   # 后补的 ResNet-50 对照
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
GT_DIR = os.path.join(ROOT, "datasets", "WHU-RRoad", "test")
TEST_MANIFEST = os.path.join(ROOT, "manifests", "whu_rroad_test.csv")
TRAIN_MANIFEST = os.path.join(ROOT, "manifests", "whu_road_1m_train.csv")
VAL_MANIFEST = os.path.join(ROOT, "manifests", "whu_road_1m_val.csv")
# E1a：城市源域（WHU Road 2m 子集，378 训练 / 60 验证）
URBAN_TRAIN = os.path.join(ROOT, "manifests", "whu_road_2m_train.csv")
URBAN_VAL = os.path.join(ROOT, "manifests", "whu_road_2m_val.csv")
# E2：同域（WHU-RRoad 自身划分）
RR_TRAIN = os.path.join(ROOT, "manifests", "whu_rroad_train.csv")
# E2 正确做法：训练集再切 85/15，验证集独立，测试集只用于最终评测
# （原先直接拿测试集当 val 选模型，属数据泄漏，2026-10-30 修正）
RR_SPLIT_TRAIN = os.path.join(ROOT, "manifests", "whu_rroad_split_train.csv")
RR_SPLIT_VAL = os.path.join(ROOT, "manifests", "whu_rroad_split_val.csv")


def epochs_done(out_dir: str) -> int:
    """从 train.log 数已完成的轮数（训练脚本每轮打印一次 "epoch N 完成"）。"""
    p = os.path.join(out_dir, "train.log")
    if not os.path.exists(p):
        return 0
    n = 0
    with open(p, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("epoch ") and "完成" in line:
                n += 1
    return n


def target_epochs(cmd) -> int:
    return int(cmd[cmd.index("--epochs") + 1]) if "--epochs" in cmd else 0


def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(path: str, st: dict):
    """写状态文件。Windows 上 os.replace 可能被杀毒/索引占用而报 PermissionError，
    因此这里必须容错：状态记录失败绝不能中断实验队列（已发生过一次，损失两个作业）。"""
    text = json.dumps(st, ensure_ascii=False, indent=2)
    tmp = path + ".tmp"
    for attempt in range(3):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
            return
        except (PermissionError, OSError):
            time.sleep(0.5)
    try:                              # 最后兜底：直接覆盖写，放弃原子性
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        print(f"  警告：状态文件写入失败（不影响实验）：{e}", flush=True)


def job(name, train_cmd, limits):
    return {"name": name, "train": train_cmd, **limits}


def build_stage(stage: str, args):
    common = ["--epochs", str(args.epochs), "--limit-train", str(args.limit_train),
              "--batch", "4", "--crop", "256", "--device", "cuda"]
    if stage == "A":
        return [
            job("e1_unet",
                [PY, "-u", "train_unet.py", "--manifest", TRAIN_MANIFEST,
                 "--val-manifest", VAL_MANIFEST] + common,
                {}),
            job("e1_deeplab_mv3",
                [PY, "-u", "train_tv.py", "--arch", "deeplabv3_mobilenet_v3_large",
                 "--manifest", TRAIN_MANIFEST, "--val-manifest", VAL_MANIFEST] + common,
                {}),
            job("e1_lraspp_mv3",
                [PY, "-u", "train_tv.py", "--arch", "lraspp_mobilenet_v3_large",
                 "--manifest", TRAIN_MANIFEST, "--val-manifest", VAL_MANIFEST] + common,
                {}),
        ]
    if stage == "E1a":
        # 城市(2m) → 农村，样本少所以轮次给足
        a = ["--epochs", str(args.e1a_epochs), "--limit-train", "378",
             "--batch", "4", "--crop", "256", "--device", "cuda"]
        return [
            job("e1a_unet",
                [PY, "-u", "train_unet.py", "--manifest", URBAN_TRAIN,
                 "--val-manifest", URBAN_VAL] + a, {}),
            job("e1a_deeplab_mv3",
                [PY, "-u", "train_tv.py", "--arch", "deeplabv3_mobilenet_v3_large",
                 "--manifest", URBAN_TRAIN, "--val-manifest", URBAN_VAL] + a, {}),
            job("e1a_lraspp_mv3",
                [PY, "-u", "train_tv.py", "--arch", "lraspp_mobilenet_v3_large",
                 "--manifest", URBAN_TRAIN, "--val-manifest", URBAN_VAL] + a, {}),
        ]
    if stage == "E2":
        # 同域参考：在 WHU-RRoad 自身训练集上训，测试集自然同域，用来量"掉多少"
        e = ["--epochs", str(args.epochs), "--limit-train", str(args.e2_limit),
             "--batch", "4", "--crop", "256", "--device", "cuda"]
        return [
            job("e2_unet",
                [PY, "-u", "train_unet.py", "--manifest", RR_SPLIT_TRAIN,
                 "--val-manifest", RR_SPLIT_VAL, "--limit-val", "150"] + e, {}),
            job("e2_deeplab_mv3",
                [PY, "-u", "train_tv.py", "--arch", "deeplabv3_mobilenet_v3_large",
                 "--manifest", RR_SPLIT_TRAIN, "--val-manifest", RR_SPLIT_VAL,
                 "--limit-val", "150"] + e, {}),
        ]
    if stage == "M1":
        # 方法版：U-Net + 方向场头（创新点 1），与 E1b 同数据同轮数，直接可比
        m = ["--epochs", str(args.epochs), "--limit-train", str(args.limit_train),
             "--batch", "4", "--crop", "256", "--device", "cuda", "--ori-weight", "0.5",
             "--workers", "2"]
        return [
            job("m1_ori_e1b",
                [PY, "-u", "train_ori.py", "--manifest", TRAIN_MANIFEST,
                 "--val-manifest", VAL_MANIFEST] + m, {}),
        ]
    if stage == "R50":
        return [
            job("e1_deeplab_r50_pre",
                [PY, "-u", "train_tv.py", "--arch", "deeplabv3_resnet50", "--pretrained",
                 "--manifest", TRAIN_MANIFEST, "--val-manifest", VAL_MANIFEST,
                 "--epochs", str(args.r50_epochs), "--limit-train", str(args.r50_limit),
                 "--batch", "2", "--crop", "256", "--device", "cuda"],
                {}),
        ]
    raise SystemExit(f"未知 stage: {stage}")


def run(cmd, log_path, cwd=HERE):
    """执行子进程并把输出写进日志。

    注意：打开日志这一步也可能被杀毒/索引占用而 PermissionError，
    因此必须先重试打开；否则错误发生在子进程启动之前，业务层的重试逻辑根本管不到
    （2026-10-30 就是这样连崩两次的）。
    """
    f = None
    for attempt in range(5):
        try:
            f = open(log_path, "a", encoding="utf-8")
            break
        except PermissionError:
            time.sleep(1.0 + attempt)
    if f is None:
        print(f"  无法打开日志 {log_path}，改用不落盘模式执行", flush=True)
        p = subprocess.run(cmd, cwd=cwd)
        return p.returncode

    with f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", choices=["A", "M1", "E1a", "E2", "R50"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--limit-train", type=int, default=5435)
    ap.add_argument("--e1a-epochs", type=int, default=30)
    ap.add_argument("--e2-limit", type=int, default=5000)
    ap.add_argument("--r50-epochs", type=int, default=8)
    ap.add_argument("--r50-limit", type=int, default=2000)
    ap.add_argument("--eval-limit", type=int, default=3000)
    ap.add_argument("--job-delay", type=float, default=15.0,
                    help="作业之间强制等待秒数；Windows 上 torch 的 shm.dll 会被上一个进程占用，抢跑会直接崩")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args(argv)

    os.makedirs(RUNS, exist_ok=True)
    queue_log = os.path.join(RUNS, f"queue_{args.stage}.log")
    state_path = os.path.join(RUNS, f"state_{args.stage}.json")
    state = load_state(state_path)
    jobs = build_stage(args.stage, args)

    print(f"队列 stage={args.stage}，共 {len(jobs)} 个作业；日志 {queue_log}")
    t_all = time.time()
    for k, j in enumerate(jobs, 1):
        out = os.path.join(RUNS, j["name"])
        os.makedirs(out, exist_ok=True)
        best = os.path.join(out, "best.pt")
        print(f"\n[{k}/{len(jobs)}] {j['name']}", flush=True)

        done, target = epochs_done(out), target_epochs(j["train"])
        if args.skip_existing and os.path.exists(best) and done >= target:
            print(f"  训练已完成（{done}/{target} 轮），跳过")
        else:
            if done:
                print(f"  检测到已完成 {done}/{target} 轮，从断点续训")
            t0 = time.time()
            rc = run(j["train"] + ["--out", out, "--resume"], os.path.join(out, "train.log"))
            if rc != 0:
                print(f"  训练返回 {rc}，等待 30 s 后重试一次（多为 Windows 文件锁/DLL 占用）", flush=True)
                time.sleep(30)
                rc = run(j["train"] + ["--out", out, "--resume"], os.path.join(out, "train.log"))
            print(f"  训练结束 rc={rc}，用时 {(time.time()-t0)/3600:.2f} h", flush=True)
            state[j["name"]] = state.get(j["name"], {})
            state[j["name"]].update({"train_done": epochs_done(out), "train_target": target,
                                     "train_rc": rc, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
            save_state(state_path, state)
            if rc != 0:
                print("  训练失败，队列中止（日志见 train.log）")
                return rc

        pred_dir = os.path.join(out, "pred")
        if args.eval_limit and os.path.isdir(pred_dir) and len(os.listdir(pred_dir)) >= args.eval_limit:
            print("  已有预测，跳过推理")
        else:
            t0 = time.time()
            rc = run([PY, "-u", "predict.py", "--ckpt", best, "--manifest", TEST_MANIFEST,
                      "--out", pred_dir, "--limit", str(args.eval_limit), "--batch", "4",
                      "--device", "cuda"], os.path.join(out, "predict.log"))
            print(f"  推理结束 rc={rc}，用时 {(time.time()-t0)/60:.0f} min", flush=True)
            state.setdefault(j["name"], {}).update(
                {"pred_n": len(os.listdir(pred_dir)) if os.path.isdir(pred_dir) else 0,
                 "pred_rc": rc, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
            save_state(state_path, state)

        if os.path.exists(os.path.join(out, "eval", "summary.csv")):
            print("  已有评测结果，跳过评测", flush=True)
            continue

        t0 = time.time()
        rc = run([PY, "-u", os.path.join(ROOT, "eval", "eval_dir.py"),
                  "--gt", GT_DIR, "--pred", pred_dir, "--gt-suffix", "_mask.png",
                  "--out", os.path.join(out, "eval"), "--gsd", "0.3", "--workers", "6",
                  "--limit", str(args.eval_limit)],
                 os.path.join(out, "eval.log"), cwd=os.path.join(ROOT, "eval"))
        print(f"  评测结束 rc={rc}，用时 {(time.time()-t0)/60:.0f} min", flush=True)
        state.setdefault(j["name"], {}).update(
            {"eval_rc": rc, "eval_path": os.path.join(out, "eval", "summary.csv"),
             "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        save_state(state_path, state)
        if rc != 0:
            print("  评测失败，但继续下一个作业（训练成果已保存）")
        if args.job_delay:
            print(f"  等待 {args.job_delay:.0f} s 再启动下一个作业（释放 torch 句柄）", flush=True)
            time.sleep(args.job_delay)

    print(f"\n队列完成，总用时 {(time.time()-t_all)/3600:.2f} h")
    print(f"状态文件：{state_path}")
    print("汇报表：")
    for j in jobs:
        s = os.path.join(RUNS, j["name"], "eval", "summary.csv")
        print(f"  {j['name']:24s} {'有结果' if os.path.exists(s) else '缺结果'}  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
