# -*- coding: utf-8 -*-
"""接力脚本：等当前队列跑完，再依次启动后续 stage。

为什么要这个：GPU 只有一块（2 GB），队列之间不能并行，但人也未必盯到半夜。
这里轮询"当前队列是否结束"的标记，结束后自动接下一个 stage，实现无人值守的连续实验。

用法：
    python run_ladder.py --wait-log ../runs/queue_A_stdout.log --stages E1a E2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def wait_for(log_path: str, marker: str, timeout_h: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                if marker in f.read():
                    return True
        time.sleep(60)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-log", required=True, help="被等待的队列输出日志")
    ap.add_argument("--marker", default="队列完成")
    ap.add_argument("--timeout-h", type=float, default=12.0)
    ap.add_argument("--stages", nargs="+", default=["E1a", "E2"])
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args(argv)

    print(f"等待 {args.wait_log} 出现「{args.marker}」，最多等 {args.timeout_h} 小时", flush=True)
    if not wait_for(args.wait_log, args.marker, args.timeout_h):
        print("等待超时，接力取消（请手动检查前一个队列）")
        return 1

    print("前一个队列已结束，开始接力", flush=True)
    for st in args.stages:
        cmd = [PY, "-u", "run_queue.py", "--stage", st] + args.extra_args
        print(f"\n>>> 启动 stage={st}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=HERE).returncode
        print(f"<<< stage={st} 结束 rc={rc}", flush=True)
        if rc != 0:
            print("该 stage 失败，继续下一个", flush=True)
    print("接力全部完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
