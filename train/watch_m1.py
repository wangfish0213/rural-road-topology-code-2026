"""守望 E2 队列完成后自动接力 M1（2026-10-30 补挂，替代已死掉的 run_ladder）。

轮询 state_E2.json：当 e2_deeplab_mv3 出现 eval_rc 字段（无论成败）即认为
E2 队列已走完，启动 M1 队列。12 小时超时保护。
"""
import json
import subprocess
import sys
import time

STATE = r"F:\e\科研\论文发表\第六篇论文\runs\state_E2.json"
CWD = r"F:\e\科研\论文发表\第六篇论文\train"
PY = r"F:\pyenvs\road\Scripts\python.exe"
DEADLINE = time.time() + 12 * 3600

while time.time() < DEADLINE:
    try:
        with open(STATE, encoding="utf-8") as f:
            d = json.load(f)
        e = d.get("e2_deeplab_mv3", {})
        if "eval_rc" in e:
            print(f"[watch] E2 队列已结束（deeplab eval_rc={e['eval_rc']}），启动 M1", flush=True)
            rc = subprocess.run(
                [PY, "-u", "run_queue.py", "--stage", "M1", "--epochs", "12",
                 "--limit-train", "5435", "--eval-limit", "3000", "--job-delay", "15"],
                cwd=CWD,
            ).returncode
            print(f"[watch] M1 队列结束 rc={rc}", flush=True)
            break
        u = d.get("e2_unet", {})
        print(f"[watch] 等待中… e2_unet={ {k: u.get(k) for k in ('train_done', 'pred_rc', 'eval_rc')} }", flush=True)
    except Exception as ex:  # 状态文件被写一半时 json 可能读失败，忽略等下一轮
        print(f"[watch] 读取异常（忽略）: {ex}", flush=True)
    time.sleep(300)
else:
    print("[watch] 超时 12h 退出，M1 未启动", flush=True)
