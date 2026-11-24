# -*- coding: utf-8 -*-
"""合成样例自测：不依赖任何真实数据，验证指标实现的正确性与方向性。

跑法：python selfcheck.py
全部断言通过则打印 PASS，并输出每个用例的指标表，便于人工核对数值是否合理。
"""

from __future__ import annotations

import sys

import numpy as np

from metrics import EvalConfig, evaluate, skeletonize


def blank(h=200, w=300):
    return np.zeros((h, w), dtype=np.uint8)


def hline(img, y, x0, x1, t=3):
    img[y - t // 2:y + t // 2 + 1, x0:x1] = 255
    return img


def vline(img, x, y0, y1, t=3):
    img[y0:y1, x - t // 2:x + t // 2 + 1] = 255
    return img


def case_identical():
    gt = blank()
    hline(gt, 100, 20, 280)
    vline(gt, 150, 40, 160)
    pred = gt.copy()
    return gt, pred, {"iou": (0.999, 1.001), "apls": (0.99, 1.001),
                      "breakage_rate": (-1e-9, 1e-9), "false_connection_rate": (-1e-9, 1e-9)}


def case_gap():
    gt = blank()
    hline(gt, 100, 20, 280)
    pred = gt.copy()
    pred[95:106, 140:165] = 0          # 中间挖断 25 px
    return gt, pred, {"breakage_rate": (0.5, 1.001), "apls": (0.0, 0.95),
                      "line_recall": (0.5, 1.0)}


def case_false_link():
    gt = blank()
    hline(gt, 80, 20, 280)
    hline(gt, 140, 20, 280)
    pred = gt.copy()
    vline(pred, 150, 80, 141)          # 凭空连一条竖线，把两条平行路串起来
    return gt, pred, {"false_connection_rate": (0.5, 1.001), "breakage_rate": (-1e-9, 1e-9)}


def case_shift_2px():
    gt = blank()
    hline(gt, 100, 20, 280)
    pred = np.zeros_like(gt)
    pred[97:103, 20:280] = 255          # 整体上移 2 px
    return gt, pred, {"line_recall": (0.95, 1.001), "iou": (0.3, 0.8)}


def case_empty_pred():
    gt = blank()
    hline(gt, 100, 20, 280)
    pred = blank()
    return gt, pred, {"iou": (-1e-9, 1e-9), "apls": (-1e-9, 1e-9)}


def case_cross():
    gt = blank()
    hline(gt, 100, 20, 280)
    vline(gt, 150, 20, 180)
    pred = gt.copy()
    return gt, pred, {"iou": (0.999, 1.001), "apls": (0.99, 1.001)}


CASES = [
    ("完全一致", case_identical),
    ("中间断裂 25px", case_gap),
    ("伪连接（两平行路被串）", case_false_link),
    ("整体偏移 2px（容差内）", case_shift_2px),
    ("预测全空", case_empty_pred),
    ("十字交叉一致", case_cross),
]

COLS = ["iou", "line_precision", "line_recall", "apls", "breakage_rate", "false_connection_rate", "cc_ratio"]


def main() -> int:
    # 骨架化自检：细化后每行/列不应出现 2 像素以上厚度
    probe = np.zeros((50, 50), dtype=bool)
    probe[20:30, 10:40] = True
    sk = skeletonize(probe)
    assert sk.sum() < probe.sum() / 3, "骨架化未显著细化"

    cfg = EvalConfig(gsd=0.3, tol_px=3.0, apls_samples=40, seed=0)
    failures = []

    print(f"{'用例':<24}" + "".join(f"{c:>16}" for c in COLS))
    print("-" * (24 + 16 * len(COLS)))
    for name, fn in CASES:
        gt, pred, expect = fn()
        m = evaluate(pred, gt, cfg)
        row = "".join(f"{m[c]:>16.3f}" for c in COLS)
        print(f"{name:<24}{row}")
        for key, (lo, hi) in expect.items():
            val = m[key]
            if not (lo <= val <= hi):
                failures.append(f"[{name}] {key}={val:.3f} 不在 [{lo}, {hi}]")
        # 通用一致性：所有比率型指标落在 [0,1]
        for key in ("iou", "f1", "line_precision", "line_recall", "apls",
                    "breakage_rate", "false_connection_rate"):
            if not (-1e-9 <= m[key] <= 1 + 1e-9):
                failures.append(f"[{name}] {key}={m[key]:.3f} 越界")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS：6 个用例全部符合预期")
    return 0


if __name__ == "__main__":
    sys.exit(main())
