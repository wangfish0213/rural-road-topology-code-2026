# -*- coding: utf-8 -*-
"""创新点 1：方向场（orientation field）辅助的中心线提取。

思路：道路掩膜经骨架化后出现毛刺与伪分叉，根源是网络只学了"哪里是路"，
没学"路往哪走"。这里加一路方向场回归（用 cos2θ / sin2θ 两通道表示，
避免 θ 在 0/π 处跳变），骨架化后按方向一致性剪掉毛刺。

本文件提供三样东西：
  1. UNetOri   —— 掩膜头 + 方向场头的多任务网络；
  2. orientation_from_mask —— 从真值掩膜生成方向场监督（骨架邻域 PCA）；
  3. prune_spurs —— 骨架毛刺剪除（按分支长度与方向偏差）。
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from unet import UNet, conv_block

# 复用 eval/metrics 里的骨架化实现，保证训练与评测口径完全一致
_EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval")
if os.path.isdir(_EVAL_DIR) and _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from metrics import skeletonize  # noqa: E402


# ---------------------------------------------------------------- 网络

class UNetOri(UNet):
    """在 U-Net 解码末端接两个头：掩膜（1 通道）与方向场（2 通道）。"""

    def __init__(self, cin: int = 3, base: int = 32):
        super().__init__(cin=cin, cout=1, base=base)
        b = base
        self.ori_head = nn.Sequential(
            conv_block(b, b),
            nn.Conv2d(b, 2, 1),
        )

    def forward(self, x):
        # 复用 UNet 的编码器，但需要解码特征，所以这里重写一遍
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        s4 = self.e4(F.max_pool2d(s3, 2))
        d = self.c3(torch.cat([self.up3(s4), s3], 1))
        d = self.c2(torch.cat([self.up2(d), s2], 1))
        d = self.c1(torch.cat([self.up1(d), s1], 1))
        return self.head(d), self.ori_head(d)


# ---------------------------------------------------------------- 监督生成

def orientation_from_mask(mask: np.ndarray, rad: int = 6, expand: bool = True, skel=None):
    """由二值道路掩膜生成方向场监督（结构张量法，全矢量化，约 0.1 s/1024 图）。

    做法：骨架化 → 以高斯窗对骨架点求二阶矩（Sxx, Syy, Sxy）→ 主方向
    θ = 0.5·atan2(2Sxy, Sxx−Syy) → 转成 (cos2θ, sin2θ) 两通道，避免 0/π 跳变。
    与逐点 PCA 等价，但不用循环，速度提升十几倍。

    返回 (cos2θ, sin2θ, weight)，weight 为道路像素（可监督区域）。
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        z = np.zeros(m.shape, np.float32)
        return z, z.copy(), z.copy()

    sk = (skeletonize(m) if skel is None else skel).astype(np.float32)
    sigma = max(1.0, rad / 2.0)
    ax = np.arange(-rad, rad + 1, dtype=np.float32)
    X, Y = np.meshgrid(ax, ax)
    G = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    sxx = ndi.convolve(sk, G * X * X, mode="constant")
    syy = ndi.convolve(sk, G * Y * Y, mode="constant")
    sxy = ndi.convolve(sk, G * X * Y, mode="constant")

    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)      # 主方向（自 x 轴起，逆时针）
    cos2 = np.cos(2 * theta).astype(np.float32)
    sin2 = np.sin(2 * theta).astype(np.float32)

    # 骨架外区域按最近骨架点的方向补齐，保证每条路上都有监督
    if expand:
        _, (iy, ix) = ndi.distance_transform_edt(~sk.astype(bool), return_indices=True)
        cos2, sin2 = cos2[iy, ix], sin2[iy, ix]

    weight = m.astype(np.float32)
    return cos2, sin2, weight


def ori_loss(pred: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor,
             weight: torch.Tensor) -> torch.Tensor:
    """方向场损失：按权重求 1 - cos(角度差)，等价于两通道向量的余弦相似度。"""
    p = F.normalize(pred, dim=1, eps=1e-6)
    t = torch.cat([cos_t, sin_t], dim=1)
    t = F.normalize(t, dim=1, eps=1e-6)
    sim = (p * t).sum(dim=1, keepdim=True)            # cos(2Δθ)
    loss = (1 - sim) * weight
    denom = weight.sum().clamp_min(1.0)
    return loss.sum() / denom


# ---------------------------------------------------------------- 骨架剪枝

def prune_spurs(skel: np.ndarray, max_len: int = 6, ori=None, max_dev_deg: float = 50.0) -> np.ndarray:
    """剪掉骨架上的短毛刺。

    max_len：短于该长度（像素）的端点分支判为毛刺；
    ori：可选的 (cos2θ, sin2θ) 场，给定时再判断分支方向与局部走向的偏差，
         偏差小于 max_dev_deg 的分支保留（可能是真路的末端）。
    """
    s = skel.astype(np.uint8).copy()
    if s.sum() == 0:
        return s.astype(bool)

    def neighbours(y, x):
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < s.shape[0] and 0 <= xx < s.shape[1] and s[yy, xx]:
                    out.append((yy, xx))
        return out

    for _ in range(64):  # 反复剪，直到没有新的短端点分支
        ys, xs = np.nonzero(s)
        ends = [(y, x) for y, x in zip(ys, xs) if len(neighbours(y, x)) == 1]
        if not ends:
            break
        removed = 0
        for y, x in ends:
            if not s[y, x]:
                continue
            path = [(y, x)]
            prev, cur = None, (y, x)
            while len(path) <= max_len:
                nbs = [n for n in neighbours(*cur) if n != prev]
                if len(nbs) != 1:
                    break
                prev, cur = cur, nbs[0]
                path.append(cur)
                if len(neighbours(*cur)) != 2:   # 到达交叉点或端点，停
                    break
            if len(path) - 1 > max_len:
                continue
            if ori is not None and len(path) >= 3:
                c2, s2 = ori
                y0, x0 = path[0]
                y1, x1 = path[-1]
                theta = math.atan2(y1 - y0, x1 - x0)
                # 取分叉点处的方向：端点处方向被毛刺自身带偏，不可用
                jy, jx = path[-1]
                pc, ps = float(c2[jy, jx]), float(s2[jy, jx])
                if abs(pc) + abs(ps) > 1e-3:
                    dev = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(2 * (theta - math.atan2(ps, pc) / 2))))))
                    if dev < max_dev_deg:
                        continue            # 方向与局部走向一致，认为是真路末端
            for yy, xx in path[:-1]:    # 保留连接点
                s[yy, xx] = 0
                removed += 1
        if removed == 0:
            break
    return s.astype(bool)
