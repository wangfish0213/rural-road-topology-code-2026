# -*- coding: utf-8 -*-
"""模型与数据：轻量 U-Net + 道路切片数据集 + Dice/BCE 组合损失。

设计取舍：
  - 参数量约 1.9M（base=32），目的是让 2 GB 显存的 MX450 也能跑起来；
  - 数据只按 CSV 索引读取，训练/验证/测试共用同一份 manifest，避免口径不一致；
  - 损失用 BCE(pos_weight) + Dice：前景只占 2.3%，纯 BCE 会退化成全背景。
"""

from __future__ import annotations

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class RoadTile(Dataset):
    """读取 manifest 中的 1024 切片；训练时随机裁剪 + 翻转/旋转，验证时中心裁剪。"""

    def __init__(self, manifest: str, crop: int = 256, limit: int | None = None,
                 train: bool = True, cache: bool = False):
        with open(manifest, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if limit:
            rows = rows[:limit]
        self.rows = rows
        self.crop = crop
        self.train = train
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def _load(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]
        r = self.rows[idx]
        img = np.array(Image.open(r["image"]).convert("RGB"))
        msk = np.array(Image.open(r["mask"]))
        if self.cache:
            self._cache[idx] = (img, msk)
        return img, msk

    def __getitem__(self, idx):
        img, msk = self._load(idx)
        h, w = msk.shape[:2]
        c = self.crop
        if self.train:
            y = random.randint(0, max(0, h - c))
            x = random.randint(0, max(0, w - c))
        else:
            y, x = max(0, (h - c) // 2), max(0, (w - c) // 2)
        img = np.ascontiguousarray(img[y:y + c, x:x + c])
        msk = np.ascontiguousarray(msk[y:y + c, x:x + c])

        if self.train:
            if random.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = random.randint(0, 3)
            if k:
                img, msk = np.rot90(img, k), np.rot90(msk, k)

        img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
        msk = (msk > 0).astype(np.float32)[None]
        return torch.from_numpy(img), torch.from_numpy(msk)


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4 层下采样的标准 U-Net，通道 32/64/128/256。"""

    def __init__(self, cin: int = 3, cout: int = 1, base: int = 32):
        super().__init__()
        b = base
        self.e1 = conv_block(cin, b)
        self.e2 = conv_block(b, b * 2)
        self.e3 = conv_block(b * 2, b * 4)
        self.e4 = conv_block(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.c3 = conv_block(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.c2 = conv_block(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.c1 = conv_block(b * 2, b)
        self.head = nn.Conv2d(b, cout, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        s4 = self.e4(F.max_pool2d(s3, 2))
        x = self.c3(torch.cat([self.up3(s4), s3], 1))
        x = self.c2(torch.cat([self.up2(x), s2], 1))
        x = self.c1(torch.cat([self.up1(x), s1], 1))
        return self.head(x)


class BCEDiceLoss(nn.Module):
    """BCE（前景加权）+ Dice，兼顾像素级稳定与稀疏前景的重叠度。"""

    def __init__(self, pos_weight: float = 6.0, dice_w: float = 1.0, bce_w: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)   # 用普通属性，避免 buffer 与输入设备不一致
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.eps = eps

    def forward(self, logits, target):
        pw = torch.as_tensor([self.pos_weight], dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        p = torch.sigmoid(logits)
        num = 2 * (p * target).sum(dim=(1, 2, 3)) + self.eps
        den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
        dice = (1 - num / den).mean()
        return self.bce_w * bce + self.dice_w * dice


@torch.no_grad()
def pixel_scores(logits, target, thr: float = 0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return iou, f1, prec, rec
