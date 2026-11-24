# -*- coding: utf-8 -*-
"""论文插图基础库（PIL + numpy，不依赖 matplotlib）。

为什么不用 matplotlib：本机 Python 环境没有 matplotlib，且 pip 安装被中断；
项目既有的 tools/plot_cc_figs.py 已经用 PIL 出图。这里把常用件抽出来，
保证 6 张图字号、配色、比例尺、坐标刻度完全一致。

尺寸约定：期刊通栏宽 16 cm，在 600 dpi 下即 3780 px。所有长度用 cm 指定，
由 cm() 换算成像素，因此图在版面上的物理尺寸与字号都是可预期的。

关于指北针：本研究所用 WHU Road / WHU-RRoad / IRSAMap 切片**不带地理配准**，
没有可用的北向与经纬度。图上只给切片内像素坐标刻度（原点在左上角，x 向右、
y 向下）与由地面采样距离换算的比例尺；不画指北针、不编经纬度，图注里说明。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 600

# 统一配色（RGB）
INK = (26, 30, 36)
MUTED = (110, 118, 128)
RULE = (150, 158, 168)
BG = (255, 255, 255)
PANEL_BG = (246, 247, 249)

C_GT = (255, 214, 0)          # 真值中心线：亮黄
C_PRED = (235, 62, 62)        # 预测中心线：红
C_ADD = (32, 148, 243)        # 纠错新增链接：蓝
C_DEL = (0, 176, 160)         # 被剪除的孤立分量：青绿
C_ACC = (56, 176, 92)         # 接受的链接
C_REJ = (150, 150, 158)       # 拒绝的候选
C_SPUR = (255, 60, 160)       # 毛刺：洋红
C_RING = (255, 255, 255)      # 标注圈

FONT_CANDIDATES = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
FONT_BOLD_CANDIDATES = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"]

_FONT_CACHE: dict = {}


def cm(value: float) -> int:
    """厘米 → 像素（按 DPI）。"""
    return int(round(value / 2.54 * DPI))


def pt(value: float) -> int:
    """磅 → 像素（按 DPI）。7 pt 在 600 dpi 下约 58 px。"""
    return int(round(value / 72.0 * DPI))


def font(size_pt: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size_pt, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    cands = (FONT_BOLD_CANDIDATES if bold else []) + FONT_CANDIDATES
    px = pt(size_pt)
    for p in cands:
        if Path(p).exists():
            try:
                f = ImageFont.truetype(p, px)
                _FONT_CACHE[key] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


# ---------------------------------------------------------------- 画布

class Canvas:
    """白底画布。坐标单位为像素，原点左上。"""

    def __init__(self, width: int, height: int):
        self.w, self.h = int(width), int(height)
        self.img = Image.new("RGB", (self.w, self.h), BG)
        self.draw = ImageDraw.Draw(self.img)

    # -- 文字
    def text(self, xy, s: str, size: float = 7.0, bold: bool = False,
             fill=INK, anchor: str = "la", spacing: int = 4):
        self.draw.multiline_text(xy, s, font=font(size, bold), fill=fill,
                                 anchor=anchor, spacing=spacing)

    def text_size(self, s: str, size: float, bold: bool = False):
        return self.draw.multiline_textbbox((0, 0), s, font=font(size, bold))[2:]

    # -- 线
    def line(self, xy, fill=INK, width: int = 2):
        self.draw.line(xy, fill=fill, width=width)

    def rect(self, box, outline=None, fill=None, width: int = 2):
        self.draw.rectangle(box, outline=outline, fill=fill, width=width)

    def arrow(self, xy0, xy1, fill=INK, width: int = 4, head: int | None = None):
        """带箭头的直线（head 为箭头长度像素）。"""
        self.draw.line([xy0, xy1], fill=fill, width=width)
        head = head or max(8, width * 4)
        ang = math.atan2(xy1[1] - xy0[1], xy1[0] - xy0[0])
        for s in (+1, -1):
            a = ang + s * math.radians(24)
            self.draw.polygon([xy1,
                               (xy1[0] - head * math.cos(a), xy1[1] - head * math.sin(a)),
                               (xy1[0] - head * math.cos(ang - s * math.radians(24)),
                                xy1[1] - head * math.sin(ang - s * math.radians(24)))],
                              fill=fill)

    # -- 图像
    def paste(self, rgb: np.ndarray, box):
        """把 HxWx3 数组缩放后贴到 box=(x, y, w, h)。"""
        x, y, w, h = box
        im = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)))
        if (im.width, im.height) != (w, h):
            im = im.resize((w, h), Image.LANCZOS)
        self.img.paste(im, (x, y))

    def save(self, path, dpi: int = DPI):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.img.save(path, dpi=(dpi, dpi), optimize=True)
        return path


# ---------------------------------------------------------------- 影像与叠加

def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(str(path)).convert("RGB"))


def load_mask(path, thr: int = 127) -> np.ndarray:
    a = np.asarray(Image.open(str(path)).convert("L"))
    return a > thr


def bin_edges(mask: np.ndarray) -> np.ndarray:
    """取掩膜边界像素（用于把真值画成细轮廓而不是实心块）。"""
    from scipy import ndimage as ndi
    m = mask.astype(bool)
    return m & ~ndi.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    from scipy import ndimage as ndi
    if r <= 0:
        return mask.astype(bool)
    return ndi.binary_dilation(mask.astype(bool), iterations=r)


def blend(base: np.ndarray, mask: np.ndarray, color, alpha: float = 1.0) -> np.ndarray:
    """把 color 以 alpha 混到 base 上 mask 为真的位置。"""
    out = base.astype(np.float32).copy()
    m = mask.astype(bool)
    if not m.any():
        return out.astype(np.uint8)
    c = np.array(color, dtype=np.float32)
    out[m] = out[m] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_outline(base: np.ndarray, mask: np.ndarray, color, width: int = 2,
                  alpha: float = 1.0) -> np.ndarray:
    return blend(base, dilate(bin_edges(mask), width - 1), color, alpha)


def crop(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """以 (cx, cy) 为中心取 size×size 块，越界部分镜像补齐。"""
    h, w = img.shape[:2]
    x0, y0 = int(cx - size // 2), int(cy - size // 2)
    x1, y1 = x0 + size, y0 + size
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - w), max(0, y1 - h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    sub = img[y0:y1, x0:x1]
    pad = [(py0, py1), (px0, px1)] + [(0, 0)] * (img.ndim - 2)
    if any(p[0] or p[1] for p in pad):
        sub = np.pad(sub, pad, mode="reflect") if min(sub.shape[:2]) > 2 else \
            np.pad(sub, pad, mode="edge")
    return sub


# ---------------------------------------------------------------- 版面元素

def nice_length(gsd: float, target_px: float) -> float:
    """在给定地面采样距离下，选一个"整"的比例尺长度（米），使其像素长最接近 target_px。

    候选取 1/2/5 × 10^k 米，按与目标长度的对数距离最小者胜出——不用"第一个不小于"，
    否则从 100 m 会直接跳到 1000 m，比例尺画出来会横贯半个面板。
    """
    raw = target_px * gsd
    cands = [m * (10.0 ** e) for m in (1, 2, 5) for e in range(-2, 5)]
    return min(cands, key=lambda v: abs(math.log(v / raw))) if raw > 0 else 100.0


def scalebar(canvas: "Canvas", x: int, y: int, gsd: float, target_px: float,
             color=(255, 255, 255), edge=(20, 20, 20), label_pt: float = 6.5,
             length_m: float | None = None):
    """在 (x, y) 处画水平比例尺；length_m 给定时用它，否则按 gsd 与 target_px 自动取整。"""
    length_m = length_m if length_m is not None else nice_length(gsd, target_px)
    bar_px = int(round(length_m / gsd))
    h = max(6, pt(1.6))
    # 描边保证在亮/暗影像上都可见
    canvas.rect((x - 2, y - 2, x + bar_px + 2, y + h + 2), fill=edge)
    canvas.rect((x, y, x + bar_px, y + h), fill=color)
    lab = f"{length_m:g} m"
    canvas.text((x + bar_px / 2, y + h + pt(1.2)), lab, size=label_pt, fill=color,
                anchor="ma")
    return bar_px, length_m


def pixel_axis(canvas: "Canvas", box, size_px: int, color=MUTED, label_pt: float = 6.0,
               step: int | None = None, show_x: bool = True, show_y: bool = True):
    """在面板下方／左侧画像素坐标刻度。box=(x, y, w, h) 为图像区域。

    多面板并排时只让最左侧面板画 y 轴（其余 show_y=False），否则相邻面板的刻度
    会压到隔壁图上。
    """
    x, y, w, h = box
    step = step or (200 if size_px >= 1000 else 100)
    tick = pt(0.8)
    if show_x:
        for v in range(0, size_px + 1, step):
            px = x + int(v / size_px * w)
            canvas.line([(px, y + h), (px, y + h + tick)], fill=color, width=2)
            canvas.text((px, y + h + tick + pt(0.4)), str(v), size=label_pt, fill=color,
                        anchor="ma")
    if show_y:
        for v in range(0, size_px + 1, step):
            py = y + int(v / size_px * h)
            canvas.line([(x, py), (x - tick, py)], fill=color, width=2)
            canvas.text((x - tick - pt(0.6), py), str(v), size=label_pt, fill=color,
                        anchor="rm")


def panel_label(canvas: "Canvas", x: int, y: int, s: str, size: float = 7.2):
    """面板角标 (a)/(b)...，白描边黑字，压在影像上也清晰。"""
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        canvas.text((x + dx, y + dy), s, size=size, bold=True, fill=(255, 255, 255))
    canvas.text((x, y), s, size=size, bold=True, fill=INK)


def legend_chip(canvas: "Canvas", x: int, y: int, color, s: str, size: float = 6.6,
                kind: str = "line", outline=(255, 255, 255)):
    """图例项：线段或色块 + 文字。返回占用宽度。"""
    lw = cm(0.55)
    if kind == "line":
        canvas.rect((x - 2, y - 2, x + lw + 2, y + pt(0.9) + 2), fill=outline)
        canvas.rect((x, y, x + lw, y + pt(0.9)), fill=color)
    else:
        canvas.rect((x, y, x + lw, y + lw), fill=color, outline=outline)
    tw = canvas.text_size(s, size)[0]
    canvas.text((x + lw + pt(1.6), y + pt(0.5)), s, size=size, fill=INK, anchor="lm")
    return lw + pt(1.6) + tw


def frame(canvas: "Canvas", box, color=RULE, width: int = 2):
    x, y, w, h = box
    canvas.rect((x - 1, y - 1, x + w, y + h), outline=color, width=width)


def colorbar(canvas: "Canvas", box, cmap, vmin: float, vmax: float, ticks: list,
             title: str = "", label_pt: float = 6.2):
    """竖直色标。cmap(v) -> (r, g, b)；box=(x, y, w, h)。"""
    x, y, w, h = box
    for i in range(h):
        v = vmax - (vmax - vmin) * i / max(1, h - 1)
        canvas.rect((x, y + i, x + w, y + i), fill=cmap(v))
    canvas.rect((x - 1, y - 1, x + w + 1, y + h), outline=RULE, width=2)
    for t in ticks:
        py = y + int((vmax - t) / (vmax - vmin) * h)
        canvas.line([(x + w, py), (x + w + pt(0.7), py)], fill=INK, width=2)
        canvas.text((x + w + pt(1.2), py), f"{t:g}", size=label_pt, fill=INK, anchor="lm")
    if title:
        canvas.text((x + w / 2, y - pt(1.4)), title, size=label_pt, fill=INK, anchor="mb")
