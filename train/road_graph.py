# -*- coding: utf-8 -*-
"""骨架 → 道路图：节点、边链、候选连接的生成。

为什么需要这一步：掩膜骨架化只给出像素集合，"哪里该连、哪里该断"是图上的决策。
本文件把骨架变成图（节点=端点/交叉点，边=两节点之间的像素链），
再枚举"可能该连的端点对"，交给语义判别器（correction.py）决定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class Node:
    idx: int
    y: int
    x: int
    deg: int
    kind: str                      # "endpoint" / "junction"
    dirs: list = field(default_factory=list)   # 出边方向（单位向量，行/列分量）


@dataclass
class Edge:
    a: int
    b: int
    path: list                     # [(y, x), ...]
    length: float


def degree_map(skel: np.ndarray) -> np.ndarray:
    """每个骨架像素的 8 邻域骨架像素数。"""
    s = skel.astype(np.uint8)
    p = np.pad(s, 1)
    h, w = s.shape
    deg = np.zeros_like(s, dtype=np.uint8)
    for dy, dx in N8:
        deg += p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return deg * s


def _build_crossing_lut():
    """交叉数（Rutovitz A）：8 邻域环序中 0→1 的跳变次数。

    为什么不用"邻居个数"判交叉：8 连通骨架走斜线时会出现 (东, 东南, 南)
    这类三点相邻的台阶，邻居数为 3 但其实只有 1 个分支，会造出大量假交叉点
    （实测真实掩膜上假交叉点可达 452 个，平均边长只有 4 px）。
    交叉数按环序数跳变，台阶只会算 1 个分支，是标准做法。
    """
    lut = np.zeros(256, dtype=np.uint8)
    for idx in range(256):
        n = [(idx >> k) & 1 for k in range(8)]
        seq = n + [n[0]]
        lut[idx] = sum(1 for i in range(8) if seq[i] == 0 and seq[i + 1] == 1)
    return lut


_CROSS_LUT = _build_crossing_lut()


def crossing_map(skel: np.ndarray) -> np.ndarray:
    """每个骨架像素的交叉数：1=端点/普通路径，2=路径，>=3=真交叉点。"""
    s = skel.astype(np.uint8)
    p = np.pad(s, 1)
    h, w = s.shape
    idx = (p[0:h, 1:w + 1]
           | (p[0:h, 2:w + 2] << 1)
           | (p[1:h + 1, 2:w + 2] << 2)
           | (p[2:h + 2, 2:w + 2] << 3)
           | (p[2:h + 2, 1:w + 1] << 4)
           | (p[2:h + 2, 0:w] << 5)
           | (p[1:h + 1, 0:w] << 6)
           | (p[0:h, 0:w] << 7)).astype(np.uint8)
    return (_CROSS_LUT[idx] * s).astype(np.uint8)


def skeleton_to_graph(skel: np.ndarray):
    """返回 (nodes, edges)。节点取端点与交叉点，度数为 2 的像素压进边链。"""
    skel = skel.astype(bool)
    if skel.sum() == 0:
        return [], []

    deg = degree_map(skel)
    cross = crossing_map(skel)
    ys, xs = np.nonzero(skel)
    coords = set(zip(ys.tolist(), xs.tolist()))
    # 节点 = 端点（邻居数 1）或真交叉点（交叉数 >= 3）；斜向台阶不再误判
    node_px = {(y, x) for y, x in coords if deg[y, x] == 1 or cross[y, x] >= 3}

    # 纯环（整条没有端点/交叉点）时，任取一点当节点，避免图丢失
    if not node_px:
        y, x = next(iter(coords))
        node_px = {(y, x)}

    nodes, index_of = [], {}
    for y, x in sorted(node_px):
        index_of[(y, x)] = len(nodes)
        kind = "endpoint" if deg[y, x] == 1 else "junction"
        nodes.append(Node(len(nodes), y, x, int(deg[y, x]), kind))

    def nbrs(y, x):
        out = []
        for dy, dx in N8:
            p = (y + dy, x + dx)
            if p in coords:
                out.append(p)
        return out

    edges, seen = [], set()
    for n in nodes:
        for start in nbrs(n.y, n.x):
            key = (min((n.y, n.x), start), max((n.y, n.x), start))
            if key in seen:
                continue
            path = [(n.y, n.x), start]
            seen.add(key)
            visited = {(n.y, n.x), start}
            prev, cur = (n.y, n.x), start
            while cur not in node_px:                 # 沿度数为 2 的链走到下一个节点
                # 必须排除已访问像素：闭合环路没有端点，否则会无限绕圈（曾导致 MemoryError）
                nxt = [p for p in nbrs(*cur) if p != prev and p not in visited]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                visited.add(cur)
                path.append(cur)
                seen.add((min(prev, cur), max(prev, cur)))
                if len(path) > 20000:                 # 保险丝，防异常图
                    break
            if cur in index_of:
                length = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
                edges.append(Edge(index_of[(n.y, n.x)], index_of[cur], path, length))
                vy, vx = path[1][0] - n.y, path[1][1] - n.x
                norm = math.hypot(vy, vx) or 1.0
                nodes[index_of[(n.y, n.x)]].dirs.append((vy / norm, vx / norm))

    return nodes, edges


def endpoint_pairs(nodes, max_dist: float = 40.0):
    """枚举距离阈值内的端点对——它们才是"可能该连"的候选。

    伪连接的根源就在这一步：隔着田埂的两条路端点也很近，
    所以后面必须靠影像语义判别，不能只靠距离阈值。
    """
    eps = [n for n in nodes if n.kind == "endpoint"]
    pairs = []
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            a, b = eps[i], eps[j]
            d = math.hypot(a.y - b.y, a.x - b.x)
            if 1.0 < d <= max_dist:
                pairs.append((a, b, d))
    return pairs


def sample_line(y0, x0, y1, x1, n: int = 24):
    """在两点连线上等距采样，返回整型坐标。"""
    ts = np.linspace(0.0, 1.0, n)
    ys = np.clip(np.round(y0 + (y1 - y0) * ts).astype(int), 0, 10 ** 9)
    xs = np.clip(np.round(x0 + (x1 - x0) * ts).astype(int), 0, 10 ** 9)
    return ys, xs


def line_coverage(mask: np.ndarray, y0, x0, y1, x1, pad: int = 2) -> float:
    """两像素连线上落在 mask（允许 pad 像素膨胀）内的比例，衡量"连起来像不像路"。"""
    from scipy import ndimage as ndi

    yy, xx = sample_line(y0, x0, y1, x1)
    h, w = mask.shape
    ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    yy, xx = yy[ok], xx[ok]
    if len(yy) == 0:
        return 0.0
    m = mask.astype(bool)
    if pad > 0:
        m = ndi.binary_dilation(m, iterations=pad)
    return float(m[yy, xx].mean())
