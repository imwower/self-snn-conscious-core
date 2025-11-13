# -*- coding: utf-8 -*-
"""
文本位图渲染 + 脉冲编码（TTFS / rate）
- 字体来源：fonts/*.json，未收录字符回退 '?'
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List


def load_font(path: Path) -> Dict[str, List[List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_text_bitmap(text: str, font: Dict[str, List[List[int]]], pad: int = 1) -> List[List[int]]:
    """
    将字符串渲染为二值位图（0=黑，1=白；输出时转换）。
    """
    glyphs = [font.get(ch, font.get("?", [[0]])) for ch in text]
    if not glyphs:
        return [[1]]
    gh = max(len(g) for g in glyphs)
    width = sum(len(g[0]) for g in glyphs) + pad * (len(glyphs) + 1)
    height = gh + 2 * pad
    bm = [[1 for _ in range(width)] for _ in range(height)]
    x = pad
    for g in glyphs:
        h = len(g)
        w = len(g[0])
        yoff = pad + (gh - h) // 2
        for i in range(h):
            for j in range(w):
                if g[i][j]:
                    bm[yoff + i][x + j] = 0
        x += w + pad
    return bm


def scale_bitmap(bm: List[List[int]], sx: int = 1, sy: int = 1) -> List[List[int]]:
    if sx <= 1 and sy <= 1:
        return bm
    h = len(bm)
    w = len(bm[0]) if h else 0
    out = [[1 for _ in range(w * sx)] for _ in range(h * sy)]
    for i in range(h):
        for j in range(w):
            v = bm[i][j]
            for di in range(sy):
                for dj in range(sx):
                    out[i*sy + di][j*sx + dj] = v
    return out


def rotate_bitmap(bm: List[List[int]], angle_deg: float) -> List[List[int]]:
    """最近邻旋转，空白填白。"""
    if abs(angle_deg) < 1e-3:
        return bm
    rad = math.radians(angle_deg)
    h = len(bm)
    w = len(bm[0]) if h else 0
    cx, cy = w / 2.0, h / 2.0
    out = [[1 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            sx = int(round(cx + dx*math.cos(-rad) - dy*math.sin(-rad)))
            sy = int(round(cy + dx*math.sin(-rad) + dy*math.cos(-rad)))
            if 0 <= sy < h and 0 <= sx < w:
                out[y][x] = bm[sy][sx]
    return out


def bitmap_to_image(bm: List[List[int]]) -> List[List[int]]:
    """二值位图转灰度图（0/255）。"""
    return [[0 if v == 0 else 255 for v in row] for row in bm]


def bitmap_to_vec(bm: List[List[int]]) -> List[float]:
    out = []
    for row in bm:
        for v in row:
            out.append(1.0 if v else 0.0)
    return out


def ttfs_encode(bitmap: List[List[int]], Tmax: int = 20) -> List[int]:
    out = []
    for row in bitmap:
        for v in row:
            if v == 0:
                out.append(random.randint(0, max(1, Tmax//3)))
            else:
                out.append(-1)
    return out


def rate_encode(bitmap: List[List[int]], steps: int = 20) -> List[int]:
    out = []
    for row in bitmap:
        for v in row:
            r = random.randint(steps//4, steps//2) if v == 0 else random.randint(0, steps//5)
            out.append(r)
    return out

