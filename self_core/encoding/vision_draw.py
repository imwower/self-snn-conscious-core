# -*- coding: utf-8 -*-
"""
视觉几何绘制与图像载入：
- 基本几何（圆/矩形/多边形）
- 简单噪声
- PGM/PPM 载入为向量
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from self_core.utils.io import load_pgm_ppm_to_gray_vec


def blank_rgb(w: int, h: int, r: int, g: int, b: int) -> List[List[List[int]]]:
    return [[[r, g, b] for _ in range(w)] for _ in range(h)]


def draw_rect(img: List[List[List[int]]], x0: int, y0: int, x1: int, y1: int, color=(0, 0, 0), fill=True):
    h = len(img)
    w = len(img[0]) if h else 0
    for y in range(max(0, min(y0, y1)), min(h, max(y0, y1))):
        for x in range(max(0, min(x0, x1)), min(w, max(x0, x1))):
            if fill or y in (y0, y1-1) or x in (x0, x1-1):
                img[y][x] = [color[0], color[1], color[2]]


def draw_circle(img: List[List[List[int]]], cx: int, cy: int, r: int, color=(0, 0, 0), fill=True):
    h = len(img)
    w = len(img[0]) if h else 0
    r2 = r*r
    for y in range(max(0, cy-r), min(h, cy+r)):
        for x in range(max(0, cx-r), min(w, cx+r)):
            d2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
            if (fill and d2 <= r2) or (not fill and abs(d2-r2) <= r):
                img[y][x] = [color[0], color[1], color[2]]


def draw_polygon(img: List[List[List[int]]], pts: List[Tuple[int, int]], color=(0, 0, 0), fill=True):
    if not pts:
        return
    if fill:
        ys = [y for _, y in pts]
        y_min, y_max = min(ys), max(ys)
        for y in range(y_min, y_max+1):
            xs = []
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i+1) % len(pts)]
                if y1 == y2:
                    continue
                if (y >= min(y1, y2)) and (y < max(y1, y2)):
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    xs.append(int(x))
            xs.sort()
            for i in range(0, len(xs), 2):
                x0 = xs[i]
                x1 = xs[i+1] if i+1 < len(xs) else xs[i]
                for x in range(x0, x1+1):
                    if 0 <= y < len(img) and 0 <= x < len(img[0]):
                        img[y][x] = [color[0], color[1], color[2]]
    else:
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i+1) % len(pts)]
            steps = max(abs(x2-x1), abs(y2-y1))
            for t in range(steps+1):
                x = int(x1 + (x2 - x1) * t / (steps or 1))
                y = int(y1 + (y2 - y1) * t / (steps or 1))
                if 0 <= y < len(img) and 0 <= x < len(img[0]):
                    img[y][x] = [color[0], color[1], color[2]]


def draw_ellipse(img: List[List[List[int]]], cx: int, cy: int, rx: int, ry: int, color=(0, 0, 0), fill=True):
    h = len(img)
    w = len(img[0]) if h else 0
    rx2 = rx*rx
    ry2 = ry*ry
    for y in range(max(0, cy-ry), min(h, cy+ry)):
        for x in range(max(0, cx-rx), min(w, cx+rx)):
            dx = x - cx
            dy = y - cy
            val = (dx*dx)/max(1, rx2) + (dy*dy)/max(1, ry2)
            if (fill and val <= 1.0) or (not fill and abs(val-1.0) <= 0.05):
                img[y][x] = [color[0], color[1], color[2]]


def add_noise_rgb(img: List[List[List[int]]], amount: float = 0.02):
    h = len(img)
    w = len(img[0]) if h else 0
    n = int(h * w * amount)
    for _ in range(n):
        x = random.randrange(0, w)
        y = random.randrange(0, h)
        ch = random.randrange(0, 3)
        img[y][x][ch] = max(0, min(255, img[y][x][ch] + random.randint(-30, 30)))


def load_image_to_vec(path: Path) -> List[float]:
    return load_pgm_ppm_to_gray_vec(path)
