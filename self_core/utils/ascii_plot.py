# -*- coding: utf-8 -*-
"""
ASCII 图：条形图/折线/热图（简化）
"""
from __future__ import annotations

from typing import List


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def plot_bar(vals: List[float], width: int = 60, char: str = "█") -> str:
    if not vals:
        return ""
    mx = max(vals)
    mn = min(vals)
    span = mx - mn if mx > mn else 1.0
    out = []
    for v in vals:
        h = int((v - mn) / span * width)
        out.append(char * h)
    return "\n".join(out)


def plot_line(vals: List[float], width: int = 60, height: int = 10) -> str:
    if not vals:
        return ""
    mn = min(vals)
    mx = max(vals)
    span = mx - mn if mx > mn else 1.0
    n = len(vals)
    if n > width:
        step = n / width
        xs = [vals[int(i*step)] for i in range(width)]
    else:
        xs = vals + [vals[-1]] * (width - n)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for i, v in enumerate(xs):
        y = int((v - mn) / span * (height - 1))
        y = (height - 1) - y
        grid[y][i] = "*"
    return "\n".join("".join(r) for r in grid)


def plot_heat(matrix: List[List[float]], width: int = 60, height: int = 20) -> str:
    if not matrix:
        return ""
    h = len(matrix)
    w = len(matrix[0])
    mn = min(min(r) for r in matrix)
    mx = max(max(r) for r in matrix)
    span = mx - mn if mx > mn else 1.0
    chars = " .:-=+*#%@"
    s = []
    for y in range(h):
        row = []
        for x in range(w):
            t = (matrix[y][x] - mn) / span
            idx = int(t * (len(chars) - 1))
            row.append(chars[idx])
        s.append("".join(row))
    return "\n".join(s)

