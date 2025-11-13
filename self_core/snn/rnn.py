# -*- coding: utf-8 -*-
"""
简易复发骨架：x_{t+1} = g_rec * W * x_t + noise
"""
from __future__ import annotations

import random
from typing import List


def matvec(W: List[List[float]], x: List[float]) -> List[float]:
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


def rnn_step(W: List[List[float]], x: List[float], g_rec: float, noise: float = 0.01) -> List[float]:
    y = matvec(W, x)
    out = []
    for v in y:
        out.append(g_rec * v + (random.random()*2-1)*noise)
    return out

