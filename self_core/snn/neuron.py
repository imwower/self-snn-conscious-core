# -*- coding: utf-8 -*-
"""
LIF/ALIF 神经元（列表实现）+ 替代梯度
- 仅用于 free-play 简化仿真
"""
from __future__ import annotations

from typing import List, Tuple


def fast_sigmoid(x: float) -> float:
    return x / (1.0 + abs(x))


def lif_step(v: List[float], I: List[float], v_th: float = 1.0, leak: float = 0.95) -> Tuple[List[int], List[float]]:
    """
    单步 LIF：v = leak*v + I；若 v>v_th 则发放并重置为 0
    """
    spikes = [0 for _ in v]
    for i in range(len(v)):
        v[i] = leak * v[i] + I[i]
        if v[i] > v_th:
            spikes[i] = 1
            v[i] = 0.0
    return spikes, v

