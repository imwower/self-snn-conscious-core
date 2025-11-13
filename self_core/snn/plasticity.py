# -*- coding: utf-8 -*-
"""
可塑性（简化）：
- iSTDP：抑制权重按发放相关性调整
- Synaptic scaling：兴奋性权重朝目标率缩放
- Intrinsic plasticity：阈值向目标分布缓动
"""
from __future__ import annotations

from typing import List


def istdp_update(w_inh: List[List[float]], pre: List[int], post: List[int], eta: float = 1e-3) -> List[List[float]]:
    for i in range(len(w_inh)):
        for j in range(len(w_inh[0])):
            if pre[j] and post[i]:
                w_inh[i][j] += eta  # 抑制增强
            else:
                w_inh[i][j] *= (1.0 - 0.001)  # 衰减
            w_inh[i][j] = max(0.0, min(2.0, w_inh[i][j]))
    return w_inh


def synaptic_scaling(w_exc: List[List[float]], rate: float, rate_target: float, eta: float = 1e-3) -> List[List[float]]:
    scale = 1.0 + eta * (rate_target - rate)
    for i in range(len(w_exc)):
        for j in range(len(w_exc[0])):
            w_exc[i][j] *= scale
    return w_exc


def intrinsic_plasticity(v_th: float, rate: float, rate_target: float, eta: float = 1e-4) -> float:
    return max(0.1, min(3.0, v_th + eta * (rate - rate_target)))

