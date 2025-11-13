# -*- coding: utf-8 -*-
"""
Homeostasis 控制律（简化）：
- 按目标发放率调节阈值
- 按分枝比调节 g_rec
"""
from __future__ import annotations

from typing import Dict, Tuple


def controller_step(cfg: Dict, g_rec: float, v_th: float, rate: float, branch_hat: float) -> Tuple[float, float]:
    target_rate = float(cfg["targets"]["rate_core"])
    rate_delta = float(cfg["control"]["rate_delta"])
    v_th = v_th + (rate - target_rate) * rate_delta
    v_th = max(0.2, min(2.0, v_th))
    eta_m = float(cfg["control"]["eta_m"])
    g_rec = g_rec - (branch_hat - 1.0) * eta_m
    g_rec = max(0.5, min(1.5, g_rec))
    return g_rec, v_th

