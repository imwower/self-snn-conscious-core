# -*- coding: utf-8 -*-
"""
SPSA/坐标爬山（占位），用于低频扰动与回退
"""
from __future__ import annotations

import random
from typing import Dict


def spsa_step(params: Dict, alpha: float = 0.05, delta: float = 0.02) -> Dict:
    out = params.copy()
    for k in ("g_rec", "temperature"):
        if k in out:
            sign = 1 if random.random() < 0.5 else -1
            out[k] = out[k] * (1 + sign * delta * alpha)
    return out

