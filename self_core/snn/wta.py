# -*- coding: utf-8 -*-
"""
多胜者 WTA：选取 top-k 保留，其余抑制为 0
"""
from __future__ import annotations

from typing import List


def wta_topk(x: List[float], k: int) -> List[float]:
    if k <= 0:
        return [0.0 for _ in x]
    idx = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:k]
    out = [0.0 for _ in x]
    for i in idx:
        out[i] = x[i]
    return out

