# -*- coding: utf-8 -*-
"""
整数步延迟环形缓冲
"""
from __future__ import annotations

from typing import List


class DelayLine:
    def __init__(self, length: int, dim: int):
        self.length = max(1, int(length))
        self.buff = [[0.0 for _ in range(dim)] for _ in range(self.length)]
        self.idx = 0

    def step(self, x: List[float]) -> List[float]:
        out = self.buff[self.idx]
        self.buff[self.idx] = x[:]
        self.idx = (self.idx + 1) % self.length
        return out[:]

