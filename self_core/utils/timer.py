# -*- coding: utf-8 -*-
"""
计时器工具
"""
from __future__ import annotations

import time
from typing import Optional


class Timer:
    """简单计时器上下文。"""
    def __init__(self, name: str = "timer"):
        self.name = name
        self.t0: Optional[float] = None

    def __enter__(self):
        self.t0 = time.time()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        t1 = time.time()
        dt = (t1 - (self.t0 or t1))
        print(f"[{self.name}] {dt:.3f}s")

