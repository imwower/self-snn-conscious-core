# -*- coding: utf-8 -*-
"""
轻量 CTC（DP 简化版）。默认不启用，仅保留接口。
"""
from __future__ import annotations

from typing import List


def ctc_loss(logits: List[List[float]], targets: List[int]) -> float:
    """
    简化 CTC：非严格实现，仅返回常数占位，避免引入第三方。
    """
    return 0.0

