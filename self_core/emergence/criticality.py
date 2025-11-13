# -*- coding: utf-8 -*-
"""
临界性统计：
- 雪崩大小从 freeplay 日志提取
- 幂律 MLE（连续近似）与 KS
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple


def avalanche_sizes_from_logs(path: Path) -> List[int]:
    if not path.exists():
        return []
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            s = int(max(1, obj.get("energy", 1.0)))
            arr.append(s)
    sizes = []
    prev = 0
    for e in arr:
        sizes.append(max(1, e - prev))
        prev = e
    return [s for s in sizes if s > 0]


def fit_powerlaw_mle(data: List[int]) -> Tuple[float, float]:
    xmin = 1.0
    xs = [x for x in data if x >= xmin]
    if not xs:
        return 2.0, xmin
    S = sum(math.log(x/xmin) for x in xs)
    alpha = 1.0 + len(xs) / max(1e-9, S)
    return max(1.01, alpha), xmin


def ks_statistic(data: List[int], alpha: float, xmin: float) -> float:
    xs = sorted(x for x in data if x >= xmin)
    n = len(xs)
    if n == 0:
        return 1.0
    def cdf_model(x: float) -> float:
        if x < xmin:
            return 0.0
        return 1.0 - (x/xmin) ** (1.0 - alpha)
    D = 0.0
    for i, x in enumerate(xs, 1):
        F_emp = i / n
        F_mod = cdf_model(x)
        D = max(D, abs(F_emp - F_mod))
    return D

