# -*- coding: utf-8 -*-
"""
对比学习与正则项
"""
from __future__ import annotations

import math
from typing import List


def cosine_sim(u: List[float], v: List[float]) -> float:
    nu = math.sqrt(sum(x*x for x in u)) + 1e-9
    nv = math.sqrt(sum(x*x for x in v)) + 1e-9
    return sum(x*y for x, y in zip(u, v)) / (nu*nv)


def info_nce_loss(A: List[List[float]], B: List[List[float]], temperature: float = 0.07) -> float:
    N = min(len(A), len(B))
    if N == 0:
        return 0.0
    sims = [[cosine_sim(A[i], B[j]) for j in range(N)] for i in range(N)]
    loss = 0.0
    for i in range(N):
        logits = [s / temperature for s in sims[i]]
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        denom = sum(exps)
        pos = exps[i] / max(1e-9, denom)
        loss += -math.log(pos + 1e-9)
    return loss / N


def l2(u: List[float], v: List[float]) -> float:
    return sum((x-y)*(x-y) for x, y in zip(u, v)) / max(1, len(u))


def center_loss(vecs: List[List[float]], center: List[float]) -> float:
    if not vecs:
        return 0.0
    return sum(l2(v, center) for v in vecs) / len(vecs)


def avg_rate_penalty(rate: float, target: float = 1.0) -> float:
    return (rate - target)*(rate - target)


def sparsity_penalty(sparse_codes: List[List[float]]) -> float:
    if not sparse_codes:
        return 0.0
    s = 0.0
    n = 0
    for code in sparse_codes:
        for v in code:
            s += v*v
            n += 1
    return s / max(1, n)

