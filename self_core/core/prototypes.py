# -*- coding: utf-8 -*-
"""
原型库：
- K×D 原型向量
- 余弦/欧氏距离
- 利用率与类内方差记录（简化）
"""
from __future__ import annotations

import math
import random
from typing import List, Dict


def cosine(u: List[float], v: List[float]) -> float:
    nu = math.sqrt(sum(x*x for x in u)) + 1e-9
    nv = math.sqrt(sum(x*x for x in v)) + 1e-9
    return sum(x*y for x, y in zip(u, v)) / (nu*nv)


class Prototypes:
    def __init__(self, k: int, d: int):
        self.k = k
        self.d = d
        scale = 1.0 / math.sqrt(d)
        self.protos: List[List[float]] = [[(random.random()*2-1)*scale for _ in range(d)] for _ in range(k)]
        self.usage = [0 for _ in range(k)]

    def match(self, z: List[float]) -> int:
        sims = [cosine(p, z) for p in self.protos]
        idx = max(range(self.k), key=lambda i: sims[i])
        self.usage[idx] += 1
        return idx

    def state_dict(self) -> Dict:
        return {"prototypes": self.protos, "usage": self.usage}

