# -*- coding: utf-8 -*-
"""
意识核主接口：
- forward_batch：将输入嵌入与原型库/WTA 交互，输出稀疏码与发放率估计
- free_play_step：无输入时的复发步进（产生分枝比估计）
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

from self_core.core.prototypes import Prototypes, cosine
from self_core.snn.wta import wta_topk
from self_core.snn.rnn import rnn_step
from self_core.snn.neuron import lif_step


class ConsciousCore:
    def __init__(self, dim: int, prototypes: int):
        self.dim = dim
        self.protos = Prototypes(prototypes, dim)
        # 复发矩阵（简化：对角主导）
        scale = 1.0 / math.sqrt(dim)
        self.W_rec = [[0.0 for _ in range(dim)] for _ in range(dim)]
        for i in range(dim):
            self.W_rec[i][i] = 0.9
            if i+1 < dim:
                self.W_rec[i][i+1] = 0.1 * scale
        self.v = [0.0 for _ in range(dim)]
        self.v_th = 1.0

    def forward_batch(self, zs: List[List[float]]) -> Dict:
        sparse_codes: List[List[float]] = []
        rates: List[float] = []
        for z in zs:
            sims = [cosine(p, z) for p in self.protos.protos]
            k = max(1, len(sims)//8)
            code = wta_topk(sims, k=k)
            sparse_codes.append(code)
            rates.append(sum(1 for v in code if v > 0) / max(1, len(code)))
        return {"sparse_codes": sparse_codes, "rates": sum(rates)/max(1, len(rates))}

    def free_play_step(self, g_rec: float, noise: float = 0.01) -> Dict:
        x = [1.0 if random.random() < 0.05 else 0.0 for _ in range(self.dim)]
        I = rnn_step(self.W_rec, x, g_rec=g_rec, noise=noise)
        spikes, self.v = lif_step(self.v, I, v_th=self.v_th)
        rate = sum(spikes) / max(1, len(spikes))
        return {"spikes": spikes, "rate": rate}

    def state_dict(self) -> Dict:
        d = self.protos.state_dict()
        d.update({"W_rec": self.W_rec, "v_th": self.v_th})
        return d

