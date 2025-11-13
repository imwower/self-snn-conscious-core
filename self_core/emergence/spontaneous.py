# -*- coding: utf-8 -*-
"""
自发活动模拟器：
- 使用 ConsciousCore 的 free_play_step
- 估计分枝比（branch_hat）与能耗
- homeostasis 与 criticality 控制律（简化）
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

from self_core.core.conscious_core import ConsciousCore
from self_core.emergence.homeostasis import controller_step


class FreePlaySimulator:
    def __init__(self, cfg: Dict, run_dir: Path, seed: int = 42):
        random.seed(seed)
        self.cfg = cfg
        d = 48
        k = 32
        self.core = ConsciousCore(dim=d, prototypes=k)
        self.g_rec = float(cfg["init"]["g_rec"])
        self.v_th = float(cfg["init"]["V_th"])
        self.rate_target = float(cfg["targets"]["rate_core"])
        self.ema_branch = 1.0
        self.alpha = float(cfg["control"]["beta_m"])

        self.energy = 0.0
        self.last_spikes = 1

    def step(self) -> Dict:
        rec = self.core.free_play_step(g_rec=self.g_rec, noise=0.02)
        spikes = sum(rec["spikes"]) if isinstance(rec["spikes"], list) else int(rec["spikes"])
        self.energy += spikes
        b = spikes / max(1, self.last_spikes)
        self.ema_branch = self.alpha * b + (1 - self.alpha) * self.ema_branch
        self.last_spikes = max(1, spikes)
        self.g_rec, self.v_th = controller_step(self.cfg, self.g_rec, self.v_th, rec["rate"], self.ema_branch)
        e_over_i = 1.0 + 0.1 * (self.g_rec - 1.0)
        rate_hist = [max(0, min(10, int(rec["rate"]*10)))]
        return {
            "branch_hat": self.ema_branch,
            "rate": rec["rate"],
            "energy": self.energy,
            "g_rec": self.g_rec,
            "v_th": self.v_th,
            "e_over_i": e_over_i,
            "rate_hist": rate_hist
        }

