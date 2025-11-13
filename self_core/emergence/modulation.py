# -*- coding: utf-8 -*-
"""
神经调质门控（占位）：
- ACh：编码期↑（抑制复发、增强输入）
- NE：探索强度（温度/增益）
- DA：奖励门控
"""
from __future__ import annotations

from typing import Dict


def apply_modulation(params: Dict, phase: str = "explore") -> Dict:
    out = params.copy()
    if phase == "encode":
        out["g_rec"] *= 0.95
    elif phase == "consolidate":
        out["g_rec"] *= 0.98
    elif phase == "explore":
        out["temperature"] = out.get("temperature", 0.03) * 1.05
    return out

