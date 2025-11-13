#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E/I 平衡探针：
- 从 freeplay.jsonl 估计兴奋/抑制电流比例与发放分布偏差（简单近似）
- 输出 ei.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_core.utils.io import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    args = ap.parse_args()
    logger = get_logger("eibalance_probe")

    logp = Path(args.run) / "freeplay.jsonl"
    if not logp.exists():
        logger.info("no freeplay logs")
        return
    with open(logp, "r", encoding="utf-8") as f:
        arr = [json.loads(x) for x in f if x.strip()]

    if not arr:
        logger.info("empty logs")
        return
    e_over_i = arr[-1].get("e_over_i", 1.0)
    hist = arr[-1].get("rate_hist", [0, 1, 0])
    out = {"e_over_i": e_over_i, "rate_hist": hist}
    with open(Path(args.run) / "ei.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"ei saved | E/I={e_over_i:.3f}")


if __name__ == "__main__":
    main()

