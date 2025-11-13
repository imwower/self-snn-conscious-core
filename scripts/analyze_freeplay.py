#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free-Play 分析与 ASCII 可视化：
- 读取 freeplay.jsonl
- 打印分枝比/能耗/多样性等曲线摘要
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

from self_core.utils.io import get_logger
from self_core.utils.ascii_plot import plot_line


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    args = ap.parse_args()
    logger = get_logger("analyze_freeplay")

    logp = Path(args.run) / "freeplay.jsonl"
    arr = load_jsonl(logp)
    if not arr:
        logger.info("no freeplay logs")
        return
    branch = [x["branch_hat"] for x in arr]
    rate = [x["rate"] for x in arr]
    energy = [x["energy"] for x in arr]
    print("Branching ratio (EMA):")
    print(plot_line(branch, width=60, height=10))
    print("Avg firing rate:")
    print(plot_line(rate, width=60, height=10))
    print("Energy:")
    print(plot_line(energy, width=60, height=10))
    logger.info("analysis done")


if __name__ == "__main__":
    main()

