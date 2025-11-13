#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临界性体检：
- 对 freeplay.jsonl 计算雪崩幂律 MLE+KS 概要（简化）
- 输出 crit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_core.utils.io import get_logger
from self_core.emergence.criticality import avalanche_sizes_from_logs, fit_powerlaw_mle, ks_statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    args = ap.parse_args()
    logger = get_logger("crit_test")

    logp = Path(args.run) / "freeplay.jsonl"
    sizes = avalanche_sizes_from_logs(logp)
    if not sizes:
        logger.info("no avalanches found")
        with open(Path(args.run) / "crit.json", "w", encoding="utf-8") as f:
            json.dump({"note": "no avalanches"}, f)
        return

    alpha, xmin = fit_powerlaw_mle(sizes)
    ks = ks_statistic(sizes, alpha, xmin)
    out = {"alpha": alpha, "xmin": xmin, "ks": ks, "n": len(sizes)}
    with open(Path(args.run) / "crit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"crit saved | alpha={alpha:.2f} ks={ks:.3f}")


if __name__ == "__main__":
    main()

