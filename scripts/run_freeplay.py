#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free-Play/Sleep 自发活动仿真：
- 无输入时，复发网络自发点火，记录分枝比估计、能耗、参数
- homeostasis/criticality 控制律小步调节
- 输出 freeplay.jsonl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from self_core.utils.io import read_config, new_run_dir, get_logger, write_jsonl
from self_core.emergence.spontaneous import FreePlaySimulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/emergence.toml")
    ap.add_argument("--run", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = read_config(Path(args.config))
    run_dir = Path(args.run) if args.run else new_run_dir()
    logger = get_logger("run_freeplay")
    logp = run_dir / "freeplay.jsonl"

    sim = FreePlaySimulator(cfg=cfg, run_dir=run_dir, seed=args.seed)
    steps = int(cfg["freeplay"]["steps"])
    log_every = int(cfg["freeplay"]["log_every"])
    for t in range(steps):
        rec = sim.step()
        if (t+1) % log_every == 0:
            rec["time"] = time.time()
            write_jsonl(logp, rec)
            logger.info(f"t={t+1} branch_hat={rec['branch_hat']:.3f} rate={rec['rate']:.3f} g_rec={rec['g_rec']:.3f}")

    logger.info(f"freeplay done | run={run_dir}")


if __name__ == "__main__":
    main()

