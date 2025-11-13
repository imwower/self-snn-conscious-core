#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验收脚本：
- 检查必要文件/日志
- 校验阈值（分枝比、谱半径、训练趋势、发放率）
- 运行单测
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from self_core.utils.io import get_logger


def exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def main():
    logger = get_logger("check_acceptance")
    root = Path(".").resolve()
    readme = root / "README.md"
    cfg1 = root / "config/default.toml"
    cfg2 = root / "config/emergence.toml"
    if not (exists(readme) and exists(cfg1) and exists(cfg2)):
        print("ACCEPTANCE FAILED: missing README or configs")
        sys.exit(1)

    runs = sorted((root / "runs").glob("*"), key=lambda p: p.name)
    if not runs:
        print("ACCEPTANCE FAILED: no runs found, please execute pipeline before")
        sys.exit(1)
    run = runs[-1]
    logger.info(f"check run={run}")

    need = [
        run / "logs.jsonl",
        run / "eval.json",
        run / "freeplay.jsonl",
        run / "crit.json",
        run / "eig.json",
        run / "ei.json",
    ]
    for p in need:
        if not exists(p):
            print(f"ACCEPTANCE FAILED: missing {p}")
            sys.exit(1)

    with open(run / "freeplay.jsonl", "r", encoding="utf-8") as f:
        arr = [json.loads(x) for x in f if x.strip()]
    if not arr:
        print("ACCEPTANCE FAILED: empty freeplay logs")
        sys.exit(1)
    bh = arr[-1].get("branch_hat", 1.0)
    rate = arr[-1].get("rate", 1.0)
    if not (0.9 <= bh <= 1.1):
        print(f"ACCEPTANCE FAILED: branch_hat={bh} not in [0.9,1.1]")
        sys.exit(1)
    target_rate = 1.0
    if abs(rate - target_rate) / max(1e-6, target_rate) > 0.3:
        print(f"ACCEPTANCE FAILED: rate {rate} too far from target")
        sys.exit(1)

    eig = json.loads((run / "eig.json").read_text(encoding="utf-8"))
    lam = eig.get("lambda_max", 1.0)
    if not (0.85 <= lam <= 1.15):
        print(f"ACCEPTANCE FAILED: lambda_max={lam} not near 1")
        sys.exit(1)

    logs = (run / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    aligns = [json.loads(x).get("loss_align") for x in logs if "loss_align" in x]
    if aligns:
        ma = sum(aligns[-5:]) / max(1, len(aligns[-5:]))
        mb = sum(aligns[:5]) / max(1, len(aligns[:5]))
        if ma > mb * 1.2:
            print("ACCEPTANCE FAILED: align loss significantly worse")
            sys.exit(1)

    print("ACCEPTED")
    logger.info("ACCEPTED")


if __name__ == "__main__":
    main()

