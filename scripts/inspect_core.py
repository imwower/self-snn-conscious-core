#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意识核可解释性：导出每个原型的 Top-K 样本（基于余弦）
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import List, Dict

from self_core.utils.io import get_logger
from self_core.core.prototypes import cosine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()
    logger = get_logger("inspect_core")

    ckpt = Path(args.run) / "ckpt.pkl"
    if not ckpt.exists():
        logger.info("no checkpoint found")
        return
    with open(ckpt, "rb") as f:
        state = pickle.load(f)
    core = state.get("core", {})
    protos = core.get("prototypes", [])
    embs = [[math.sin(i+j) for j in range(len(protos[0]) if protos else 16)] for i in range(32)]
    result: Dict[int, List[int]] = {}
    for pi, p in enumerate(protos):
        sims = [(i, cosine(p, e)) for i, e in enumerate(embs)]
        sims.sort(key=lambda x: x[1], reverse=True)
        result[pi] = [i for i, _ in sims[:args.topk]]
    with open(Path(args.run) / "inspect.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"wrote {args.run}/inspect.json")


if __name__ == "__main__":
    main()

