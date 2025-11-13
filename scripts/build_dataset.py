#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建中英对齐数据集（不含图片）：
- 读取 concepts JSONL（每行包含 id, zh[], en[]）
- 生成 pairs.jsonl：{id, zh, en}
- 切分 train/val/test
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from self_core.utils.io import ensure_dir, write_jsonl, get_logger, new_run_dir


def load_concepts(path: Path) -> List[Dict]:
    arr: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                arr.append(json.loads(line))
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", type=str, default="data/raw", help="保留参数（不使用）")
    ap.add_argument("--out", type=str, required=True, help="data/processed 输出目录")
    ap.add_argument("--concepts", type=str, default="examples/concepts_small.jsonl")
    ap.add_argument("--pairs_per_concept", type=int, default=2)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    run_dir = Path(args.run) if args.run else new_run_dir()
    logger = get_logger("build_dataset")

    concepts: List[Dict] = load_concepts(Path(args.concepts))
    pairs: List[Dict] = []
    for it in concepts:
        cid = it.get("id")
        if not cid:
            continue
        zh_list = it.get("zh", [cid])
        en_list = it.get("en", [cid])
        for _ in range(max(1, int(args.pairs_per_concept))):
            zh = random.choice(zh_list)
            en = random.choice(en_list)
            pairs.append({"id": cid, "zh": zh, "en": en})

    random.shuffle(pairs)
    out_root = Path(args.out)
    train_p = out_root / "train"
    val_p = out_root / "val"
    test_p = out_root / "test"
    for p in (train_p, val_p, test_p):
        ensure_dir(p)

    n = len(pairs)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]

    for sub, arr in (("train", train), ("val", val), ("test", test)):
        out = out_root / sub / "triples.jsonl"
        for rec in arr:
            write_jsonl(out, rec)
        logger.info(f"Wrote {out} ({len(arr)} records)")

    logger.info(f"done | run={run_dir}")


if __name__ == "__main__":
    main()
