#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建三视角对齐数据集：
- 扫描 data/raw/{text,semantic}/<concept_id>/ 下的图片
- 组合为 triples.jsonl：{id, zh, en, img_text, img_sem}
- 切分 train/val/test
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from self_core.utils.io import ensure_dir, write_jsonl, get_logger, new_run_dir


def collect_files(root: Path) -> Dict[str, List[Path]]:
    data: Dict[str, List[Path]] = {}
    if not root.exists():
        return data
    for p in root.glob("*/*"):
        if p.is_file() and p.suffix.lower() in (".pgm", ".ppm"):
            cid = p.parent.name
            data.setdefault(cid, []).append(p)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", type=str, required=True, help="data/raw 目录")
    ap.add_argument("--out", type=str, required=True, help="data/processed 输出目录")
    ap.add_argument("--concepts", type=str, default="examples/concepts_small.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    run_dir = Path(args.run) if args.run else new_run_dir()
    logger = get_logger("build_dataset")

    text_root = Path(args.indir) / "text"
    sem_root = Path(args.indir) / "semantic"
    text_map = collect_files(text_root)
    sem_map = collect_files(sem_root)

    concepts: Dict[str, Dict] = {}
    cpath = Path(args.concepts)
    if cpath.exists():
        with open(cpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                concepts[obj["id"]] = obj

    triples: List[Dict] = []
    for cid, files in text_map.items():
        sem_files = sem_map.get(cid, [])
        if not files or not sem_files:
            continue
        random.shuffle(files)
        random.shuffle(sem_files)
        n = min(len(files), len(sem_files))
        info = concepts.get(cid, {"zh":[cid], "en":[cid]})
        zh = random.choice(info.get("zh", [cid]))
        en = random.choice(info.get("en", [cid]))
        for i in range(n):
            triples.append({
                "id": cid,
                "zh": zh,
                "en": en,
                "img_text": str(files[i]),
                "img_sem": str(sem_files[i]),
            })

    random.shuffle(triples)
    out_root = Path(args.out)
    train_p = out_root / "train"
    val_p = out_root / "val"
    test_p = out_root / "test"
    for p in (train_p, val_p, test_p):
        ensure_dir(p)

    n = len(triples)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test = triples[:n_test]
    val = triples[n_test:n_test + n_val]
    train = triples[n_test + n_val:]

    for sub, arr in (("train", train), ("val", val), ("test", test)):
        out = out_root / sub / "triples.jsonl"
        for rec in arr:
            write_jsonl(out, rec)
        logger.info(f"Wrote {out} ({len(arr)} records)")

    logger.info(f"done | run={run_dir}")


if __name__ == "__main__":
    main()

