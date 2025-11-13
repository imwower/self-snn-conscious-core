#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测脚本：
- 读取 run 日志或数据，计算检索/聚类/SNN 指标
- 输出 eval.json 与 ASCII 可视化
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import List, Dict

 # 兼容直接以 `python scripts/eval.py` 运行：把仓库根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_core.utils.io import get_logger, read_config, new_run_dir
from self_core.eval.metrics import retrieval_metrics
from self_core.encoding.system_font import render_text_to_bitmap, find_chinese_font
import pickle


def read_triples(path: Path) -> List[Dict]:
    arr = []
    if not path.exists():
        return arr
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                arr.append(json.loads(line))
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="")
    ap.add_argument("--config", type=str, default="config/default.toml")
    args = ap.parse_args()
    run_dir = Path(args.run) if args.run else new_run_dir()
    logger = get_logger("eval")

    cfg = read_config(Path(args.config))
    val = read_triples(Path(cfg["data"]["val"]))
    if not val:
        logger.info("no val set, evaluating on train instead")
        val = read_triples(Path(cfg["data"]["train"]))

    # 从 ckpt 读取投影矩阵
    ckpt = Path(run_dir) / "ckpt.pkl"
    if not ckpt.exists():
        logger.info("no checkpoint found for eval")
        return
    with open(ckpt, "rb") as f:
        state = pickle.load(f)
    W_zh = state.get("W_zh")
    W_en = state.get("W_en")

    def project(x, W):
        return [sum(x[k]*W[k][j] for k in range(len(x))) for j in range(len(W[0]))]
    def normalize(v):
        import math
        s = math.sqrt(sum(x*x for x in v)) + 1e-9
        return [x/s for x in v]
    def vec_reduce(v, target=256):
        if len(v) == target:
            return v[:]
        out = [0.0 for _ in range(target)]
        for i, x in enumerate(v):
            out[i % target] += x
        s = (sum(x*x for x in out)) ** 0.5 + 1e-9
        return [x/s for x in out]
    font_path = find_chinese_font()
    # 文本→向量缓存
    _cache = {}
    def text_to_vec(text: str):
        v = _cache.get(text)
        if v is not None:
            return v
        bm = render_text_to_bitmap(text, font_path=font_path, size=28, padding=2, stroke_width=1, stroke_fill=0)
        vv = [(255 - x)/255.0 for row in bm for x in row]
        _cache[text] = vv
        return vv

    zh_embs = []
    en_embs = []
    labels = []
    for r in val:
        vzh = normalize(project(vec_reduce(text_to_vec(r["zh"])), W_zh))
        ven = normalize(project(vec_reduce(text_to_vec(r["en"])), W_en))
        zh_embs.append(vzh)
        en_embs.append(ven)
        labels.append(r["id"])

    ret = retrieval_metrics(zh_embs, en_embs, labels, topk=(1, 5))
    out = {"retrieval_zh2en": ret}
    with open(run_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"eval done | run={run_dir}")


if __name__ == "__main__":
    main()
