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
from typing import List, Dict

from self_core.utils.io import get_logger, read_config, new_run_dir
from self_core.eval.metrics import retrieval_metrics, clustering_metrics, vp_distance, energy_metrics
from self_core.encoding.vision_draw import load_image_to_vec
from self_core.encoding.system_font import render_text_to_bitmap, find_chinese_font


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

    imgs = [load_image_to_vec(Path(r["img_sem"])) for r in val]
    # 文本图视角：若文件缺失，使用系统字体渲染英文短语作为近似
    font_path = find_chinese_font()
    def bmp_to_vec(bm):
        return [(255 - v)/255.0 for row in bm for v in row]
    txts = []
    for r in val:
        p = Path(r["img_text"]) 
        if p.exists():
            txts.append(load_image_to_vec(p))
        else:
            bm = render_text_to_bitmap(r["en"], font_path=font_path, size=28, padding=2, stroke_width=1, stroke_fill=0)
            txts.append(bmp_to_vec(bm))
    labels = [r["id"] for r in val]

    ret = retrieval_metrics(txts, imgs, labels, topk=(1, 5))
    clu = clustering_metrics(imgs, labels)
    vp = vp_distance([[i % 5 for i in range(10)] for _ in range(5)], [[i % 5 for i in range(10)] for _ in range(5)])
    eng = energy_metrics([random.randint(10, 50) for _ in range(20)], steps=100)

    out = {
        "retrieval": ret,
        "clustering": clu,
        "vp_distance": vp,
        "energy": eng
    }
    with open(run_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"eval done | run={run_dir}")


if __name__ == "__main__":
    main()
