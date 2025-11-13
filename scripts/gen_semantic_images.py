#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义图片生成器（PPM），零依赖。
- 读取概念清单（JSONL）
- 用几何组合（圆/矩形/多边形/叶片）生成图标式语义图片
- 输出 PPM 文件
- 每次生成写入 JSONL 日志
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List

from self_core.utils.io import ensure_dir, write_ppm, get_logger, new_run_dir, write_jsonl
from self_core.encoding.vision_draw import blank_rgb, draw_circle, draw_rect, draw_polygon, add_noise_rgb
from self_core.encoding.vision_draw import draw_ellipse


def draw_icon(cid: str, rgb, W: int, H: int):
    """根据概念绘制更贴近物体的图标。"""
    cx, cy = W//2, H//2
    if cid == "apple":
        # 红苹果 + 叶片 + 枝
        draw_circle(rgb, cx, cy+5, 20, color=(220, 0, 0), fill=True)
        draw_circle(rgb, cx, cy+5, 20, color=(0, 0, 0), fill=False)
        draw_rect(rgb, cx-2, cy-10, cx+2, cy-2, color=(100, 60, 20), fill=True)
        leaf = [(cx+5, cy-12), (cx+18, cy-10), (cx+10, cy-2)]
        draw_polygon(rgb, leaf, color=(0, 160, 0), fill=True)
    elif cid == "banana":
        # 弯月状香蕉（多边形近似）
        pts = []
        for t in range(-30, 210, 15):
            rad = math.radians(t)
            pts.append((cx + int(22*math.cos(rad)), cy + int(10*math.sin(rad))))
        for t in range(210, -30, -15):
            rad = math.radians(t)
            pts.append((cx + int(16*math.cos(rad)), cy + int(4*math.sin(rad))))
        draw_polygon(rgb, pts, color=(240, 210, 0), fill=True)
        draw_polygon(rgb, pts, color=(0, 0, 0), fill=False)
    elif cid == "cat":
        # 猫脸：圆 + 三角耳 + 眼睛
        draw_circle(rgb, cx, cy+4, 18, color=(200, 180, 160), fill=True)
        ears = [(cx-10, cy-5), (cx-20, cy-15), (cx-2, cy-10)]
        draw_polygon(rgb, ears, color=(200, 180, 160), fill=True)
        ears2 = [(cx+10, cy-5), (cx+20, cy-15), (cx+2, cy-10)]
        draw_polygon(rgb, ears2, color=(200, 180, 160), fill=True)
        draw_circle(rgb, cx-6, cy+2, 2, color=(0,0,0), fill=True)
        draw_circle(rgb, cx+6, cy+2, 2, color=(0,0,0), fill=True)
    elif cid == "dog":
        draw_circle(rgb, cx, cy+6, 18, color=(180, 140, 100), fill=True)
        draw_rect(rgb, cx-18, cy, cx-6, cy+6, color=(180, 140, 100), fill=True)
        draw_rect(rgb, cx+6, cy, cx+18, cy+6, color=(180, 140, 100), fill=True)
        draw_circle(rgb, cx-6, cy+6, 2, color=(0,0,0), fill=True)
        draw_circle(rgb, cx+6, cy+6, 2, color=(0,0,0), fill=True)
    elif cid == "car":
        draw_rect(rgb, cx-18, cy-2, cx+18, cy+10, color=(60,120,220), fill=True)
        draw_rect(rgb, cx-10, cy-8, cx+10, cy-2, color=(200,200,255), fill=True)
        draw_circle(rgb, cx-12, cy+11, 5, color=(0,0,0), fill=True)
        draw_circle(rgb, cx+12, cy+11, 5, color=(0,0,0), fill=True)
    elif cid == "tree":
        draw_rect(rgb, cx-3, cy, cx+3, cy+18, color=(120,70,20), fill=True)
        draw_circle(rgb, cx, cy-2, 12, color=(20,160,20), fill=True)
        draw_circle(rgb, cx-10, cy+2, 10, color=(20,150,20), fill=True)
        draw_circle(rgb, cx+10, cy+2, 10, color=(20,150,20), fill=True)
    elif cid == "bird":
        draw_ellipse(rgb, cx, cy, 20, 12, color=(80, 150, 220), fill=True)
        draw_polygon(rgb, [(cx+10, cy), (cx+18, cy-3), (cx+18, cy+3)], color=(240,150,0), fill=True)
        draw_circle(rgb, cx-6, cy-2, 2, color=(0,0,0), fill=True)
    elif cid == "fish":
        draw_ellipse(rgb, cx, cy, 22, 12, color=(80, 160, 200), fill=True)
        draw_polygon(rgb, [(cx-20, cy), (cx-30, cy-6), (cx-30, cy+6)], color=(80,160,200), fill=True)
        draw_circle(rgb, cx+8, cy-2, 2, color=(0,0,0), fill=True)
    elif cid == "house":
        draw_rect(rgb, cx-16, cy-2, cx+16, cy+18, color=(240,240,200), fill=True)
        draw_polygon(rgb, [(cx-18, cy-2), (cx, cy-18), (cx+18, cy-2)], color=(200,80,60), fill=True)
        draw_rect(rgb, cx-4, cy+4, cx+4, cy+18, color=(160,100,60), fill=True)
        draw_rect(rgb, cx+8, cy+4, cx+14, cy+10, color=(200,200,255), fill=True)
    elif cid == "book":
        draw_rect(rgb, cx-18, cy-10, cx+18, cy+12, color=(230,230,255), fill=True)
        draw_rect(rgb, cx-2, cy-10, cx, cy+12, color=(200,200,220), fill=True)
    elif cid == "cup":
        draw_rect(rgb, cx-12, cy-8, cx+8, cy+12, color=(240,240,240), fill=True)
        draw_rect(rgb, cx+8, cy-2, cx+14, cy+6, color=(240,240,240), fill=True)
    elif cid == "phone":
        draw_rect(rgb, cx-10, cy-16, cx+10, cy+16, color=(230,230,230), fill=True)
        draw_rect(rgb, cx-9, cy-15, cx+9, cy+15, color=(30,30,30), fill=True)
        draw_circle(rgb, cx, cy+14, 1, color=(200,200,200), fill=True)
    elif cid == "chair":
        draw_rect(rgb, cx-12, cy+2, cx+12, cy+8, color=(180,120,60), fill=True)
        draw_rect(rgb, cx-12, cy-10, cx-8, cy+2, color=(180,120,60), fill=True)
        draw_rect(rgb, cx+8, cy-10, cx+12, cy+2, color=(180,120,60), fill=True)
    elif cid == "table":
        draw_rect(rgb, cx-20, cy-4, cx+20, cy+2, color=(160,100,60), fill=True)
        draw_rect(rgb, cx-18, cy+2, cx-14, cy+16, color=(160,100,60), fill=True)
        draw_rect(rgb, cx+14, cy+2, cx+18, cy+16, color=(160,100,60), fill=True)
    else:
        # 默认形状
        draw_circle(rgb, cx, cy, 16, color=(120,120,200), fill=True)


def draw_leaf(cx: int, cy: int, r: int):
    """生成一个“叶片”的多边形点集。"""
    pts = []
    for t in range(0, 180, 20):
        rad = math.radians(t)
        x = int(cx + r * math.cos(rad))
        y = int(cy - 0.5 * r * math.sin(rad))
        pts.append((x, y))
    for t in range(180, 360, 20):
        rad = math.radians(t)
        x = int(cx + 0.5 * r * math.cos(rad))
        y = int(cy - r * math.sin(rad))
        pts.append((x, y))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--num_per_concept", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    run_dir = Path(args.run) if args.run else new_run_dir()
    logger = get_logger("gen_semantic_images")
    log_jsonl = run_dir / "logs.jsonl"

    items = []
    with open(args.concepts, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))

    for it in items:
        cid = it.get("id")
        if not cid:
            continue
        for k in range(args.num_per_concept):
            W, H = 96, 96
            rgb = blank_rgb(W, H, 255, 255, 255)
            # 固定图标模板以提升语义对应性
            draw_icon(cid, rgb, W, H)
            subdir = out_dir / cid
            ensure_dir(subdir)
            path = subdir / f"{cid}_{k}.ppm"
            write_ppm(path, rgb)
            write_jsonl(log_jsonl, {
                "time": time.time(), "concept_id": cid, "file": str(path),
                "template": cid
            })
            logger.info(f"Wrote {path}")

    logger.info(f"done | run={run_dir}")


if __name__ == "__main__":
    main()
