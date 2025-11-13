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
from self_core.encoding.vision_draw import draw_ellipse, draw_line


def draw_icon(cid: str, rgb, W: int, H: int):
    """根据概念绘制更贴近物体的图标（细化版）。"""
    cx, cy = W//2, H//2
    if cid == "apple":
        # 红苹果 + 叶片 + 枝 + 高光
        draw_circle(rgb, cx, cy+6, 22, color=(220, 0, 0), fill=True)
        draw_circle(rgb, cx, cy+6, 22, color=(120, 0, 0), fill=False)
        draw_rect(rgb, cx-2, cy-10, cx+2, cy-2, color=(100, 60, 20), fill=True)
        leaf = [(cx+6, cy-12), (cx+18, cy-8), (cx+10, cy)]
        draw_polygon(rgb, leaf, color=(0, 160, 0), fill=True)
        # 高光
        draw_ellipse(rgb, cx-6, cy, 6, 3, color=(255,255,255), fill=True)
    elif cid == "banana":
        # 弯月香蕉 + 两端与斑点
        pts = []
        for t in range(-30, 210, 10):
            rad = math.radians(t)
            pts.append((cx + int(24*math.cos(rad)), cy + int(10*math.sin(rad))))
        for t in range(210, -30, -10):
            rad = math.radians(t)
            pts.append((cx + int(16*math.cos(rad)), cy + int(4*math.sin(rad))))
        draw_polygon(rgb, pts, color=(245, 220, 0), fill=True)
        draw_polygon(rgb, pts, color=(120, 100, 0), fill=False)
        # 两端
        draw_circle(rgb, cx-20, cy+2, 2, color=(110, 90, 0), fill=True)
        draw_circle(rgb, cx+20, cy-2, 2, color=(110, 90, 0), fill=True)
        # 斑点
        for i in range(12):
            draw_circle(rgb, cx-10+i, cy + ((-1)**i)*2, 1, color=(170,140,0), fill=True)
    elif cid == "cat":
        # 猫脸 + 五官 + 胡须
        draw_circle(rgb, cx, cy+6, 20, color=(200, 180, 160), fill=True)
        draw_polygon(rgb, [(cx-12, cy-2), (cx-22, cy-16), (cx-4, cy-8)], color=(200,180,160), fill=True)
        draw_polygon(rgb, [(cx+12, cy-2), (cx+22, cy-16), (cx+4, cy-8)], color=(200,180,160), fill=True)
        draw_circle(rgb, cx-6, cy+4, 2, color=(0,0,0), fill=True)
        draw_circle(rgb, cx+6, cy+4, 2, color=(0,0,0), fill=True)
        draw_triangle_nose = [(cx-2, cy+8), (cx+2, cy+8), (cx, cy+10)]
        draw_polygon(rgb, draw_triangle_nose, color=(120,80,80), fill=True)
        # 胡须
        for dy in (7, 10, 13):
            draw_line(rgb, cx-4, cy+dy, cx-16, cy+dy-2, color=(0,0,0))
            draw_line(rgb, cx+4, cy+dy, cx+16, cy+dy-2, color=(0,0,0))
    elif cid == "dog":
        # 狗：头 + 耳朵 + 眼鼻舌
        draw_circle(rgb, cx, cy+8, 20, color=(180, 140, 100), fill=True)
        draw_rect(rgb, cx-20, cy, cx-8, cy+8, color=(180, 140, 100), fill=True)
        draw_rect(rgb, cx+8, cy, cx+20, cy+8, color=(180, 140, 100), fill=True)
        draw_circle(rgb, cx-6, cy+8, 2, color=(0,0,0), fill=True)
        draw_circle(rgb, cx+6, cy+8, 2, color=(0,0,0), fill=True)
        draw_circle(rgb, cx, cy+12, 2, color=(60,40,40), fill=True)
        draw_rect(rgb, cx-2, cy+14, cx+2, cy+16, color=(220,60,60), fill=True)
    elif cid == "car":
        # 车：车身 + 窗 + 轮 + 车灯 + 线条
        draw_rect(rgb, cx-22, cy, cx+22, cy+12, color=(60,120,220), fill=True)
        draw_rect(rgb, cx-12, cy-8, cx+12, cy, color=(200,200,255), fill=True)
        draw_circle(rgb, cx-14, cy+14, 6, color=(0,0,0), fill=True)
        draw_circle(rgb, cx+14, cy+14, 6, color=(0,0,0), fill=True)
        draw_circle(rgb, cx-14, cy+14, 2, color=(200,200,200), fill=True)
        draw_circle(rgb, cx+14, cy+14, 2, color=(200,200,200), fill=True)
        draw_rect(rgb, cx-24, cy+4, cx-18, cy+8, color=(250,250,100), fill=True)
        draw_line(rgb, cx-22, cy+12, cx+22, cy+12, color=(0,0,80))
    elif cid == "tree":
        # 树：树干 + 层层树冠 + 深浅叶
        draw_rect(rgb, cx-3, cy+4, cx+3, cy+24, color=(120,70,20), fill=True)
        for r, col in [(14,(20,160,20)), (12,(30,170,30)), (10,(40,180,40))]:
            draw_circle(rgb, cx, cy, r, color=col, fill=True)
            draw_circle(rgb, cx-10, cy+4, max(6, r-6), color=col, fill=True)
            draw_circle(rgb, cx+10, cy+4, max(6, r-6), color=col, fill=True)
    elif cid == "bird":
        draw_ellipse(rgb, cx, cy+2, 24, 14, color=(80, 150, 220), fill=True)
        # 翅膀
        draw_polygon(rgb, [(cx, cy+2), (cx+6, cy+6), (cx-2, cy+8)], color=(60,120,180), fill=True)
        draw_polygon(rgb, [(cx+14, cy+2), (cx+22, cy-2), (cx+22, cy+6)], color=(240,150,0), fill=True)  # 嘴
        draw_circle(rgb, cx-8, cy-2, 2, color=(0,0,0), fill=True)
    elif cid == "fish":
        draw_ellipse(rgb, cx, cy+2, 24, 12, color=(80, 160, 200), fill=True)
        draw_polygon(rgb, [(cx-22, cy+2), (cx-32, cy-6), (cx-32, cy+10)], color=(80,160,200), fill=True)
        draw_triangle_top = [(cx-4, cy-6), (cx+2, cy-10), (cx+8, cy-6)]
        draw_triangle_bottom = [(cx-4, cy+10), (cx+2, cy+14), (cx+8, cy+10)]
        draw_polygon(rgb, draw_triangle_top, color=(70,150,190), fill=True)
        draw_polygon(rgb, draw_triangle_bottom, color=(70,150,190), fill=True)
        draw_circle(rgb, cx+10, cy-2, 2, color=(0,0,0), fill=True)
        # 鳞片点阵
        for iy in range(-6, 8, 4):
            for ix in range(-10, 12, 4):
                draw_circle(rgb, cx+ix, cy+iy, 1, color=(70,140,180), fill=True)
    elif cid == "house":
        draw_rect(rgb, cx-18, cy-2, cx+18, cy+22, color=(240,240,200), fill=True)
        draw_polygon(rgb, [(cx-20, cy-2), (cx, cy-20), (cx+20, cy-2)], color=(200,80,60), fill=True)
        draw_rect(rgb, cx-4, cy+8, cx+4, cy+22, color=(160,100,60), fill=True)
        draw_rect(rgb, cx+8, cy+4, cx+16, cy+12, color=(200,200,255), fill=True)
        draw_rect(rgb, cx-16, cy+4, cx-8, cy+12, color=(200,200,255), fill=True)
    elif cid == "book":
        draw_rect(rgb, cx-20, cy-12, cx+20, cy+14, color=(230,230,255), fill=True)
        draw_rect(rgb, cx-4, cy-12, cx-2, cy+14, color=(200,200,220), fill=True)
        # 页纹
        for yy in range(cy-10, cy+12, 3):
            draw_line(rgb, cx-18, yy, cx+18, yy, color=(210,210,240))
    elif cid == "cup":
        draw_rect(rgb, cx-14, cy-8, cx+10, cy+14, color=(240,240,240), fill=True)
        draw_rect(rgb, cx+10, cy-2, cx+18, cy+6, color=(240,240,240), fill=True)
        # 杯沿液面
        draw_rect(rgb, cx-12, cy-8, cx+8, cy-4, color=(180,120,60), fill=True)
        draw_line(rgb, cx-14, cy-8, cx+10, cy-8, color=(200,200,200))
    elif cid == "phone":
        draw_rect(rgb, cx-12, cy-20, cx+12, cy+20, color=(230,230,230), fill=True)
        draw_rect(rgb, cx-11, cy-19, cx+11, cy+19, color=(30,30,30), fill=True)
        draw_circle(rgb, cx, cy+18, 1, color=(200,200,200), fill=True)
        draw_rect(rgb, cx-4, cy-18, cx+4, cy-16, color=(180,180,180), fill=True)
    elif cid == "chair":
        draw_rect(rgb, cx-16, cy, cx+16, cy+6, color=(180,120,60), fill=True)  # 坐面
        draw_rect(rgb, cx-16, cy-16, cx-12, cy, color=(180,120,60), fill=True)  # 背左
        draw_rect(rgb, cx+12, cy-16, cx+16, cy, color=(180,120,60), fill=True)  # 背右
        draw_rect(rgb, cx-16, cy+6, cx-12, cy+22, color=(140,90,40), fill=True)
        draw_rect(rgb, cx+12, cy+6, cx+16, cy+22, color=(140,90,40), fill=True)
    elif cid == "table":
        draw_rect(rgb, cx-26, cy-4, cx+26, cy+2, color=(160,100,60), fill=True)
        draw_rect(rgb, cx-24, cy+2, cx-20, cy+24, color=(160,100,60), fill=True)
        draw_rect(rgb, cx+20, cy+2, cx+24, cy+24, color=(160,100,60), fill=True)
    else:
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
