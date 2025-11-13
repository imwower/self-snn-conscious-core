#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文字图片生成器（PGM/PPM），零依赖。
- 读取概念清单（JSONL）
- 将中英文短语渲染为位图，支持缩放、微旋转、噪声与纹理背景
- 输出 PGM/PPM 文件
- 每条生成操作写入 JSONL 日志
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Dict, Any

from self_core.utils.io import ensure_dir, write_pgm, write_ppm, get_logger, new_run_dir, write_jsonl
from self_core.encoding.system_font import render_text_to_bitmap, find_chinese_font


def make_texture_bg(w: int, h: int) -> List[List[int]]:
    """生成简单纹理背景（棋盘/轻噪声），降低对文字干扰。"""
    bg = [[255 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if ((x // 8) + (y // 8)) % 2 == 0:
                bg[y][x] = 245
            if random.random() < 0.002:
                bg[y][x] = 230
    return bg


def composite(img: List[List[int]], bg: List[List[int]], ox: int, oy: int) -> List[List[int]]:
    """将前景 img 覆盖到背景 bg 的 (ox,oy)。灰度 0..255 采用 min 合成，保留反锯齿边缘。"""
    H = len(bg)
    W = len(bg[0]) if H else 0
    for y in range(len(img)):
        for x in range(len(img[0])):
            ty, tx = oy + y, ox + x
            if 0 <= ty < H and 0 <= tx < W:
                v = img[y][x]
                if v < bg[ty][tx]:
                    bg[ty][tx] = v
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=str, required=True, help="概念 JSONL（每行包含 id, zh[], en[]）")
    ap.add_argument("--out", type=str, required=True, help="输出目录，按概念写入图片")
    ap.add_argument("--num_per_phrase", type=int, default=4, help="每个短语生成的样本数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_system_font", type=int, default=1, help="使用系统字体渲染文本(需 Pillow)。1=启用, 0=禁用")
    ap.add_argument("--font_path", type=str, default="", help="系统字体文件路径（可选）")
    ap.add_argument("--font_size", type=int, default=36, help="系统字体字号（像素）")
    ap.add_argument("--stroke_width", type=int, default=1, help="系统字体描边像素（增强清晰度）")
    ap.add_argument("--run", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    run_dir = Path(args.run) if args.run else new_run_dir()
    logger = get_logger("gen_text_images")
    log_jsonl = run_dir / "logs.jsonl"

    # 不再依赖内置位图字体，统一使用系统字体渲染

    items = []
    with open(args.concepts, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))

    for it in items:
        cid = it.get("id", f"c_{random.randint(0,9999)}")
        zh_list = it.get("zh", [])
        en_list = it.get("en", [])
        for phrase in zh_list + en_list:
            for idx in range(args.num_per_phrase):
                is_color = random.random() < 0.5
                # 统一系统字体渲染
                font_path = args.font_path or find_chinese_font()
                if not font_path:
                    raise FileNotFoundError("No system font found. Please specify --font_path to a TTF/TTC/OTF file.")
                bm = render_text_to_bitmap(phrase, font_path=font_path, size=int(args.font_size), padding=2, stroke_width=int(args.stroke_width), stroke_fill=0)

                # 合成到背景
                H = max(64, len(bm) + 8)
                W = max(64, len(bm[0]) + 8) if bm else 64
                bg = make_texture_bg(W, H)
                ox = max(2, (W - (len(bm[0]) if bm else 0)) // 2)
                oy = max(2, (H - len(bm)) // 2)
                # 系统字体灰度直接合成
                img = composite(bm, bg, ox, oy)

                subdir = out_dir / cid
                ensure_dir(subdir)
                stem = f"{cid}_{phrase.replace(' ', '_')}_{idx}"
                if is_color:
                    # 转彩色（灰度复制到 RGB）
                    rgb = [[[pix, pix, pix] for pix in row] for row in img]
                    path = subdir / f"{stem}.ppm"
                    write_ppm(path, rgb)
                else:
                    path = subdir / f"{stem}.pgm"
                    write_pgm(path, img)

                rec = {
                    "time": time.time(),
                    "concept_id": cid,
                    "phrase": phrase,
                    "file": str(path),
                    "scale": scale,
                    "angle": angle,
                    "color": is_color,
                    "seed": args.seed
                }
                write_jsonl(log_jsonl, rec)
                logger.info(f"Wrote {path}")

    logger.info(f"done | run={run_dir}")


if __name__ == "__main__":
    main()
