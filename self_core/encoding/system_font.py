# -*- coding: utf-8 -*-
"""
系统字体渲染（可选）：使用 Pillow 调用系统中文字体将文本栅格化为灰度位图。

说明：
- 本模块仅在本地已安装 Pillow 且系统存在中文字体文件时可用；否则会抛出 ImportError 或返回 None。
- 通过 find_chinese_font() 在常见系统字体目录中搜索常用中文字体；也可直接传入 --font_path 指向具体字体文件。

用法：
    from self_core.encoding.system_font import render_text_to_bitmap, find_chinese_font
    fp = find_chinese_font()
    bm = render_text_to_bitmap("苹果", font_path=fp, size=28, padding=4)
    # bm 是 HxW 的 0..255 灰度二维列表，可直接用 write_pgm 写出
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List


def _try_import_pillow():  # 延迟导入，避免环境未安装时报错
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        return Image, ImageDraw, ImageFont
    except Exception as e:  # pragma: no cover - 环境无 Pillow 时触发
        raise ImportError("Pillow is required for system font rendering. Install via 'pip install pillow'.") from e


def find_chinese_font() -> Optional[str]:
    """在常见目录中搜索常见中文字体文件，返回第一个存在的路径，找不到则返回 None。"""
    candidates = []
    if os.name == "posix":
        # macOS 常见字体
        candidates += [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB W3.otf",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/华文黑体.ttf",
            "/Library/Fonts/Songti.ttc",
        ]
        # Linux 常见路径（Noto CJK）
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
    else:
        # Windows
        windir = os.environ.get("WINDIR", r"C:\\Windows")
        candidates += [
            os.path.join(windir, "Fonts", "simsun.ttc"),  # 宋体
            os.path.join(windir, "Fonts", "simhei.ttf"),  # 黑体
            os.path.join(windir, "Fonts", "msyh.ttc"),    # 微软雅黑
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def find_chinese_fonts(max_count: int = 8) -> List[str]:
    """返回可用的中文/支持中文的系统字体列表（最多 max_count 个）。"""
    found: List[str] = []
    candidates = []
    if os.name == "posix":
        candidates += [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB W3.otf",
            "/System/Library/Fonts/Hiragino Sans GB W6.otf",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/华文黑体.ttf",
            "/Library/Fonts/Songti.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    else:
        windir = os.environ.get("WINDIR", r"C:\\Windows")
        candidates += [
            os.path.join(windir, "Fonts", "simsun.ttc"),
            os.path.join(windir, "Fonts", "simhei.ttf"),
            os.path.join(windir, "Fonts", "msyh.ttc"),
            os.path.join(windir, "Fonts", "arial.ttf"),
        ]
    seen = set()
    for p in candidates:
        if os.path.exists(p) and p not in seen:
            found.append(p)
            seen.add(p)
            if len(found) >= max_count:
                break
    # 至少返回一个
    if not found:
        fp = find_chinese_font()
        if fp:
            found.append(fp)
    return found


def render_text_to_bitmap(text: str, font_path: Optional[str] = None, size: int = 28, padding: int = 4, stroke_width: int = 0, stroke_fill: int = 0) -> List[List[int]]:
    """
    使用系统字体栅格化中文文本为灰度位图（0..255）。
    - font_path: 字体文件路径（ttf/ttc/otf）。若为 None，将自动搜索常见中文字体。
    - size: 字号（像素）
    - padding: 边距（像素）
    返回：二维列表 HxW（0=黑，255=白）。
    """
    Image, ImageDraw, ImageFont = _try_import_pillow()

    if font_path is None:
        font_path = find_chinese_font()
    if not font_path or not os.path.exists(font_path):
        raise FileNotFoundError("No Chinese font found. Specify --font_path to a TTF/TTC file or install a CJK font.")

    try:
        font = ImageFont.truetype(font_path, size=size)
    except Exception as e:
        raise RuntimeError(f"Failed to load font: {font_path}") from e

    # 先用一个小画布测量文本尺寸
    tmp_img = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(tmp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = (bbox[2] - bbox[0]) + padding * 2
    h = (bbox[3] - bbox[1]) + padding * 2

    img = Image.new("L", (max(1, w), max(1, h)), color=255)
    draw = ImageDraw.Draw(img)
    # 使用描边增强清晰度
    try:
        draw.text((padding, padding), text, font=font, fill=0, stroke_width=stroke_width, stroke_fill=stroke_fill)
    except TypeError:
        # 旧版 Pillow 不支持 stroke 参数则退化
        draw.text((padding, padding), text, font=font, fill=0)

    # 转为二维列表（0..255）
    px = list(img.getdata())
    out: List[List[int]] = []
    for y in range(h):
        row = px[y* w:(y+1)*w]
        out.append([int(v) for v in row])
    return out
