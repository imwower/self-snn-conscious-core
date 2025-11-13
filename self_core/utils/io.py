# -*- coding: utf-8 -*-
"""
IO 与日志工具：
- JSONL/TOML 读写
- PGM/PPM 写入（P5/P6）
- 随机种子与 run 目录
- 统一日志（控制台 INFO）
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def new_run_dir() -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    p = Path("runs") / ts
    ensure_dir(p)
    return p


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def read_config(path: Path) -> Dict[str, Any]:
    """
    优先读取 TOML；若失败尝试 JSON；返回 dict。
    """
    if path.suffix.lower() == ".toml" and tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)
    # 回退 JSON
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_pgm(path: Path, img: List[List[int]]) -> None:
    """
    写 PGM (P5) 二进制。img[h][w] 取 0..255。
    """
    h = len(img)
    w = len(img[0]) if h else 0
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        header = f"P5\n{w} {h}\n255\n"
        f.write(header.encode("ascii"))
        for row in img:
            f.write(bytearray(max(0, min(255, int(px))) for px in row))


def write_ppm(path: Path, img: List[List[List[int]]]) -> None:
    """
    写 PPM (P6) 二进制。img[h][w][3] 取 0..255。
    """
    h = len(img)
    w = len(img[0]) if h else 0
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        header = f"P6\n{w} {h}\n255\n"
        f.write(header.encode("ascii"))
        for row in img:
            for r, g, b in row:
                f.write(bytes((max(0, min(255, int(r))),
                               max(0, min(255, int(g))),
                               max(0, min(255, int(b))))))


def load_pgm_ppm_to_gray_vec(path: Path) -> List[float]:
    """
    读取 PGM/PPM 为灰度向量（0..1）。简化解析，适配本项目写出的图像。
    """
    with open(path, "rb") as f:
        head = f.readline().strip()
        if head == b"P5":
            dims = f.readline()
            while dims.startswith(b"#"):
                dims = f.readline()
            w, h = [int(x) for x in dims.strip().split()]
            maxv = int(f.readline().strip())
            data = f.read()
            out = [px / max(1, maxv) for px in data]
            return out[:w*h]
        elif head == b"P6":
            dims = f.readline()
            while dims.startswith(b"#"):
                dims = f.readline()
            w, h = [int(x) for x in dims.strip().split()]
            maxv = int(f.readline().strip())
            data = f.read()
            out = []
            for i in range(0, len(data), 3):
                r, g, b = data[i], data[i+1], data[i+2]
                out.append((0.299*r + 0.587*g + 0.114*b) / max(1, maxv))
            return out[:w*h]
        else:
            raise ValueError("Unknown image magic")

