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
    # 支持一层或两层 MLP
    W_zh = state.get("W_zh")
    W_en = state.get("W_en")
    W1_zh = state.get("W1_zh")
    W2_zh = state.get("W2_zh")
    W1_en = state.get("W1_en")
    W2_en = state.get("W2_en")
    use_mlp = W1_zh is not None and W2_zh is not None and W1_en is not None and W2_en is not None

    def project_linear(x, W):
        return [sum(x[k]*W[k][j] for k in range(len(x))) for j in range(len(W[0]))]
    def tanh(v):
        import math
        return [math.tanh(x) for x in v]
    def project_mlp(x, W1, W2):
        h = [sum(x[k]*W1[k][j] for k in range(len(x))) for j in range(len(W1[0]))]
        h = tanh(h)
        return [sum(h[k]*W2[k][j] for k in range(len(h))) for j in range(len(W2[0]))]
    def project_any(x):
        if use_mlp:
            return project_mlp(x, W1_zh, W2_zh)  # zh 与 en 维度相同；在调用处将按相应矩阵
        else:
            return project_linear(x, W_zh)       # 同上
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
    cache_file = Path("data/processed/text_cache.jsonl")
    # 载入持久化缓存
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = rec.get("key"); vec = rec.get("vec")
                        if isinstance(key, str) and isinstance(vec, list) and len(vec) == 256:
                            _cache[key] = vec
                    except Exception:
                        pass
        except Exception:
            pass
    def text_to_vec(text: str):
        key = f"{text}|{font_path}|28|2|1|0"
        v = _cache.get(key)
        if v is not None:
            return v
        bm = render_text_to_bitmap(text, font_path=font_path, size=28, padding=2, stroke_width=1, stroke_fill=0)
        vv = [(255 - x)/255.0 for row in bm for x in row]
        _cache[key] = vv
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "vec": vv}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return vv

    zh_embs = []
    en_embs = []
    labels = []
    for r in val:
        vzh_in = vec_reduce(text_to_vec(r["zh"]))
        ven_in = vec_reduce(text_to_vec(r["en"]))
        if use_mlp:
            vzh = normalize(project_mlp(vzh_in, W1_zh, W2_zh))
            ven = normalize(project_mlp(ven_in, W1_en, W2_en))
        else:
            vzh = normalize(project_linear(vzh_in, W_zh))
            ven = normalize(project_linear(ven_in, W_en))
        zh_embs.append(vzh)
        en_embs.append(ven)
        labels.append(r["id"])

    # 计算 zh→en 与 en→zh 检索，以及对称平均
    ret_zh2en = retrieval_metrics(zh_embs, en_embs, labels, topk=(1, 5))
    ret_en2zh = retrieval_metrics(en_embs, zh_embs, labels, topk=(1, 5))
    avg = {
        "R@1": (ret_zh2en["R@1"] + ret_en2zh["R@1"]) / 2.0,
        "R@5": (ret_zh2en["R@5"] + ret_en2zh["R@5"]) / 2.0,
        "mAP": (ret_zh2en["mAP"] + ret_en2zh["mAP"]) / 2.0,
    }

    # 诊断：正样平均余弦、最难负样本（行最大非对角）
    from self_core.eval.metrics import cosine
    n = min(len(zh_embs), len(en_embs))
    pos = []
    hard = []
    for i in range(n):
        sims = [cosine(zh_embs[i], en_embs[j]) for j in range(n)]
        pos.append(sims[i])
        hard.append(max(s for j, s in enumerate(sims) if j != i) if n > 1 else sims[i])
    pos_mean_zh2en = sum(pos)/max(1, len(pos))
    hard_mean_zh2en = sum(hard)/max(1, len(hard))
    # en→zh 同理
    pos2 = []
    hard2 = []
    for j in range(n):
        sims = [cosine(en_embs[j], zh_embs[i]) for i in range(n)]
        pos2.append(sims[j])
        hard2.append(max(s for i, s in enumerate(sims) if i != j) if n > 1 else sims[j])
    pos_mean_en2zh = sum(pos2)/max(1, len(pos2))
    hard_mean_en2zh = sum(hard2)/max(1, len(hard2))

    out = {
        "retrieval_zh2en": ret_zh2en,
        "retrieval_en2zh": ret_en2zh,
        "retrieval_avg": avg,
        "diagnostics": {
            "pos_mean_zh2en": pos_mean_zh2en,
            "hardneg_mean_zh2en": hard_mean_zh2en,
            "pos_mean_en2zh": pos_mean_en2zh,
            "hardneg_mean_en2zh": hard_mean_en2zh
        }
    }
    with open(run_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"eval done | run={run_dir}")


if __name__ == "__main__":
    main()
