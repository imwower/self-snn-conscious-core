#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本（零依赖）：
- 读取 config/default.toml
- 加载 triples.jsonl，采样小批
- 文本/图像编码 → 模态投影 → 共享表征
- 多视角 InfoNCE + 一致性/中心/稀疏/速率正则（简化）
- SGD+动量 更新模态投影矩阵
- JSONL 日志 + checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path
import sys
from typing import List, Dict

 # 兼容直接以 `python scripts/train.py` 运行：把仓库根目录加入 sys.path（导入 self_core）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_core.utils.io import get_logger, new_run_dir, read_config, write_jsonl, ensure_dir
from self_core.encoding.system_font import render_text_to_bitmap, find_chinese_font
# 不再需要核心模块（仅中英对齐）
from self_core.losses.contrastive import info_nce_loss, cosine_sim, l2, center_loss


def load_triples(path: Path) -> List[Dict]:
    arr = []
    if not path.exists():
        return arr
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                arr.append(json.loads(line))
    return arr


def batch_iter(triples: List[Dict], batch: int):
    i = 0
    n = len(triples)
    while True:
        if n == 0:
            yield []
        bs = []
        for _ in range(batch):
            if n == 0:
                break
            rec = triples[i % n]
            i += 1
            bs.append(rec)
        yield bs


def normalize(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x*x for x in v)) + 1e-9
    return [x/s for x in v]


def outer(u: List[float], v: List[float]) -> List[List[float]]:
    return [[ui*vi for vi in v] for ui in u]


def apply_update(W: List[List[float]], G: List[List[float]], lr: float, mom: float, V: List[List[float]]):
    for i in range(len(W)):
        for j in range(len(W[0])):
            V[i][j] = mom*V[i][j] + (1-mom)*G[i][j]
            W[i][j] -= lr * V[i][j]


def project(x: List[float], W: List[List[float]]) -> List[float]:
    return [sum(x[k]*W[k][j] for k in range(len(x))) for j in range(len(W[0]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.toml")
    ap.add_argument("--run", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    cfg = read_config(Path(args.config))
    run_dir = Path(args.run) if args.run else new_run_dir()
    logs = run_dir / "logs.jsonl"
    ckpt = run_dir / "ckpt.pkl"
    ensure_dir(run_dir)

    logger = get_logger("train")
    logger.info(f"run_dir={run_dir}")

    # 数据
    train = load_triples(Path(cfg["data"]["train"]))
    val = load_triples(Path(cfg["data"]["val"]))

    if not train:
        logger.info("no train data, please run build_dataset.py")
        return

    # 使用系统字体渲染文本为灰度位图
    font_path = find_chinese_font()
    if not font_path:
        logger.info("No system font found. Please specify a CJK-capable font on this machine.")

    D = int(cfg["core"]["dim"])
    in_text = 256

    def rand_mat(m, n):
        scale = 1.0 / math.sqrt(m)
        return [[(random.random()*2-1)*scale for _ in range(n)] for _ in range(m)]

    W_zh = rand_mat(in_text, D)
    W_en = rand_mat(in_text, D)
    V_zh = rand_mat(in_text, D)
    V_en = rand_mat(in_text, D)
    # 仅中英训练，不使用图像流与核心模块

    steps_per_epoch = int(cfg["train"].get("steps_per_epoch", 80))
    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"])
    mom = float(cfg["train"]["momentum"])  # noqa: F841
    log_every = int(cfg["train"]["log_every"])
    eval_every = int(cfg["train"]["eval_every"])
    # 调度与初始温度/学习率
    temp = float(cfg["loss"]["temperature"])  # 初始温度
    base_lr = float(cfg["train"]["lr"])       # 初始学习率
    current_lr = base_lr
    schedule = cfg.get("schedule", {})
    lr_milestones = list(schedule.get("lr_milestones", []))
    lr_gamma = float(schedule.get("lr_gamma", 0.5)) if schedule.get("lr_milestones") else 1.0
    temp_milestones = list(schedule.get("temp_milestones", []))
    temp_gamma = float(schedule.get("temp_gamma", 0.9)) if schedule.get("temp_milestones") else 1.0

    bgen = batch_iter(train, batch_size)

    def vec_reduce(v: List[float], target: int = 256) -> List[float]:
        if len(v) == target:
            return v[:]
        out = [0.0 for _ in range(target)]
        for i, x in enumerate(v):
            out[i % target] += x
        s = math.sqrt(sum(x*x for x in out)) + 1e-9
        return [x/s for x in out]

    # 文本位图→向量 缓存，避免每步重复渲染/展平/归一化
    _text_vec_cache: Dict[str, List[float]] = {}
    # 持久化缓存（JSONL）：data/processed/text_cache.jsonl
    cache_file = Path("data/processed/text_cache.jsonl")
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
                            _text_vec_cache.setdefault(key, vec)
                    except Exception:
                        pass
        except Exception:
            pass

    def _gray_bitmap_to_vec(bm: List[List[int]]) -> List[float]:
        out: List[float] = []
        for row in bm:
            for v in row:
                out.append((255 - v) / 255.0)
        return out

    def text_to_vec_cached(text: str) -> List[float]:
        """将文本渲染为灰度位图并转成归一化后的 256 维向量，带内存缓存。"""
        key = f"{text}|{font_path}|28|2|1|0"
        if key in _text_vec_cache:
            return _text_vec_cache[key]
        bm = render_text_to_bitmap(text, font_path=font_path, size=28, padding=2, stroke_width=1, stroke_fill=0)
        v = vec_reduce(_gray_bitmap_to_vec(bm))
        _text_vec_cache[key] = v
        # 追加写入持久化缓存
        try:
            ensure_dir(cache_file.parent)
            with open(cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "vec": v}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return v

    def info_nce_sym_grads(A: List[List[float]], B: List[List[float]], temperature: float) -> (List[List[float]], List[List[float]], float):
        """对称 InfoNCE 梯度近似（忽略归一化反传）：
        - 返回对 zA 与 zB 的梯度，以及对称 InfoNCE 损失的标量值。
        """
        N = min(len(A), len(B))
        if N == 0:
            return [[0.0 for _ in range(len(A[0]) if A else 0)] for _ in range(0)], [[0.0 for _ in range(len(B[0]) if B else 0)] for _ in range(0)], 0.0
        D = len(A[0])
        # 相似度矩阵（已假设 A,B 为单位向量）
        S = [[sum(A[i][k]*B[j][k] for k in range(D)) for j in range(N)] for i in range(N)]
        # 行 softmax（A→B）
        def softmax_row(z):
            m = max(z)
            ex = [math.exp((x/temperature) - (m/temperature)) for x in z]
            s = sum(ex) + 1e-9
            return [x/s for x in ex]
        P_row = [softmax_row(S[i]) for i in range(N)]
        # 列 softmax（B→A）
        P_col = []
        for j in range(N):
            col = [S[i][j] for i in range(N)]
            m = max(col)
            ex = [math.exp((x/temperature) - (m/temperature)) for x in col]
            s = sum(ex) + 1e-9
            P_col.append([x/s for x in ex])  # 索引 [i]

        gA = [[0.0 for _ in range(D)] for _ in range(N)]
        gB = [[0.0 for _ in range(D)] for _ in range(N)]
        # A→B 方向梯度
        for i in range(N):
            # dL/dzA_i = (Σ_j p_ij zB_j - zB_i)/T
            tmp = [0.0 for _ in range(D)]
            for j in range(N):
                pij = P_row[i][j]
                for k in range(D):
                    tmp[k] += pij * B[j][k]
                    gB[j][k] += pij * A[i][k] / max(1e-9, temperature)
            for k in range(D):
                gA[i][k] += (tmp[k] - B[i][k]) / max(1e-9, temperature)
                gB[i][k] -= A[i][k] / max(1e-9, temperature)
        # B→A 方向梯度
        for j in range(N):
            tmpB = [0.0 for _ in range(D)]
            for i in range(N):
                pji = P_col[j][i]
                for k in range(D):
                    tmpB[k] += pji * A[i][k]
                    gA[i][k] += pji * B[j][k] / max(1e-9, temperature)
            for k in range(D):
                gB[j][k] += (tmpB[k] - A[j][k]) / max(1e-9, temperature)
                gA[j][k] -= B[j][k] / max(1e-9, temperature)
        # 对称损失 = (A→B + B→A)/2
        def ce_row(ps):
            return -math.log(ps + 1e-9)
        loss_a2b = sum(ce_row(P_row[i][i]) for i in range(N)) / max(1, N)
        loss_b2a = 0.0
        for j in range(N):
            loss_b2a += -math.log(P_col[j][j] + 1e-9)
        loss_b2a /= max(1, N)
        loss_sym = 0.5*(loss_a2b + loss_b2a)
        return gA, gB, loss_sym

    step = 0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            batch = next(bgen)
            zh_vecs: List[List[float]] = []
            en_vecs: List[List[float]] = []
            for rec in batch:
                zh_vecs.append(text_to_vec_cached(rec["zh"]))
                en_vecs.append(text_to_vec_cached(rec["en"]))
                # 不再读取图片

            z_zh = [normalize(project(v, W_zh)) for v in zh_vecs]
            z_en = [normalize(project(v, W_en)) for v in en_vecs]
            # 不再使用图像嵌入

            w_align = float(cfg["loss"]["align_weight"])  # noqa: F841
            w_agree = float(cfg["loss"]["agree_weight"])  # noqa: F841
            w_center = float(cfg["loss"]["center_weight"])  # noqa: F841
            # 对称 InfoNCE 损失与梯度（完整负样本）
            gZ_zh, gZ_en, loss_align = info_nce_sym_grads(z_zh, z_en, temperature=temp)
            loss_agree = sum(l2(a, b) for a, b in zip(z_zh, z_en)) / max(1, len(z_zh))
            center = [0.0 for _ in range(D)]
            for v in (z_zh + z_en):
                for i in range(D):
                    center[i] += v[i]
            center = [x / max(1, len(z_zh) * 2) for x in center]
            loss_center = center_loss(z_zh + z_en, center)

            G_zh = [[0.0 for _ in range(D)] for _ in range(in_text)]
            G_en = [[0.0 for _ in range(D)] for _ in range(in_text)]
            for i in range(len(batch)):
                oz = outer(zh_vecs[i], gZ_zh[i])
                oe = outer(en_vecs[i], gZ_en[i])
                for r in range(in_text):
                    for c in range(D):
                        G_zh[r][c] += oz[r][c]
                        G_en[r][c] += oe[r][c]
            # 学习率调度：按步里程碑调整 current_lr
            if step in lr_milestones and lr_gamma != 1.0:
                current_lr *= lr_gamma
                write_jsonl(logs, {"time": time.time(), "schedule": True, "lr": current_lr, "step": step})
            # 温度调度：按步里程碑调整 temp
            if step in temp_milestones and temp_gamma != 1.0:
                temp *= temp_gamma
                write_jsonl(logs, {"time": time.time(), "schedule": True, "temperature": temp, "step": step})

            apply_update(W_zh, G_zh, lr=current_lr, mom=float(cfg["train"]["momentum"]), V=V_zh)
            apply_update(W_en, G_en, lr=current_lr, mom=float(cfg["train"]["momentum"]), V=V_en)

            step += 1
            if step % log_every == 0:
                write_jsonl(logs, {
                    "time": time.time(),
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss_align + loss_agree + loss_center),
                    "loss_align": float(loss_align),
                    "loss_agree": float(loss_agree),
                    "loss_center": float(loss_center),
                    "rates": 0.0,
                    "lr": current_lr,
                    "temperature": temp
                })
                logger.info(f"step={step} loss={loss_align + loss_agree + loss_center:.4f} align={loss_align:.4f}")

            if step % eval_every == 0 and val:
                vrec = random.choice(val)
                zzh = normalize(project(text_to_vec_cached(vrec["zh"]), W_zh))
                zen = normalize(project(text_to_vec_cached(vrec["en"]), W_en))
                cs = cosine_sim(zzh, zen)
                write_jsonl(logs, {"time": time.time(), "eval": True, "cos_zh_en": cs})
                logger.info(f"eval cos(zh,en)={cs:.3f}")

        with open(ckpt, "wb") as f:
            pickle.dump({
                "W_zh": W_zh, "W_en": W_en
            }, f)
        logger.info(f"checkpoint saved: {ckpt}")

    logger.info("train done")


if __name__ == "__main__":
    main()
