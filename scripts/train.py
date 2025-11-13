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
from typing import List, Dict

from self_core.utils.io import get_logger, new_run_dir, read_config, write_jsonl, ensure_dir
from self_core.encoding.vision_draw import load_image_to_vec
from self_core.encoding.system_font import render_text_to_bitmap, find_chinese_font
from self_core.core.conscious_core import ConsciousCore
from self_core.losses.contrastive import info_nce_loss, cosine_sim, l2, center_loss, avg_rate_penalty, sparsity_penalty


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
    in_img = 256

    def rand_mat(m, n):
        scale = 1.0 / math.sqrt(m)
        return [[(random.random()*2-1)*scale for _ in range(n)] for _ in range(m)]

    W_zh = rand_mat(in_text, D)
    W_en = rand_mat(in_text, D)
    W_img_text = rand_mat(in_img, D)
    W_img_sem = rand_mat(in_img, D)
    V_zh = rand_mat(in_text, D)
    V_en = rand_mat(in_text, D)
    V_img_text = rand_mat(in_img, D)
    V_img_sem = rand_mat(in_img, D)

    core = ConsciousCore(dim=D, prototypes=int(cfg["core"]["prototypes"]))

    steps_per_epoch = int(cfg["train"].get("steps_per_epoch", 80))
    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"])
    mom = float(cfg["train"]["momentum"])  # noqa: F841
    log_every = int(cfg["train"]["log_every"])
    eval_every = int(cfg["train"]["eval_every"])
    temp = float(cfg["loss"]["temperature"])  # noqa: F841

    bgen = batch_iter(train, batch_size)

    def vec_reduce(v: List[float], target: int = 256) -> List[float]:
        if len(v) == target:
            return v[:]
        out = [0.0 for _ in range(target)]
        for i, x in enumerate(v):
            out[i % target] += x
        s = math.sqrt(sum(x*x for x in out)) + 1e-9
        return [x/s for x in out]

    step = 0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            batch = next(bgen)
            zh_vecs: List[List[float]] = []
            en_vecs: List[List[float]] = []
            it_vecs: List[List[float]] = []
            is_vecs: List[List[float]] = []
            def gray_bitmap_to_vec(bm: List[List[int]]) -> List[float]:
                # 将 0..255 灰度二维数组拉平成 0..1 向量
                out: List[float] = []
                for row in bm:
                    for v in row:
                        out.append((255 - v) / 255.0)  # 黑色更大
                return out

            for rec in batch:
                zh_bm = render_text_to_bitmap(rec["zh"], font_path=font_path, size=28, padding=2, stroke_width=1, stroke_fill=0)
                en_bm = render_text_to_bitmap(rec["en"], font_path=font_path, size=28, padding=2, stroke_width=1, stroke_fill=0)
                zh_vecs.append(vec_reduce(gray_bitmap_to_vec(zh_bm)))
                en_vecs.append(vec_reduce(gray_bitmap_to_vec(en_bm)))
                it_path = Path(rec["img_text"])  # 文字图片
                if it_path.exists():
                    it_vecs.append(vec_reduce(load_image_to_vec(it_path)))
                else:
                    # 若缺失，则用系统字体重新渲染英文短语近似替代
                    it_bm = render_text_to_bitmap(rec["en"], font_path=font_path, size=28, padding=2, stroke_width=1, stroke_fill=0)
                    it_vecs.append(vec_reduce(gray_bitmap_to_vec(it_bm)))
                is_vecs.append(vec_reduce(load_image_to_vec(Path(rec["img_sem"]))))

            z_zh = [normalize(project(v, W_zh)) for v in zh_vecs]
            z_en = [normalize(project(v, W_en)) for v in en_vecs]
            z_it = [normalize(project(v, W_img_text)) for v in it_vecs]
            z_is = [normalize(project(v, W_img_sem)) for v in is_vecs]

            core_out = core.forward_batch(z_is)

            w_align = float(cfg["loss"]["align_weight"])  # noqa: F841
            w_agree = float(cfg["loss"]["agree_weight"])  # noqa: F841
            w_center = float(cfg["loss"]["center_weight"])  # noqa: F841
            w_sparse = float(cfg["loss"]["sparse_weight"])  # noqa: F841
            w_rate = float(cfg["loss"]["rate_weight"])  # noqa: F841

            loss_align = info_nce_loss(z_zh, z_is, temperature=temp) \
                       + info_nce_loss(z_en, z_is, temperature=temp) \
                       + info_nce_loss(z_zh, z_en, temperature=temp)
            loss_agree = sum(l2(a, b) for a, b in zip(z_zh, z_en)) / max(1, len(z_zh))
            center = [0.0 for _ in range(D)]
            for v in (z_zh + z_en + z_it + z_is):
                for i in range(D):
                    center[i] += v[i]
            center = [x / max(1, len(z_zh) * 4) for x in center]
            loss_center = center_loss(z_zh + z_en + z_it + z_is, center)
            loss_sparse = sparsity_penalty(core_out["sparse_codes"])  # noqa: F841
            loss_rate = avg_rate_penalty(core_out["rates"], target=1.0)  # noqa: F841

            # 简化：只对 zh<->is, en<->is 正样做梯度近似
            def grad_cos(u: List[float], v: List[float]) -> List[float]:
                c = cosine_sim(u, v)
                return [vj - c * uj for uj, vj in zip(u, v)]

            G_zh = [[0.0 for _ in range(D)] for _ in range(in_text)]
            G_en = [[0.0 for _ in range(D)] for _ in range(in_text)]
            G_is = [[0.0 for _ in range(D)] for _ in range(in_img)]

            for i in range(len(batch)):
                gz = grad_cos(z_zh[i], z_is[i])
                ge = grad_cos(z_en[i], z_is[i])
                # x ⊗ dz
                oz = outer(zh_vecs[i], gz)
                oe = outer(en_vecs[i], ge)
                oi = outer(is_vecs[i], [-(g) for g in gz])
                oi2 = outer(is_vecs[i], [-(g) for g in ge])
                for r in range(in_text):
                    for c in range(D):
                        G_zh[r][c] += oz[r][c]
                        G_en[r][c] += oe[r][c]
                for r in range(in_img):
                    for c in range(D):
                        G_is[r][c] += (oi[r][c] + oi2[r][c]) * 0.5

            apply_update(W_zh, G_zh, lr=float(cfg["train"]["lr"]), mom=float(cfg["train"]["momentum"]), V=V_zh)
            apply_update(W_en, G_en, lr=float(cfg["train"]["lr"]), mom=float(cfg["train"]["momentum"]), V=V_en)
            apply_update(W_img_sem, G_is, lr=float(cfg["train"]["lr"]), mom=float(cfg["train"]["momentum"]), V=V_img_sem)

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
                    "loss_sparse": float(loss_sparse),
                    "loss_rate": float(loss_rate),
                    "rates": core_out["rates"]
                })
                logger.info(f"step={step} loss={loss_align + loss_agree + loss_center:.4f} align={loss_align:.4f}")

            if step % eval_every == 0 and val:
                vrec = random.choice(val)
                zh_bm = render_text_to_bitmap(vrec["zh"], font_path=font_path, size=28, padding=2, stroke_width=1, stroke_fill=0)
                is_vec = vec_reduce(load_image_to_vec(Path(vrec["img_sem"])))
                zzh = normalize(project(vec_reduce(gray_bitmap_to_vec(zh_bm)), W_zh))
                zis = normalize(project(is_vec, W_img_sem))
                cs = cosine_sim(zzh, zis)
                write_jsonl(logs, {"time": time.time(), "eval": True, "cos_zh_imgsem": cs})
                logger.info(f"eval cos(zh,img_sem)={cs:.3f}")

        with open(ckpt, "wb") as f:
            pickle.dump({
                "W_zh": W_zh, "W_en": W_en, "W_img_text": W_img_text, "W_img_sem": W_img_sem,
                "core": core.state_dict()
            }, f)
        logger.info(f"checkpoint saved: {ckpt}")

    logger.info("train done")


if __name__ == "__main__":
    main()
