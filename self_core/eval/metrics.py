# -*- coding: utf-8 -*-
"""
评测指标：
- 检索：R@K、mAP
- 聚类：Silhouette、Davies–Bouldin、Fisher
- SNN：Victor–Purpura（简化）、能耗
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


def cosine(u: List[float], v: List[float]) -> float:
    nu = math.sqrt(sum(x*x for x in u)) + 1e-9
    nv = math.sqrt(sum(x*x for x in v)) + 1e-9
    return sum(x*y for x, y in zip(u, v)) / (nu*nv)


def retrieval_metrics(Q: List[List[float]], K: List[List[float]], labels: List[str], topk: Tuple[int, int] = (1, 5)) -> Dict:
    n = min(len(Q), len(K))
    if n == 0:
        return {"R@1": 0.0, "R@5": 0.0, "mAP": 0.0}
    r1 = 0
    r5 = 0
    ap_sum = 0.0
    for i in range(n):
        sims = [(j, cosine(Q[i], K[j])) for j in range(n)]
        sims.sort(key=lambda x: x[1], reverse=True)
        if sims[0][0] == i:
            r1 += 1
        if any(j == i for j, _ in sims[:min(5, n)]):
            r5 += 1
        rank = 1 + next((ri for ri, (j, _) in enumerate(sims) if j == i), n)
        ap_sum += 1.0 / rank
    return {"R@1": r1 / n, "R@5": r5 / n, "mAP": ap_sum / n}


def silhouette(X: List[List[float]], y: List[str]) -> float:
    if not X:
        return 0.0
    n = len(X)
    s = 0.0
    for i in range(n):
        a = []
        b = []
        for j in range(n):
            if i == j:
                continue
            d = 1.0 - cosine(X[i], X[j])
            if y[i] == y[j]:
                a.append(d)
            else:
                b.append(d)
        ai = sum(a)/max(1, len(a))
        bi = sum(b)/max(1, len(b))
        s += (bi - ai) / max(bi, ai, 1e-9)
    return s / n


def davies_bouldin(X: List[List[float]], y: List[str]) -> float:
    labels = sorted(set(y))
    cls = {c: [X[i] for i, yi in enumerate(y) if yi == c] for c in labels}
    centers = {c: [sum(v[k] for v in vs)/max(1, len(vs)) for k in range(len(X[0]))] for c, vs in cls.items()}
    S = {c: sum(1.0 - cosine(v, centers[c]) for v in vs)/max(1, len(vs)) for c, vs in cls.items()}
    R = 0.0
    m = 0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            ci, cj = labels[i], labels[j]
            Mij = 1.0 - cosine(centers[ci], centers[cj]) + 1e-9
            R += (S[ci] + S[cj]) / Mij
            m += 1
    return R / max(1, m)


def fisher_ratio(X: List[List[float]], y: List[str]) -> float:
    labels = sorted(set(y))
    overall = [sum(v[k] for v in X)/max(1, len(X)) for k in range(len(X[0]))]
    between = 0.0
    within = 0.0
    for c in labels:
        cls = [X[i] for i, yi in enumerate(y) if yi == c]
        center = [sum(v[k] for v in cls)/max(1, len(cls)) for k in range(len(X[0]))]
        between += sum((center[k] - overall[k])**2 for k in range(len(X[0])))
        within += sum(sum((v[k] - center[k])**2 for k in range(len(X[0]))) for v in cls)
    return between / max(1e-9, within)


def clustering_metrics(X: List[List[float]], y: List[str]) -> Dict:
    return {
        "silhouette": silhouette(X, y),
        "davies_bouldin": davies_bouldin(X, y),
        "fisher": fisher_ratio(X, y)
    }


def vp_distance(S1: List[List[int]], S2: List[List[int]], q: float = 1.0) -> float:
    if not S1 or not S2:
        return 0.0
    n = min(len(S1), len(S2))
    T = min(len(S1[0]), len(S2[0]))
    d = 0.0
    for i in range(n):
        for t in range(T):
            d += abs((S1[i][t] if t < len(S1[i]) else 0) - (S2[i][t] if t < len(S2[i]) else 0))
    return d / (n * T)


def energy_metrics(spikes_per_step: List[int], steps: int) -> Dict:
    total_spikes = sum(spikes_per_step)
    return {"total_spikes": total_spikes, "steps": steps, "avg_per_step": total_spikes / max(1, steps)}

