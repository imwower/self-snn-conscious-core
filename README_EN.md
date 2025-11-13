#!/usr/bin/env
# self-snn-conscious-core

[中文文档 / Chinese README](README.md)

A pure-standard-library (no third-party deps) scaffold for learning a shared semantic space (“conscious core”) across three views: Chinese/English text, text images (OCR-like), and semantic images. Samples with the same meaning should assemble into the same prototype cluster and produce consistent outputs.

In the absence of inputs (Free-Play/Sleep), the core can exhibit spontaneous activity, replay anchored by recent high-confidence prototypes, and auto-tune itself toward a near-critical, low-energy, rich yet stable regime using multi-evidence signals.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data and Generation](#data-and-generation)
- [Model and Training](#model-and-training)
- [Emergence and Auto-Tuning](#emergence-and-auto-tuning)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Configuration](#configuration)
- [Reproducibility (Cheat Sheet)](#reproducibility-cheat-sheet)
- [Roadmap](#roadmap)
- [FAQ](#faq)

## Overview

- Goal: Align CN/EN text, text images, and semantic images into a shared representation space (the “conscious core”).
- Constraint: Python >= 3.11, standard library only (teaching/verification baseline). Implementation can later be replaced by a high-performance backend without changing public interfaces.

## Features

- Tri-view alignment: CN/EN text, OCR-like text images, and semantic images get trained together toward a shared core.
- Zero dependencies: end-to-end with Python stdlib (math, random, itertools, statistics, json, tomllib, pathlib, logging, multiprocessing, …).
- Bitmap fonts + PPM/PGM: generate text images and simple semantic images by pure Python drawing.
- Conscious core: prototype library, WTA inhibition, simple recurrent placeholder.
- Evaluation: JSONL logs, ASCII plots, retrieval/cluster metrics without external libs.
- Auto-tuning: rate homeostasis, prototype grow/merge/reset, curriculum on loss weights.

## Project Structure

```text
self-snn-conscious-core/
├─ README.md
├─ README_EN.md
├─ LICENSE
├─ config/
│  ├─ default.toml          # training config
│  └─ emergence.toml        # free-play/auto-tuning (criticality/EI/modulation)
├─ data/
│  ├─ concepts/             # concepts + CN/EN phrases (.jsonl)
│  ├─ raw/                  # generated raw images
│  └─ processed/            # train/val/test splits
├─ fonts/
│  ├─ ascii_5x7.json        # ASCII bitmap font
│  └─ han_subset_12x12.json # small Chinese bitmap subset
├─ scripts/
│  ├─ gen_text_images.py     # text → OCR-like images
│  ├─ gen_semantic_images.py  # geometric semantic images
│  ├─ build_dataset.py       # assemble tri-view alignment
│  ├─ train.py               # training (contrastive/agree/center/sparsity/rate)
│  ├─ eval.py                # retrieval/cluster/energy/VP distance
│  ├─ inspect_core.py        # prototype cards (cross-modal Top-K)
│  ├─ run_freeplay.py        # spontaneous activity (Free-Play/Sleep)
│  ├─ analyze_freeplay.py    # ASCII curves for criticality/energy/diversity
│  ├─ crit_test.py           # power-law MLE + KS
│  ├─ eig_monitor.py         # spectral radius approx
│  └─ eibalance_probe.py     # E/I balance and firing distribution probes
└─ self_core/
   └─ ...
```

Note: the above is a planned baseline; modules will be filled in step by step while keeping public interfaces stable.

## Quick Start

```sh
# 0) Python 3.11+
python --version

# 1) Clone (no third-party deps)
git clone <your-repo-url> self-snn-conscious-core
cd self-snn-conscious-core

# 2) Generate a tiny dataset (20–50 concepts)
python scripts/gen_text_images.py --concepts examples/concepts_small.jsonl --out data/raw/text
python scripts/gen_semantic_images.py --concepts examples/concepts_small.jsonl --out data/raw/semantic

# 3) Build tri-view aligned dataset
python scripts/build_dataset.py --in data/raw --out data/processed --val_ratio 0.1 --test_ratio 0.1

# 4) Train (stdlib baseline)
python scripts/train.py --config config/default.toml

# 5) Evaluate
python scripts/eval.py --run runs/<timestamp>

# 6) Prototype cards (cross-modal Top-K)
python scripts/inspect_core.py --run runs/<timestamp> --topk 8
```

## Data and Generation

- Text images: render words/phrases using bitmap fonts (ASCII + small Chinese subset), optional affine/noise/background; write PGM/PPM without Pillow.
- Semantic images: geometric compositions as non-text visual anchors.
- Splits: tri-view `triples.jsonl` under `data/processed/{train,val,test}`.

## Model and Training

- Encoding: text bitmaps and images → spikes (TTFS / rate; threshold/Poisson).
- SNN units: LIF/ALIF in pure Python; optional WTA inhibition and recurrent placeholder.
- Core: prototype bank (K × D), sparse hits, distance by cosine/Euclidean.
- Losses: multi-view contrastive (zh↔img_sem, en↔img_sem, zh↔en), agree/center, optional CTC for text images.
- Optimization: SGD + momentum with surrogate gradients; JSONL logs; ASCII progress.

## Emergence and Auto-Tuning

- Free-Play/Sleep: spontaneous activity, anchored replay, and small-step auto-tuning.
- E/I balance: iSTDP + synaptic scaling + intrinsic plasticity across fast/mid/slow loops.
- Criticality evidence: branching ratio, dynamic range/susceptibility, power-law MLE+KS, spectral radius proxy.
- Safety and rollback: snapshot + revert on degraded criticality/energy/task proxies.

## Evaluation and Visualization

- Retrieval: R@1/R@5, mAP for ZH→Img, EN→Img, and ZH↔EN.
- Clustering: Silhouette, Davies–Bouldin, Fisher ratio.
- SNN: average firing rate, Victor–Purpura distance, total spikes/time steps.
- ASCII plots: progress bars, curves, heatmaps — all zero-deps.

## Configuration

- `config/default.toml` for training; `config/emergence.toml` for free-play/auto-tuning.
- Uses `tomllib` (stdlib) to read TOML; can fallback to JSON with the same name.

## Reproducibility (Cheat Sheet)

```sh
# timestamp for run grouping
TS=$(date "+%Y-%m-%d_%H-%M-%S")

# data build
python scripts/gen_text_images.py --concepts examples/concepts_small.jsonl --out data/raw/text
python scripts/gen_semantic_images.py --concepts examples/concepts_small.jsonl --out data/raw/semantic
python scripts/build_dataset.py --in data/raw --out data/processed --val_ratio 0.1 --test_ratio 0.1

# train + evaluate
python scripts/train.py --config config/default.toml --run runs/$TS
python scripts/eval.py  --run runs/$TS

# interpretability
python scripts/inspect_core.py --run runs/$TS --topk 8

# free-play + analysis
python scripts/run_freeplay.py     --config config/emergence.toml --run runs/$TS
python scripts/analyze_freeplay.py runs/$TS

# criticality probes
python scripts/crit_test.py        --run runs/$TS
python scripts/eig_monitor.py      --run runs/$TS
python scripts/eibalance_probe.py  --run runs/$TS

# unit tests
python -m unittest discover -s tests -p "test_*.py" -v
```

## Roadmap

- M0: minimal runnable stdlib baseline; tri-view alignment; basic reports and prototype cards.
- M1: attributes & compositional binding (entity + attribute separation).
- M2: short phrases → short sentences; simple temporal relations.
- M3: performance: swap internals with a high-performance backend while keeping interfaces stable.

## FAQ

- Is pure Python too slow?
  - This is a verification/teaching baseline. Start small; swap the internals later while keeping public interfaces.
- How do you render Chinese without third-party fonts?
  - Use bitmap JSON fonts (extendable) to stay dependency-free.
- Why multiple criticality signals, not just a power-law or branching ratio?
  - Single indicators can be misleading (subsampling/threshold effects). We combine branching ratio, dynamic range, power-law MLE+KS, and spectral radius.

