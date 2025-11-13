# self-snn-conscious-core（精简版：中文-英文对齐）

本仓库现聚焦“中文/英文文本对齐”单一任务：
- 使用系统字体（Pillow）将中文与英文短语渲染为清晰的灰度位图；
- 将位图展平为向量，经各自投影矩阵映射到共享语义空间；
- 以 InfoNCE（zh↔en）+ 一致性 + 中心损失训练；
- 评测提供 zh→en 检索 R@K/mAP。

说明：此前与图片/语义图标、SNN/涌现/临界性相关的模块已移除或不再使用，以降低复杂度、提升可复现性。

## 快速开始

```bash
python -m pip install -r requirements.txt
python scripts/build_dataset.py --out data/processed --concepts examples/concepts_small.jsonl --pairs_per_concept 3
TS=$(date +"%Y-%m-%d_%H-%M-%S"); echo $TS > .last_run
python scripts/train.py --config config/default.toml --run runs/$TS
python scripts/eval.py --run runs/$TS --config config/default.toml
```

产物：
- `runs/<ts>/logs.jsonl`：每 N 步记录 loss 分量与关键指标；
- `runs/<ts>/ckpt.pkl`：投影矩阵 `W_zh`、`W_en`；
- `runs/<ts>/eval.json`：`retrieval_zh2en`（R@1、R@5、mAP）。

## 项目结构

```text
self-snn-conscious-core/
├─ README.md
├─ LICENSE
├─ requirements.txt          # 仅 Pillow（系统字体渲染）
├─ config/
│  └─ default.toml           # 训练/评测最小配置
├─ examples/
│  └─ concepts_small.jsonl   # 小样本概念清单
├─ data/
│  └─ processed/             # build_dataset.py 生成（train/val/test）
├─ runs/                     # 训练日志与 checkpoint（自动生成）
├─ scripts/
│  ├─ build_dataset.py       # 从 concepts_small 生成 {id, zh, en} 对
│  ├─ train.py               # 训练（InfoNCE zh↔en + 一致性 + 中心）
│  └─ eval.py                # 评测（zh→en 检索 R@K/mAP）
└─ self_core/
   ├─ encoding/system_font.py
   ├─ losses/contrastive.py
   ├─ eval/metrics.py
   └─ utils/{io.py,ascii_plot.py,timer.py}
```

## 配置说明（最小）

`config/default.toml`

```toml
[seed]
value = 42

[data]
train = "data/processed/train/triples.jsonl"
val   = "data/processed/val/triples.jsonl"
test  = "data/processed/test/triples.jsonl"

[core]
dim = 48

[loss]
align_weight = 1.0
agree_weight = 0.2
center_weight = 0.1
temperature   = 0.07

[train]
epochs = 1
steps_per_epoch = 80
batch_size = 8
lr = 0.02
momentum = 0.9
log_every = 10
eval_every = 40
checkpoint_every = 80
```

## 运行与评测

1) 依赖安装：`pip install -r requirements.txt`（仅 Pillow）
2) 构建数据：`python scripts/build_dataset.py --out data/processed --concepts examples/concepts_small.jsonl --pairs_per_concept 3`
3) 训练：`TS=$(date +"%Y-%m-%d_%H-%M-%S"); echo $TS > .last_run; python scripts/train.py --config config/default.toml --run runs/$TS`
4) 评测：`python scripts/eval.py --run runs/$TS --config config/default.toml`

## 设计要点（简述）

- 系统字体渲染：自动查找常见中文字体（如 PingFang、SimSun、Noto CJK）；英文同样使用系统字体，避免“问号”占位。
- 轻量嵌入：将灰度位图展平并归一化，通过线性投影得到共享表示；
- 对比学习：InfoNCE（带温度）+ 一致性 + 中心损失；
- 纯 Python 训练：SGD+动量，无第三方数值库，分钟级跑通小样本；
- 日志：统一 JSONL；异常会写入 error 字段。

## FAQ

- 英文/中文不清晰或“?”：改用系统字体渲染（Pillow），中文与英文均清晰。
- 想更快试跑？减小 `steps_per_epoch`、`epochs` 或 `pairs_per_concept`。
- 仅保留最小目录？保留 scripts 与 self_core 的最小子集即可复现。

## 示例提交日志（Conventional Commits）

```
feat(pipeline): 精简为中英文本对齐，移除图片与SNN
feat(render): 使用系统字体渲染中文/英文，提升清晰度
feat(scripts): build/train/eval 最小闭环与JSONL日志
docs: 简化 README，聚焦快速开始与配置
chore: .gitignore 忽略 runs/ 与 data/
```

