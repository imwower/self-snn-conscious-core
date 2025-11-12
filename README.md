# self-snn-conscious-core

目标：在“中文/英文文本 ↔（文字图片 + 语义图片）”三视角上，学习一个共享语义空间（意识核）。让“不同输入形式但语义相同”的样本在意识核内形成同一装配体（原型簇），并输出一致。

约束：仅使用标准 Python 库（零第三方依赖）。本仓库提供一个纯 Python 教学/验证级基线，便于后续无缝替换为高性能实现。

## 目录

- [特性](#特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [数据与生成](#数据与生成)
- [模型与训练循环（纯 Python 基线）](#模型与训练循环纯-python-基线)
- [评测与可视化（纯 Python 指标）](#评测与可视化纯-python-指标)
- [自动调参与意识核维护](#自动调参与意识核维护)
- [配置与命令行](#配置与命令行)
- [开发规范与测试](#开发规范与测试)
- [路线图](#路线图)
- [许可证](#许可证)
- [FAQ](#faq)
- [附：最小示例](#附最小示例)

## 特性

- 三视角对齐：中文/英文文本、文字图片（OCR 风格）、语义图片 同步训练，进入共享意识核。
- 纯标准库实现：不依赖 numpy/PIL/tqdm 等；数据、训练、指标全部用 math、random、itertools、dataclasses、statistics、json、tomllib、pathlib、logging、multiprocessing 等完成。
- 简易图像格式：内置 PPM/PGM 写入器与位图字体（ASCII 完整，小规模汉字可扩充），可生成文字图片与简单语义图形（圆/矩形/多边形组合）。
- 意识核（原型 + 竞争抑制 + 复发占位）：提供原型库、WTA（胜者为王）抑制、基础复发通路接口。
- 可复现实验：固定随机种子、JSONL 日志、ASCII 进度条、无外部依赖的评测指标与曲线表格。
- 自动调参器：发放率 homeostasis、原型生长/合并/重置、损失权重课程日程（纯 Python 实现）。

## 项目结构

注：下述为规划中的示例结构，实际模块将逐步补齐。

```text
self-snn-conscious-core/
├─ README.md
├─ LICENSE
├─ config/
│  └─ default.toml                # 主配置（Python 3.11+ 用 tomllib 读取）
├─ data/
│  ├─ concepts/                   # 概念清单与中英词/短语映射（.jsonl）
│  ├─ raw/                        # 原始/合成数据
│  └─ processed/                  # 切分后数据（train/val/test）
├─ fonts/
│  ├─ ascii_5x7.json              # 内置 ASCII 位图字体
│  └─ han_subset_12x12.json       # 小规模常用汉字位图（可自行扩充）
├─ scripts/
│  ├─ gen_text_images.py          # 文字→图片（PPM/PGM）
│  ├─ gen_semantic_images.py      # 语义图片（几何图形组合）
│  ├─ build_dataset.py            # 生成三元组/对齐对（ZH/EN ↔ Img）
│  ├─ train.py                    # 训练循环（纯 Python）
│  ├─ eval.py                     # 评测与报告
│  ├─ inspect_core.py             # 原型可解释性卡片导出
│  └─ auto_tune.py                # 自动调参器（homeostasis/生长合并）
├─ self_core/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ io.py                    # PPM/PGM 写入、JSONL、随机种子
│  │  ├─ ascii_plot.py            # 终端小图/直方图/热图
│  │  └─ timer.py
│  ├─ encoding/
│  │  ├─ text_bits.py             # 文本→位图编码、脉冲编码（TTFS/rate）
│  │  └─ vision_draw.py           # 基础几何绘制（线/圆/多边形）
│  ├─ snn/
│  │  ├─ neuron.py                # LIF/ALIF 纯 Python 单元
│  │  ├─ wta.py                   # 胜者为王抑制
│  │  └─ rnn.py                   # 简单复发占位
│  ├─ core/
│  │  ├─ prototypes.py            # 原型库 + 距离 + 生长/合并/重置
│  │  └─ conscious_core.py        # 意识核前向/更新接口
│  ├─ losses/
│  │  ├─ contrastive.py           # InfoNCE/三元组（纯 Python）
│  │  └─ ctc.py                   # 轻量 CTC（可选，DP 纯 Python）
│  └─ eval/
│     └─ metrics.py               # R@K、Silhouette、Fisher 比、DB 指数、VP 距离
└─ tests/
   └─ test_minimal.py             # unittest：可生成/训练/评测小数据
```

## 快速开始

```sh
# 0) 准备 Python 3.11+
python --version

# 1) 克隆项目（零依赖）
git clone <your-repo-url> self-snn-conscious-core
cd self-snn-conscious-core

# 2) 生成最小示例数据（20~50 个概念）
python scripts/gen_text_images.py --concepts examples/concepts_cn_en_small.jsonl --out data/raw/text
python scripts/gen_semantic_images.py --concepts examples/concepts_cn_en_small.jsonl --out data/raw/semantic

# 3) 构建三视角对齐数据集
python scripts/build_dataset.py --in data/raw --out data/processed --val_ratio 0.1 --test_ratio 0.1

# 4) 训练（纯 Python 小步验证）
python scripts/train.py --config config/default.toml

# 5) 评测与报告
python scripts/eval.py --run runs/2025-11-12_00-00-00

# 6) 查看意识核原型卡片（跨模态 Top-K）
python scripts/inspect_core.py --run runs/2025-11-12_00-00-00 --topk 8
```

提示：纯 Python 速度较慢，默认配置只跑小数据与少量迭代；主要用于验证训练流程与指标。后续可在不改接口的前提下替换为高性能实现。

## 数据与生成

### 4.1 概念清单（JSONL）

每行一个概念与中英映射：

```json
{"id": "apple", "zh": ["苹果"], "en": ["apple"], "attrs": ["red", "green"]}
```

### 4.2 文字图片（OCR 风格）

使用 `fonts/ascii_5x7.json` 与 `fonts/han_subset_12x12.json` 的位图字形，将“中文/英文词或短语”渲染为 PGM/PPM。

支持：随机字号（倍数缩放）、旋转、透视近似（仿射）、噪声、背景纹理（程序生成）。

无第三方库：自行写像素，输出 PGM/PPM（ASCII 或 binary magic number）。

命令示例：

```sh
python scripts/gen_text_images.py --concepts data/concepts/zh_en.jsonl --out data/raw/text --num_per_phrase 300
```

### 4.3 语义图片（锚定）

用几何图形组合近似常见物体（例如“苹果”= 圆 + 小叶片三角形），或用统一风格的符号图标。

目的：提供与文字同义但非文字本身的视觉锚点。

命令示例：

```sh
python scripts/gen_semantic_images.py --concepts data/concepts/zh_en.jsonl --out data/raw/semantic --num_per_concept 10
```

### 4.4 数据切分与三视角对齐

```sh
python scripts/build_dataset.py --in data/raw --out data/processed --val_ratio 0.1 --test_ratio 0.1
```

输出：

```text
data/processed/
  train/ triples.jsonl   # [{"id": "...", "zh": "...", "en": "...", "img_text": "path", "img_sem": "path"}, ...]
  val/   triples.jsonl
  test/  triples.jsonl
```

## 模型与训练循环（纯 Python 基线）

该基线用于验证框架正确性，非高性能实现。

### 5.1 编码与脉冲表示

- 文本编码：字符位图 → 序列特征 → 脉冲时间编码（TTFS）或发放率编码（rate），见 `encoding/text_bits.py`。
- 图像编码：像素 → 脉冲（阈触发/泊松采样），见 `encoding/vision_draw.py`。
- SNN 单元：`snn/neuron.py` 提供 LIF/ALIF 前向（列表 + `math.exp`），无需 numpy。

### 5.2 意识核（conscious_core.py）

- 原型库（`core/prototypes.py`）：K 个原型向量（维度 D），稀疏命中；支持：
  - 余弦/欧氏距离
  - 胜者为王（`snn/wta.py`）
  - 生长/合并/重置（见 §7）
- 复发占位：提供时步循环接口，默认关闭（可逐步启用）。

### 5.3 损失（losses/）

- 多视角对比（`losses/contrastive.py`）：InfoNCE（zh↔img_sem、en↔img_sem、zh↔en）。
- 一致性/中心：中心收敛与 zh↔en 向量一致性（L2/KL）。
- 可选 CTC（`losses/ctc.py`）：针对文字图片的弱序列监督（纯 Python 动态规划），默认关闭以保证速度。

### 5.4 训练循环（`scripts/train.py`）

- 批量读取 `triples.jsonl`，构造三视角小批；
- 前向：编码 → 意识核投影 → 原型命中/抑制；
- 反向：纯 Python 的“替代梯度近似 + 手写参数更新”（SGD/动量）；
- 日志：每 N 步输出 JSONL 指标与 ASCII 进度条；
- Checkpoint：用 `json`/`pickle` 保存权重（仅标准库）。

## 评测与可视化（纯 Python 指标）

- 跨模态检索：R@1/R@5、mAP（`eval/metrics.py`）。
- 聚类/表征质量：Silhouette、Davies–Bouldin、Fisher 比（类间/类内）。
- 一致性：zh↔en 向量 L2/KL，跨增广的原型保持率。
- SNN 指标：层均发放率、Victor–Purpura（VP）距离（时间鲁棒）。
- 能耗/时延：样本总脉冲数、时步。
- ASCII 可视化：`utils/ascii_plot.py` 以字符画方式展示条形图、热图、简单散点格。

```sh
python scripts/eval.py --run runs/2025-11-12_00-00-00
```

## 自动调参与意识核维护

在不依赖外部库的前提下，实现三类“规则引擎式”控制：

### 发放率 Homeostasis（每层）

- 目标发放率 r_star（例如：核层 0.5~2 Hz；低层 2~5 Hz）。
- 周期性比较实际 r 与 r_star：
  - 若 r > r_star + delta：小幅上调阈值 V_th 或抑制增益 g_inh；
  - 若 r < r_star - delta：相反操作。

### 原型库生长/合并/重置

- 生长：类内方差升高且样本数大 → 原型二分（K+1）。
- 合并：两中心距离 < ε 且样本重叠高 → 合并（K-1）。
- 重置：利用率最低的 q% 原型重采样到高密度区域（近邻样本初始化）。

### 课程日程（损失权重/温度）

- 早期加大（zh↔img_sem、en↔img_sem）对齐；
- 当 zh↔en KL 低于阈值后，提升 zh↔en 一致性权重；
- 拥挤度上升时，降低 InfoNCE 温度 τ，或增加原型数 K。

```sh
python scripts/auto_tune.py --run runs/2025-11-12_00-00-00 --config config/default.toml
```

## 配置与命令行

### 8.1 config/default.toml（示例）

```toml
[seed]
value = 42

[data]
train = "data/processed/train/triples.jsonl"
val   = "data/processed/val/triples.jsonl"
test  = "data/processed/test/triples.jsonl"

[image]
width = 96
height = 96

[encoding]
text_encoding = "ttfs"          # 或 "rate"
vision_encoding = "rate"

[core]
dim = 256
prototypes = 512
wta_inhibition = 0.6            # 0~1
recurrent_steps = 1

[loss]
align_weight = 1.0              # zh-img, en-img
agree_weight = 0.2              # zh-en
center_weight = 0.1
sparse_weight = 0.02
rate_weight   = 0.02
temperature   = 0.07

[train]
epochs = 5
batch_size = 16
lr = 0.01
momentum = 0.9
log_every = 50
eval_every = 200

[auto_tune]
target_rate_core = 1.0
delta = 0.2
grow_if_intra_var = 0.35
merge_if_center_dist = 0.10
reset_bottom_percent = 0.05
```

读取：项目使用 `tomllib`（标准库）读取 TOML；如需 JSON，直接把同名 `.json` 放到 `config/`，脚本会优先选 TOML，不存在则回退 JSON。

### 8.2 通用 CLI 约定

- 所有脚本支持 `--config`、`--run`、`--seed`；
- 所有路径使用 `pathlib`，兼容 Windows/Mac/Linux；
- 日志统一输出到 `runs/<timestamp>/logs.jsonl`。

## 开发规范与测试

- 代码风格：不强制引入 flake8 等（保持“零依赖”），建议本地自检。
- 单元测试：使用标准库 `unittest`。

```sh
python -m unittest discover -s tests -p "test_*.py" -v
```

可复现性：统一用 `random.seed`、`secrets`、`hash()` 控制；文件写入包含元信息（配置、种子、Git 提交号）。

## 路线图

- 里程碑 M0（当前）：纯 Python 运行最小数据，三视角对齐，输出评测报告与原型卡片。
- 里程碑 M1（属性与组合）：在文字与语义图片中加入属性（颜色/大小），意识核做到可组合绑定（实体 + 属性分离）。
- 里程碑 M2（短语→短句）：轻量复发/时序（SVO 模式），引入动作关系的简单图形。
- 里程碑 M3（性能化）：在不改公共接口情况下，用高性能实现（如向量库/SNN 框架）替换 `self_core` 内部，评测保持一致。

## 许可证

建议使用 MIT 或 Apache-2.0（可按需添加 LICENSE 文件）。

## FAQ

**Q1. 无第三方库如何生成“中文字形”的图片？**

本仓库采用位图字体 JSON（`fonts/han_subset_12x12.json`）渲染常用汉字；你可以扩充该文件或添加新的位图集。为保持零依赖，不直接加载系统字体；若后续允许依赖，可无缝切至 Pillow。

**Q2. 纯 Python 训练很慢？**

此基线用于验证方法论与度量指标。建议小数据、小批量、少 epoch；接口稳定后，可替换实现获得性能。

**Q3. 不想用 CTC？**

可以先只做“对比 + 一致性 + 中心”三类损失；`losses/ctc.py` 默认不启用。

## 附：最小示例

文字位图 → PGM 写入（节选）

```python
# scripts/snippets/text_to_pgm.py
from pathlib import Path
import json, random

def draw_bitmap_text(chars, font_path, scale=2, pad=2):
    font = json.loads(Path(font_path).read_text(encoding="utf-8"))
    glyphs = [font.get(ch, font.get("?")) for ch in chars]
    h = len(glyphs[0]); w = sum(len(g[0]) for g in glyphs) + pad*(len(glyphs)+1)
    H, W = h*scale + 2*pad, (w*scale) + 2*pad
    img = [[255]*W for _ in range(H)]
    x = pad
    for g in glyphs:
        gh, gw = len(g), len(g[0])
        for i in range(gh):
            for j in range(gw):
                if g[i][j] == 1:
                    for di in range(scale):
                        for dj in range(scale):
                            img[pad+i*scale+di][x+j*scale+dj] = 0
        x += gw*scale + pad
    return img

def save_pgm(img, path):
    H, W = len(img), len(img[0])
    with open(path, "wb") as f:
        f.write(f"P5\n{W} {H}\n255\n".encode("ascii"))
        f.write(bytearray([pix for row in img for pix in row]))

if __name__ == "__main__":
    txt = random.choice(["apple", "苹果"])
    img = draw_bitmap_text(txt, "fonts/ascii_5x7.json" if txt.isascii() else "fonts/han_subset_12x12.json")
    save_pgm(img, "demo.pgm")
```

InfoNCE（纯 Python 版，节选）

```python
# self_core/losses/contrastive.py
import math

def cosine(u, v):
    nu = math.sqrt(sum(x*x for x in u)) + 1e-9
    nv = math.sqrt(sum(x*x for x in v)) + 1e-9
    return sum(x*y for x,y in zip(u,v)) / (nu*nv)

def info_nce(batch_a, batch_b, temperature=0.07):
    # batch_*: List[List[float]]，长度 N
    N = len(batch_a)
    sim = [[cosine(batch_a[i], batch_b[j]) for j in range(N)] for i in range(N)]
    loss = 0.0
    for i in range(N):
        logits = [s/temperature for s in sim[i]]
        m = max(logits)
        exps = [math.exp(z-m) for z in logits]
        denom = sum(exps)
        pos = exps[i] / denom
        loss += -math.log(pos + 1e-9)
    return loss / N
```

