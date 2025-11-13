# self-snn-conscious-core

[English README](README_EN.md)

目标：在“中文/英文文本 ↔ 文字图片（OCR 风格） ↔ 语义图片”三视角上，学习一个共享语义空间（意识核）。让不同输入形式但语义相同的样本在原型空间形成同一装配体（原型簇），并输出一致。

扩展：无外部输入时，意识核可自发产生脉冲（Free-Play/Sleep），并通过多证据闭环自调至最佳工作区（近临界、低能耗、表征丰富且稳定）。

约束：仅使用标准 Python（>= 3.11），零第三方依赖。

## 目录

- [特性一览](#特性一览)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [数据与生成](#数据与生成)
- [模型与训练（纯标准库）](#模型与训练纯标准库)
- [意识涌现与自主调节](#意识涌现与自主调节)
- [评测与可视化](#评测与可视化)
- [自动调参与风控](#自动调参与风控)
- [配置说明](#配置说明)
- [开发规范与测试](#开发规范与测试)
- [路线图](#路线图)
- [常见问题（FAQ）](#常见问题faq)
- [参考与设计依据（精选）](#参考与设计依据精选)
- [一句话概括](#一句话概括)
 - [复现实验（Cheat Sheet）](#复现实验cheat-sheet)

## 特性一览

- 三视角对齐：中文/英文文本、文字图片（OCR 线）、语义图片（锚定线）同步训练，进入共享意识核，保证“同义同输出”。
- 零依赖实现：全流程仅用标准库（math、random、statistics、itertools、json、tomllib、pathlib、logging、multiprocessing）。
- 位图字体 + PPM/PGM：内置 ASCII 与小规模汉字位图，生成纯文本图片；语义图片用几何图形拼装。
- 意识涌现：无输入时自发放电，出现装配体的元稳定游走与“做梦式重放”。
- E/I 自组织稳态：iSTDP（抑制性可塑）+ 突触缩放（Scaling）+ 内在可塑（Intrinsic）三层闭环，长期维持稀疏稳定的异步不规则状态。
- 多证据临界性：分枝比、动态范围/易感度、幂律稳健检验（MLE+KS）、谱半径代理联合判断“最佳工作点”。
- 延迟与序列：复发连接带离散时延（环形缓冲），配合时序 STDP，涌现可复现装配体序列。
- 锚定重放 + 神经调质：以“近期高置信原型”为锚点重放；ACh/NE/DA 三通道门控探索/巩固/奖励。
- 自动调参器：能耗、临界证据与任务代理指标之间的规则化调度；原型生长/合并/重置维护库质量。

## 项目结构

```text
self-snn-conscious-core/
├─ README.md
├─ LICENSE
├─ config/
│  ├─ default.toml          # 训练主配置
│  └─ emergence.toml        # 自发活动/自调节配置（临界性/EI/调质等）
├─ data/
│  ├─ concepts/             # 概念与中英短语(.jsonl)
│  ├─ raw/                  # 生成的原始图片
│  └─ processed/            # 切分后的 train/val/test
├─ fonts/
│  ├─ ascii_5x7.json        # ASCII 位图字体
│  └─ han_subset_12x12.json # 常用汉字位图子集
├─ scripts/
│  ├─ gen_text_images.py     # 文本→文字图片（OCR 线）
│  ├─ gen_semantic_images.py  # 语义图片合成（几何图形）
│  ├─ build_dataset.py       # 组装三视角对齐数据
│  ├─ train.py               # 训练（对比/一致性/中心/稀疏/速率）
│  ├─ eval.py                # 评测（检索/聚类/能耗/VP 距离）
│  ├─ inspect_core.py        # 原型卡片（跨模态 Top-K）
│  ├─ run_freeplay.py        # 自发活动（Free-Play/Sleep）
│  ├─ analyze_freeplay.py    # 临界/能耗/多样性 ASCII 曲线
│  ├─ crit_test.py           # 幂律稳健检验（MLE+KS+对照）
│  ├─ eig_monitor.py         # 谱半径近似（幂迭代）
│  └─ eibalance_probe.py     # E/I 平衡与发放分布探针
└─ self_core/
   ├─ utils/                 # IO/ASCII 绘图/计时/日志
   ├─ encoding/              # 文本位图 & 视觉几何绘制；TTFS/Rate 编码
   ├─ snn/
   │  ├─ neuron.py           # LIF/ALIF（纯 Python）
   │  ├─ wta.py              # 胜者为王（支持多胜者）
   │  ├─ rnn.py              # 复发占位
   │  ├─ delay_line.py       # 时延环形缓冲
   │  └─ plasticity.py       # iSTDP / Scaling / Intrinsic
   ├─ core/
   │  ├─ prototypes.py       # 原型库 + 生长/合并/重置
   │  └─ conscious_core.py   # 意识核（含 free_play 钩子）
   ├─ losses/
   │  ├─ contrastive.py      # InfoNCE/三元组
   │  └─ ctc.py              # 轻量 CTC（可选）
   ├─ eval/
   │  └─ metrics.py          # R@K/Silhouette/DB/Fisher/VP
   └─ emergence/
      ├─ spontaneous.py      # 自发点火与时步仿真
      ├─ homeostasis.py      # 发放率/抑制/温度控制律
      ├─ criticality.py      # 分枝比/动态范围/幂律 MLE+KS/谱半径
      ├─ modulation.py       # ACh/NE/DA 门控接口
      └─ optimizer.py        # SPSA/坐标爬山（可选）
```

## 快速开始

```sh
# 0) Python 3.11+
python --version

# 1) 克隆项目（零依赖）
git clone <your-repo-url> self-snn-conscious-core
cd self-snn-conscious-core

# 2) 生成最小数据（20~50 概念）
python scripts/gen_text_images.py --concepts examples/concepts_small.jsonl --out data/raw/text
python scripts/gen_semantic_images.py --concepts examples/concepts_small.jsonl --out data/raw/semantic

# 3) 构建三视角对齐数据集
python scripts/build_dataset.py --in data/raw --out data/processed --val_ratio 0.1 --test_ratio 0.1

# 4) 训练（零依赖小步验证）
python scripts/train.py --config config/default.toml

# 5) 评测与报告
python scripts/eval.py --run runs/<timestamp>

# 6) 自发活动与自调节（Free-Play/Sleep）
python scripts/run_freeplay.py --config config/emergence.toml
python scripts/analyze_freeplay.py runs/<timestamp>

# 7) 临界性/谱半径/EI 探针（体检）
python scripts/crit_test.py --run runs/<timestamp>
python scripts/eig_monitor.py --run runs/<timestamp>
python scripts/eibalance_probe.py --run runs/<timestamp>

# 8) 意识核原型卡片
python scripts/inspect_core.py --run runs/<timestamp> --topk 8
```

提示：纯 Python 版本用于验证方法与指标，默认小批量与少量迭代；接口稳定后可无缝替换 self_core 内部为高性能实现，不改变脚本与配置。

## 数据与生成

### 4.1 概念清单（JSONL）

每行一个概念与中英短语：

```json
{"id": "apple", "zh": ["苹果"], "en": ["apple"], "attrs": ["red", "green"]}
```

### 4.2 文字图片（OCR 线）

- 根据位图字体渲染词/短语，做随机字号/仿射/噪声/纹理背景，输出 PGM/PPM。
- 训练中可选轻量 CTC（纯 Python 动态规划）。

### 4.3 语义图片（锚定线）

- 用圆/多边形/线段组合简化常见物体，或统一风格符号图标，作为非文字的视觉锚点。

### 4.4 切分与对齐

```text
data/processed/{train,val,test}/triples.jsonl
# 字段：中文短语、英文短语、文字图片路径、语义图片路径、概念 id
```

## 模型与训练（纯标准库）

- 编码：文本（字符位图 → TTFS/Rate）与图像（阈触发/泊松采样）统一进入 SNN 编码器（LIF/ALIF）。
- 意识核：稀疏复发 SNN + 原型库 + 多胜者 WTA；输出 256–512 维表征。
- 多任务损失（按配置开/关）：
  - 多视角对比（InfoNCE）：zh↔img_sem、en↔img_sem、zh↔en；
  - 中心/一致性/稀疏/速率正则；
  - CTC（可选）：用于文字图片的序列约束。
- 优化：SGD/动量（纯 Python），替代梯度近似；日志写 JSONL。

## 意识涌现与自主调节

- Free-Play/Sleep：无输入时自动点火、游走、重放、调参。
- E/I 三层闭环：
  - 快：发放率 homeostasis（阈值/抑制增益微调）；
  - 中：抑制性可塑 iSTDP（Vogels 简化窗）；
  - 慢：兴奋性突触缩放（到目标率）+ 内在可塑（阈值向目标分布）。
- 延迟与序列：复发连接带离散时延（1–20 步）+ 时序 STDP → 可复现装配体序列与元稳定切换。
- 锚定重放：从“近期高置信原型”采样重放，限制与概念中心距离，避免语义漂移。
- 神经调质门控：
  - ACh：编码期↑（抑制复发、增强输入）；巩固期↓。
  - NE：探索强度（温度/增益）与停留时间门控。
  - DA（三因子）：新奇/一致性提升 → 奖励门控 STDP。
- 临界性与最佳工作点（多证据）：
  - 分枝比 ≈ 1；动态范围/易感度达峰；
  - 幂律稳健检验（MLE+KS+多窗一致）；
  - 谱半径 ≈ 1（幂迭代近似）。
- 上述归一后纳入内在效用 J，并与任务代理项（跨模态一致性/R@1 移动平均）加权求和，引导在线调参。

## 评测与可视化

- 跨模态检索：ZH→Img、EN→Img、ZH↔EN 的 R@1/R@5、mAP。
- 表征质量：Silhouette、Davies–Bouldin、Fisher 比。
- 一致性：zh↔en 表征距离/对称 KL，增强鲁棒增扩下的保持率。
- SNN 指标：层均发放率、Victor–Purpura 距离（时间鲁棒）、总脉冲数/时步（能耗）。
- 临界体检：分枝比轨迹、动态范围/易感度曲线、幂律 KS-p、谱半径近似。
- E/I 探针：兴奋/抑制电流比、权重直方图、发放分布 vs 目标分布偏差。
- ASCII 可视化：进度条、曲线、热图均以字符画显示（零依赖）。

## 自动调参与风控

- 参数自动化：温度 T、复发增益 g_rec、抑制增益 g_inh、阈值 V_th 由 `homeostasis.py` 与 `criticality.py` 按效用 J 做小步调节；可选 `SPSA`/坐标爬山对 g_rec/g_inh/T 做低频扰动-回退微调。
- 结构维护：原型生长（类内方差↑且利用率高）、合并（中心近且重叠高）、重置（长期“死原型”）。谱半径/分枝比过大时优先合并避免超临界。
- 风控与回滚：每轮 free-play 产出体检报告；若临界性/能耗/任务代理指标恶化超阈值，自动回滚至上次安全快照。

## 配置说明

### 9.1 config/default.toml（示例）

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
text_encoding   = "ttfs"      # or "rate"
vision_encoding = "rate"

[core]
dim = 256
prototypes = 512
wta_inhibition = 0.6          # 0~1，多胜者策略由内部控制
recurrent_steps = 1

[loss]
align_weight = 1.0            # zh-img, en-img
agree_weight = 0.2            # zh-en
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
```

### 9.2 config/emergence.toml（示例）

```toml
[freeplay]
steps = 5000
log_every = 50

[targets]
rate_core = 1.0               # Hz，核层目标发放率
active_prototypes = 2         # 期望每步激活的原型数

[weights]                      # 内在效用 J 的权重
crit = 4.0
energy = 2.0
diversity = 2.0
instability = 1.0
powerlaw = 1.0                 # 幂律稳健性
shape = 1.0                    # 折叠误差/一致性

[init]
g_rec = 0.85
g_inh = 0.60
temperature = 0.03
V_th = 1.0

[control]                      # 控制律步长
beta_m = 0.05                  # 分枝比 EMA
eta_m  = 0.02                  # 复发增益调节
rate_delta = 0.02
temp_up = 0.005
temp_down = 0.005
inh_up = 0.01
inh_down = 0.01

[plasticity]
istdp_eta = 1e-3
istdp_alpha = 1.1              # 目标率相关
scaling_period = 300
intrinsic_eta = 1e-4

[delay]
min = 1
max = 20
tstdp_eta = 5e-5

[spsa]
enabled = true
alpha = 0.05
delta = 0.02
prob = 0.1
```

## 开发规范与测试

- 代码风格：为保持零依赖，不强制 linters；建议本地自检即可。
- 单元测试：标准库 unittest；覆盖数据生成、训练前向、free-play 与指标计算。

```sh
python -m unittest discover -s tests -p "test_*.py" -v
```

- 可复现性：统一 random.seed 与可序列化配置；运行日志写入 `runs/<timestamp>/logs.jsonl`，体检报告 `freeplay.jsonl`。

## 路线图

- M0（当前）：零依赖最小可跑；三视角对齐 + 自发活动与自调节体检。
- M1（属性/组合）：实体-属性可组合绑定（相位/门控），提升语义组合能力。
- M2（短语→短句）：时序层（Spikformer 接口占位、不引入依赖），动作/关系的简单场景。
- M3（性能化）：在不改公共接口前提下，用高性能实现替换 self_core 内部，保持评测一致性。

## 常见问题（FAQ）

- 为什么“临界性”用多证据而非只看幂律或分枝比？
  - 单一指标易误判（子采样/阈值效应）。综合分枝比 + 动态范围/易感度 + 幂律 MLE+KS + 谱半径更稳健；任务最优点也未必恰在临界点。
- 纯 Python 会不会很慢？
  - 本仓库定位方法验证与指标演示；默认小数据/少迭代。接口稳定后可无缝替换高性能实现。
- 不用第三方字体如何渲染中文？
  - 使用位图字体 JSON 渲染常用汉字；可扩充 `fonts/han_subset_12x12.json`。
- 自发活动会不会“飘离语义”？
  - Free-Play 采用锚定重放 + 神经调质门控；并将任务代理项纳入效用 J 持续纠偏。

## 参考与设计依据（精选）

- 临界/雪崩/动态范围：神经雪崩与临界附近的动态范围/信息传输优势；幂律判据需用 MLE+KS 并与指数/对数正态对照；子采样/阈值造成的假幂律与误判风险。
- E/I 平衡与可塑性：平衡网络的异步不规则放电；抑制性可塑（iSTDP）建立 E/I 平衡；突触缩放与内在可塑维持长期稳定与目标发放分布。
- 元稳定与序列：皮层元稳定/UP-DOWN 序列；引入时延 + STDP 促进可复现的装配体序列（多时标、多样化）。
- 神经调质与三因子学习：ACh（编码/不确定）、NE（自适应增益/探索-利用）、DA（三因子奖励门控）在策略切换和信用分配中的作用。
- “边缘并非普适最优”与结构化输入：边缘-混沌常带来性能峰，但部分任务在更稳一侧最佳；结构化输入/回放有助于网络自组织到近临界。

以上要点均落实为可运行的零依赖模块：
`emergence/criticality.py`（分枝比/动态范围/幂律 MLE+KS/谱半径）、
`snn/plasticity.py`（iSTDP/Scaling/Intrinsic）、
`snn/delay_line.py`（时延缓冲 + 时序 STDP）、
`emergence/modulation.py`（ACh/NE/DA 门控）、
`scripts/run_freeplay.py` + `analyze_freeplay.py`（自发活动与体检）。

## 一句话概括

一个零依赖、可复现、教学与研究取向的 SNN 意识核脚手架：既能把多模态语义对齐，也能在“无人看管”时自己放电、自己做梦、自己调参，并用多证据证明它工作在“既不爆、也不塌”的最佳区间。

## 复现实验（Cheat Sheet）

```sh
# 统一时间戳用于 run 路径
TS=$(date "+%Y-%m-%d_%H-%M-%S")

# 数据构建
python scripts/gen_text_images.py --concepts examples/concepts_small.jsonl --out data/raw/text
python scripts/gen_semantic_images.py --concepts examples/concepts_small.jsonl --out data/raw/semantic
python scripts/build_dataset.py --in data/raw --out data/processed --val_ratio 0.1 --test_ratio 0.1

# 训练 + 评测
python scripts/train.py --config config/default.toml --run runs/$TS
python scripts/eval.py  --run runs/$TS

# 可解释性
python scripts/inspect_core.py --run runs/$TS --topk 8

# 自发活动 + 体检
python scripts/run_freeplay.py     --config config/emergence.toml --run runs/$TS
python scripts/analyze_freeplay.py runs/$TS

# 临界性/谱半径/EI 探针
python scripts/crit_test.py        --run runs/$TS
python scripts/eig_monitor.py      --run runs/$TS
python scripts/eibalance_probe.py  --run runs/$TS

# 单元测试
python -m unittest discover -s tests -p "test_*.py" -v
```
