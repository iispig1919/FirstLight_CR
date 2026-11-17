# 对局复盘

[中文首页](../README.zh-CN.md) · [训练指南](training.md) · [系统架构](architecture.md)

离线分析器把一场回放在引擎里重演，再让模型逐帧评价：局势如何、此刻该不该出牌、出哪张、放哪里。每场对局得到一个可以直接用浏览器打开的报告页面和一份 JSON。它复用模仿学习的重演缓存，所以**引擎只在重演那一步需要**，挑选对局和复盘本身都是纯 Python。

## 流程

| 步骤 | 命令 | 需要引擎 |
| --- | --- | --- |
| 1. 挑对局 | `python -m native_runner.review select` | 否 |
| 2. 重演 | `python -m native_runner.training.v4.cache_builder` | 是 |
| 3. 复盘 | `python -m native_runner.review analyze` | 否 |

以下命令都在仓库根目录执行，每条都是单行，Windows 和 Linux 通用。Windows 下把 `python` 换成 `.\.venv\Scripts\python.exe`。

### 1. 准备数据并挑对局

[IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) 每个分片有 5,000 场，下载一个就够练手。在浏览器里下载 [`replays/part-000000.parquet`](https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/main/replays/part-000000.parquet)，放到 `datasets/il_replay/replays/` 下；Linux 上也可以用命令：

```sh
mkdir -p datasets/il_replay/replays
curl -L -o datasets/il_replay/replays/part-000000.parquet https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/main/replays/part-000000.parquet
```

把想看的对局复制成一个小数据集。`--deck-card` 要求某一方卡组同时包含这些牌（忽略 `-ev1` 这类形态后缀），`--tag` 指定回放 ID，`--limit` 限制数量：

```sh
python -m native_runner.review select datasets/il_replay runs/review/dataset --deck-card hog-rider --deck-card cannon --deck-card ice-golem --limit 8
```

缓存生成器在启动引擎前就会拒绝的对局不会被选中，例如没有动作、同一决策窗口超过两个动作、含不支持的卡牌。输出目录已有回放时命令会拒绝覆盖。

### 2. 在引擎里重演

先按[安装指南](getting_started.md)装好离线引擎并启动（Windows 下是 `./native_runner/start_offline.ps1`），`--ports` 填它的控制端口：

```sh
python -m native_runner.training.v4.cache_builder runs/review/dataset runs/review/cache --ports 26789 --source-count 0 --resident-slots 1
```

`--source-count 0` 表示处理数据集里的全部对局。

### 3. 复盘

```sh
python -m native_runner.review list runs/review/cache/index.json
```

```sh
python -m native_runner.review analyze runs/review/cache/index.json --dataset-root runs/review/dataset --output runs/review/out
```

`list` 显示每场的重演保真度。`analyze` 默认复盘缓存里的全部对局，用 `--tag` 可以只看指定的几场；每场在终端打印一份摘要，并写出 `<回放 ID>.review.html`（报告页面）和 `<回放 ID>.review.json`（同样的数据）。`--dataset-root` 可选，提供后会补上双方阵营、胜负、皇冠数，并列出没能评分的出牌。`--checkpoint` 换模型，默认是 General。

## 报告页面

`review.html` 是单个自包含文件，不联网、不依赖其他文件，双击即可打开，跟随系统的明暗主题。页面顶部切换玩家，其余内容都是该玩家的视角：

| 区域 | 内容 |
| --- | --- |
| 保真度横幅 | 这场重演可不可信，以及原因 |
| 指标卡 | 已评分出牌数、选牌与时机的一致率、落点中位偏差、漏水、平均圣水 |
| 时间线 | 共用时间轴的四张图：局势值、模型出牌意愿、圣水、双方塔血合计。局势值图上的圆点是你的出牌（橙色表示模型首选的不是这张牌），底部的短线是对方的出牌 |
| 出牌评价 | 每次出牌一行，可以只看选牌、落点或时机有分歧的；选中一行后右侧显示场地上你的落点、模型的落点分布，以及模型此刻对手里每张牌的偏好 |
| 模型想出牌而你在等、局势转折 | 点击一行会在时间线上框出那一段 |
| 未能评分的出牌 | 真实对局里有、但重演无法对齐的动作 |

在时间线上悬停会同时显示那一刻的全部数值；聚焦时间线后可以用左右方向键逐秒移动。时间线下方的“查看数据表”给出同样的数字，每秒一行。

页面模板改动后，可以不重新打分、直接从已有的 JSON 重画：

```sh
python -m native_runner.review page runs/review/out/<回放 ID>.review.json
```

## 重演保真度

回放只记录稀疏的出牌动作，重演结果不一定等于真实对局。每份复盘都带一个判定：

| 判定 | 含义 |
| --- | --- |
| `faithful` | 重演走完全程，胜负和皇冠数与真实对局一致 |
| `unverified` | 重演走完全程，但模拟没有在原对局结束的时刻分出胜负，结果无法比对 |
| `partial` | 重演提前中止，`first_untrusted_tick` 之前的帧可用 |
| `diverged` | 模拟结果与真实对局不同，说明中途某处已经不是原来那场对局，结论仅供参考 |

## 输出内容

`review.json` 是报告页面背后的全部数据，`players` 按引擎座位（0、1）各有一份：

| 字段 | 内容 |
| --- | --- |
| `summary` | 局势值范围、平均圣水与漏水估计、与模型的一致率、终局塔血 |
| `actions` | 每次出牌：时间、卡牌、落点、当时圣水，以及模型对时机、选牌、落点的概率，模型自己的首选，落点热力最高的 12 格 |
| `missed_windows` | 你在等、而模型持续 1 秒以上想出牌的时段，附模型的首选 |
| `swings` | 5 秒窗口内局势值变化最大的几段，附期间双方的出牌 |
| `unscored_source_actions` | 真实对局里有、但重演无法对齐评分的出牌 |
| `timeline` | 每 0.25 秒一帧的局势值、出牌意愿、圣水、双方塔血 |

约定：

- **坐标** `[x, y]` 是被复盘玩家自己的视角：`x` 0–17 从左到右，`y` 0–31 从己方国王塔到对方国王塔。塔的左右也按这个视角。
- **时间** 是开局后的经过时间，`tick` 每秒 20 个，模型每 5 个 tick 决策一次。
- **局势值** `value` 是模型价值头的输出：对局结果的期望折扣回报（胜 +1、负 −1）加上塔血塑形，双方零和。它不是校准过的胜率，不同模型的尺度和视野也不同，只在同一个模型内比较。
- **概率** 是模型自身的分布。`act_probability` 来自训练时对“出牌”加权过的门控，只适合当相对分数看。

## 如何解读

模型给出的是它的看法，不是标准答案。概率低表示“这个模型很少这么打”，不表示一定打错了；General 在空场时偏被动，所以开局阶段它常常不认同主动出牌。比较可靠的用法是把它当线索：先在“局势转折”里找到局势值掉得最多的时段，再看那几秒里模型不认同的选牌和落点，以及对方出了什么，自己判断是不是失误。

漏水估计按满圣水的帧数乘以当时的回复速度计算；提供 `--dataset-root` 时摘要会同时显示来源统计的漏水量，两者差距大通常说明重演的圣水节奏和真实对局不同。
