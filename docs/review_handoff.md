# 对局复盘工具：交接文档

[使用说明](review.md) · [测试流程](review_test_plan.md) · [中文首页](../README.zh-CN.md)

这份文档写给接手这项工作的人或编码助手，不需要其他背景。它说明：做了什么、验证到哪一步、哪些地方是猜的、接下来先做什么。

## 一句话现状

复盘工具的代码已经写完并推送，在分支 `claude/eager-dijkstra-qq2bmk` 上，比 `main` 多 8 个提交，没有开 PR。**它还从来没有在真实重演的对局上运行过，也没有成功调用过一次真实的 Claude API**：开发是在一台没有游戏引擎、也用不了 API 密钥的云端机器上做的，所有验证都基于手写的观测数据和本地桩服务器。接手后的第一件事是在有引擎的机器上把它跑起来。

## 拿到代码并确认环境

```sh
git fetch origin claude/eager-dijkstra-qq2bmk
git checkout claude/eager-dijkstra-qq2bmk
```

`requirements.txt` 新增了 `anthropic==2.9.0`，已有的虚拟环境需要重新安装一次依赖。然后确认两件事（Windows 下把 `python` 换成 `.\.venv\Scripts\python.exe`）：

```sh
python tools/check_install.py
python -m pytest native_runner/tests/test_review.py native_runner/tests/test_review_commentary.py -q
```

预期是 `Ready: 5 models` 和 `29 passed`。没有安装 `node` 时会跳过 1 个测试。

日常怎么用见[使用说明](review.md)，完整的分阶段验证见[测试流程](review_test_plan.md)。

## 做了什么

思路是复用仓库已有的模仿学习流程：那条流程本来就是“把真人回放在引擎里重演，再让模型给真人的每一步打分”，复盘只是把打分结果展示出来而不是拿去训练。所以引擎只在已有的 `cache_builder` 那一步需要，新写的代码全部是纯 Python。

```text
公开数据集 (Parquet)
  └─ review select ──────────── 挑出想看的对局，写成小数据集          不需要引擎
       └─ training.v4.cache_builder ── 在引擎里重演，产出模型输入缓存   需要引擎（仓库原有）
            └─ review analyze ──────── 模型逐帧打分，写 review.json 和报告页面
                 ├─ review summary ─── 把一个目录的复盘汇成一张表
                 └─ review explain ─── 请 Claude 写文字讲解，写回 JSON 和页面   需要 API 密钥
```

| 文件 | 职责 |
| --- | --- |
| `native_runner/review/scoring.py` | 把一名玩家的缓存序列按真人的实际动作喂给模型，逐帧得到局势值、出牌意愿、对各张牌和各个落点的概率，并从张量里解出圣水、塔血、卡牌、坐标 |
| `native_runner/review/report.py` | 把逐帧结果汇总成复盘文档：出牌列表、错过的时机、局势转折、漏水估计、保真度判定、终端摘要、多局汇总表 |
| `native_runner/review/sources.py` | 从缓存索引里读对局；从数据集读原始回放，得到阵营、胜负、皇冠和全部真实动作；挑选对局 |
| `native_runner/review/page.py`、`page_template.html` | 把复盘文档嵌进一个自包含的 HTML 页面，图表全部用原生 JS 和 SVG 画，不依赖外部库 |
| `native_runner/review/commentary.py` | 把复盘提炼成摘要，调用 Claude 得到结构化的讲解 |
| `native_runner/review/__main__.py` | 命令行：`select`、`list`、`analyze`、`page`、`explain`、`summary` |
| `tools/make_demo_review.py` | 生成一份数据是编造的示例复盘，用来在没有引擎时看页面、试 `explain` |
| `native_runner/tests/test_review.py`（21 个）、`test_review_commentary.py`（8 个） | 测试 |
| `native_runner/tests/review_helpers.py` | 测试夹具：`synthetic_owner_sequence` 用手写观测走真实的张量化和动作对齐；`scripted_review` 直接编造一份完整的复盘文档 |

除此之外只改了四处原有文件：`requirements.txt` 加了 `anthropic`，`.gitignore` 加了 `/datasets/`，两个 README 的文档表格各加一行。

## 验证到哪一步

已经验证的：

- 29 个测试通过。全量测试里有 1 个失败与本工具无关，在这个分支之前就存在：`test_probe_core_json.py::test_core_native_json_preserves_nullable_fields_precision_and_bounds`。
- 打分结果与仓库自己的 `evaluate_recurrent_sequence` 逐帧一致；把多帧合并编码、以及各帧张量大小不同的情况下结果也一致。
- 真实的 General 模型能在手写观测产生的缓存上跑通整条 `analyze`。
- 在公开数据集的 400 场真实回放上，阵营、胜负、皇冠数、动作数的映射全部正确；`select` 写出的数据集能被 `cache_builder` 自己的读取器接受。
- 报告页面在无头浏览器里按桌面和手机宽度、明暗两种主题检查过，没有脚本报错。
- 文字讲解用真实的 SDK 对着本机的桩服务器跑通，核对了实际发出的请求和流式回答的解析。
- [测试流程](review_test_plan.md)的阶段 0 到 3 从 GitHub 重新克隆后原样执行过，数字都对得上。

没有验证的，按重要程度排列：

1. **`analyze` 在真实缓存上能不能跑通。** 手写的观测里场上几乎没有单位，所以单位、效果、关系边这些张量只在一个填充测试里碰到过。
2. **模型的评价在真实对局上有没有参考价值。** 局势值、出牌意愿、选牌概率是否和直觉相符，完全未知。
3. **真实的 Claude API 是否接受这个请求**，尤其是 beta 头、`fallbacks: "default"` 和结构化输出的组合；以及讲解写得好不好。
4. **在 Windows 上运行。** 开发和测试都在 Linux 上。

## 设计上不显然的地方

**打分方式。** 循环状态始终沿着玩家的真实动作前进，所以每一帧评价的都是玩家实际面对的局面。同一帧上还会问模型“如果必须出牌会出什么、放哪里”，用来和玩家的选择对比。

**直接用了模型的私有方法。** `scoring.py` 调用了 `model._decode`、`_decoder_context`、`_location_logits`、`_selected_candidate_memory`、`_resolve_candidate_indices` 和模块级的 `_distribution`，因为公开接口不返回选牌和落点的完整分布。上游改动模型内部时这里要跟着改；`test_review.py` 里有和公开接口比对的测试会报警。

**坐标和时间。** 所有坐标都是被复盘玩家自己的视角：x 从 0 到 17 是从左到右，y 从 0 到 31 是从己方国王塔到对方国王塔。缓存里不存每帧的时间，代码假设第 i 帧的 tick 是 `90 + 5 × i`，这来自仓库的 `FIRST_POLICY_DECISION_TICK` 和 `POLICY_DECISION_TICKS`。

**保真度判定。** 回放只记录稀疏动作，重演不一定等于真实对局。四种判定：`faithful`（走完全程且胜负、皇冠一致）、`unverified`（走完全程但模拟没有在原对局结束时分出胜负）、`partial`（提前中止）、`diverged`（结果不同）。注意“走完全程”不等于每个动作都成功：个别动作可能被引擎拒绝，它们会出现在“未能评分的出牌”里。

**名字。** 技能候选在张量里的卡牌编号是 −1，代码改用词表编号查出所属卡牌。觉醒和精英形态写在名字前面，技能在名字后面标注。

**这些门槛都是估的，没有用真实数据调过**，全部在 `report.py` 顶部：

| 常量 | 现值 | 含义 |
| --- | --- | --- |
| `ACT_THRESHOLD` | 0.5 | 出牌意愿超过它算“模型想出牌” |
| `MISSED_MINIMUM_FRAMES` | 4 | 模型想出牌而玩家在等，至少持续 1 秒才算一段 |
| `SWING_WINDOW_FRAMES` | 20 | 局势转折的窗口，5 秒 |
| `SWING_MINIMUM_CHANGE` | 0.05 | 窗口内局势值变化小于它不算转折 |
| `PLACEMENT_MINIMUM_DISTANCE` | 3.0 | 落点与模型相距多少格算落点分歧 |
| `ELIXIR_SECONDS_PER_POINT` | 2.8 | 估算漏水时用的圣水回复速度 |

其中 `ACT_THRESHOLD` 最可疑：模仿学习训练门控时默认把“出牌”的权重放大 8 倍（`imitation.py` 的 `DEFAULT_IL_GATE_ACT_WEIGHT`），所以出牌意愿整体偏高，0.5 这条线很可能要调。

**文字讲解。** 模型是 `claude-opus-5-5`，流式，思考力度默认 `high`，开启了服务端回退（安全分类器误拒时自动换推荐的模型重试）。输出格式用 `anthropic.transform_schema` 生成 JSON Schema 传进 `output_config.format`，拿到完整回答后先检查停止原因、再自己解析。没有用 SDK 的 `output_format` 自动解析，是因为回答被截断时它会在能看到停止原因之前就抛出校验异常。发给模型的是提炼过的摘要而不是完整时间线，提示词明确告诉它数据里没有单位交战过程和对手的手牌圣水，要求把原因分析说成推测。

## 第一次在真实对局上跑，先看这几样

按[测试流程](review_test_plan.md)的阶段 5 执行，`summary` 命令会给出下面大部分数字。

1. **`analyze` 有没有报错。** 最可能出问题的是 `scoring.py` 里解码候选和动作的部分（`_FrameDecoder.describe`、`_played_actions`）。
2. **“未能评分的出牌”占比。** 预期是百分之几。如果很高，说明出牌时间没有对齐，去查上面那个 `90 + 5 × i` 的假设，以及 `report._unscored_actions` 的匹配逻辑。
3. **胜方的终局局势值是否更高。** 这是局势值有没有意义的最粗的检验。如果不成立，先换模型试试（`--checkpoint`，比如 2.6 速猪卡组用速猪专精模型），再考虑别的办法。
4. **漏水估计和数据集自带的统计差多少。** 差得多说明重演的圣水节奏和真实对局不同，或者估算方式有问题。
5. **“模型想出牌而你在等”是不是多得离谱。** 是的话调 `ACT_THRESHOLD`。
6. **速度。** 在场上没有单位的手写数据上是每帧约 13 毫秒，真实对局会更慢。

## 还没做的

按建议的先后顺序：

1. 上一节的真实数据验证，以及据此调门槛。
2. 用真实密钥跑通 `explain`，再拿真实对局的讲解来调提示词。
3. 把局势值校准成胜率。公开数据集里每场都有真实胜负，可以拟合一条校准曲线。
4. 分支推演：回到某一刻，让模型接管打下去，看结果差多少。引擎支持快照（`cr_native_env.py` 里的 `create_snapshot`、`restore`、`release_snapshot`），回放播放器的后退功能（`training/replay_viewer.py` 的 `_NativeReplayHistory`）是现成的用法，让模型出牌可以参考 `training/v4/offline_agent.py`。这部分我只看过接口，没有读过细节。
5. 复盘自己的对局。仓库只有读取端，采集器没有开源；读取端认的格式是 `royaleapi-battle-replay.v1`，个人数据集的目录约定在 `royaleapi_replay.py` 里（`datasets/royaleapi_player_*`）。数据从哪来还没有答案。

## 需要仓库所有者决定的事

- 这个分支是合并进 `main` 还是继续单独维护。
- 自己的对局数据从哪来（见上面第 5 条）。
- 文字讲解默认给两名玩家各写一份，每份一次请求。只关心自己这一方的话可以把默认改成只写一方。
- 服务端回退是我默认开的，不想要可以去掉。

## 接手时容易踩的坑

- 新写的复盘文件都按仓库的 `ruff` 配置格式化过，但原有的测试文件大多没有。**不要对整个 `native_runner/tests` 跑 `ruff format`**，它会把 80 多个原有文件重排一遍；只格式化自己改的文件。
- 测试里用的回放 ID 必须是皇室战争标签用的字符（`0289PYLQGRJCUV`），否则 `prepare_collected_replay` 会拒绝。
- 缓存分片的摘要文件名是 `<分片名>.cril.json`，不是 `<分片名>.json`。
- 没有任何凭据时，`anthropic` SDK 抛的是 `TypeError` 而不是它自己的异常类型，`__main__.py` 里按消息内容识别。
- 如果机器上设置了 `ANTHROPIC_BASE_URL`，SDK 会把请求发到那里。
- 在限制外网的环境里装依赖和下数据，需要放行这些域名：PyPI、`download.pytorch.org`、`download-r2.pytorch.org`（PyTorch 的安装包实际在这里）、`huggingface.co`、`us.aws.cdn.hf.co`。
- 页面里所有来自数据的文字都用 `textContent` 写入，嵌入的 JSON 把 `<` 转义了。改页面时保持这两点，卡牌名和报错信息才不会被当成 HTML。

## 云端留下的东西

本地开发用不到，可以不管，也可以清理：

- claude.ai 上有一个名为 FirstLight_CR 的个人云端环境，带安装依赖的启动脚本和上面那几个域名的白名单。没有保存任何密钥。
- 两个验证环境用的会话：“Verify FirstLight_CR environment setup”（失败，原因就是当时没放行 `download-r2.pytorch.org`）和同名带 “(retry)” 的（通过）。
- Default 环境没有被改动。
