# 对局复盘工具：测试流程

[对局复盘说明](review.md) · [中文首页](../README.zh-CN.md)

这份流程是给执行测试的人或编码助手看的，不需要其他背景。被测对象是 `native_runner/review`：挑选对局、用模型给重演的对局打分、生成报告页面、调用 Claude 写文字讲解。

流程分六个阶段，越往后需要的条件越多。阶段 0 到 3 只需要 Python 和网络，阶段 4 需要 Claude API 密钥，阶段 5 需要能运行的离线对战引擎。

## 执行约定

1. 按顺序执行。某个阶段的前提不满足时，把它记为 SKIPPED 并写明缺什么，然后继续后面的阶段。
2. 只测试，不修改仓库里的代码、测试和文档。命令失败时原样记录完整输出；不要为了让它通过而改参数、跳过测试或换模型。
3. 不要自行从网上下载游戏 APK。引擎需要的 APK 只能由仓库所有者提供。
4. 阶段 4 和阶段 5 的讲解会调用真实的 Claude API 并产生费用，整个流程最多发起 3 次真实请求。
5. 不要把密钥写进任何文件、命令输出或报告。
6. 所有命令都在仓库根目录执行，按 Linux 写。Windows 下把 `.venv/bin/python` 换成 `.\.venv\Scripts\python.exe`，把 `.venv/bin/pip` 换成 `.\.venv\Scripts\pip.exe`。
7. 全部做完后，按文末的模板汇报。

## 阶段 0：环境

前提：Python 3.12；能访问 PyPI、`download.pytorch.org` 和 `download-r2.pytorch.org`。

```sh
git fetch origin claude/eager-dijkstra-qq2bmk
git checkout claude/eager-dijkstra-qq2bmk
git log --oneline -1
python3.12 -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt pytest
.venv/bin/python tools/check_install.py
```

通过标准：最后一条命令的末行以 `Ready: 5 models` 开头。

同时记录这些环境信息，后面的阶段和报告要用：

```sh
uname -a
.venv/bin/python --version
ls /dev/kvm
node --version
[ -n "$ANTHROPIC_API_KEY" ] && echo "key: set" || echo "key: not set"
```

## 阶段 1：自动化测试

```sh
.venv/bin/python -m pytest native_runner/tests/test_review.py native_runner/tests/test_review_commentary.py -q
```

通过标准：`29 passed`。只有两种跳过是允许的：没有安装 `node` 时跳过 1 个（检查页面脚本语法）；本机存在 `~/.config/anthropic` 目录时跳过 1 个（检查没有密钥时的提示）。其他任何失败或跳过都要报告。

这些测试不访问外网，也不调用真实的 Claude API：讲解部分是用真实的 SDK 连本机的一个桩服务器。

可选：跑全量测试。

```sh
.venv/bin/python -m pytest native_runner/tests -q
```

已知有一个失败与本工具无关，在这个分支之前就存在，不用处理：`test_probe_core_json.py::test_core_native_json_preserves_nullable_fields_precision_and_bounds`。除它之外不应该有失败。

## 阶段 2：示例复盘和报告页面

示例里的对局和数字是脚本编造的，只用来检查页面和命令能不能工作。

```sh
.venv/bin/python tools/make_demo_review.py --output runs/demo-review
.venv/bin/python -m native_runner.review page runs/demo-review/example.review.json
.venv/bin/python -m native_runner.review summary runs/demo-review
```

通过标准：

- 三条命令都以 0 退出，`runs/demo-review/` 下有 `example.review.json` 和 `example.review.html`。
- `summary` 的表格里有这一行开头：`| 示例数据-不是真实对局 | faithful | 0 | 胜 | 702/702 | 22 | 2 |`，表格下方有 `胜方终局局势值高于负方：1/1`。

如果机器上有无头浏览器（例如 Playwright 加 Chromium），再打开 `example.review.html` 检查：

- 控制台没有报错；
- `#timeline svg` 有 4 个，`#actions-table tbody tr` 有 22 行；
- 点击出牌表格的第一行后，`#detail .title` 里有文字，`#detail svg` 存在；
- 把视口宽度设为 390 像素时，页面没有横向滚动条。

没有浏览器就把这一项记为 SKIPPED。

## 阶段 3：公开数据集和挑选对局

前提：能访问 `huggingface.co` 及其下载域名（下载会重定向到 `*.hf.co`）。

```sh
mkdir -p datasets/il_replay/replays
curl -L -o datasets/il_replay/replays/part-000000.parquet https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/main/replays/part-000000.parquet
.venv/bin/python -m native_runner.review select datasets/il_replay runs/review/dataset --limit 20
.venv/bin/python -m native_runner.review select datasets/il_replay runs/review/hog --deck-card hog-rider --deck-card cannon --deck-card ice-golem --limit 8
```

通过标准：

- 下载的文件大小是 17,432,059 字节。
- 两条 `select` 都以 0 退出，输出是一段 JSON：第一条的 `selected` 有 20 个回放 ID，第二条有 8 个，`skipped_tags` 都是空的。
- `runs/review/dataset/replays/part-000000.parquet` 和 `runs/review/hog/replays/part-000000.parquet` 都存在。

再把第一条 `select` 原样重跑一次。这次应该失败，报错里有 `refusing to add replays to an existing dataset`：它拒绝覆盖已有的数据集，这是预期行为。

## 阶段 4：文字讲解（真实的 Claude API）

前提：阶段 2 通过；环境变量 `ANTHROPIC_API_KEY` 已设置，对应的账号可以使用 `claude-opus-5-5`。

先看将要发送的内容。这一条不调用 API，不花钱：

```sh
.venv/bin/python -m native_runner.review explain runs/demo-review/example.review.json --player 0 --show-request > runs/demo-review/request.json
```

通过标准：`request.json` 是合法的 JSON，其中 `model` 是 `claude-opus-5-5`，`fallbacks` 是 `default`，`betas` 是 `["server-side-fallback-2026-07-01"]`，`output_config.effort` 是 `high`，`output_config.format.type` 是 `json_schema`。

然后发起真实请求，并记下耗时：

```sh
.venv/bin/python -m native_runner.review explain runs/demo-review/example.review.json --player 0
.venv/bin/python -c "import json; c=json.load(open('runs/demo-review/example.review.json',encoding='utf-8'))['players'][0]['commentary']; print(c['model'], c['usage'], len(c['moments']), [m['start_tick'] for m in c['moments']])"
```

通过标准：

- 第一条以 0 退出，输出里有一行 `玩家 0 的讲解（<模型名>，输入 N / 输出 M tokens）`，后面是讲解正文，最后一行是 `已写入 …`。
- 第二条打印出模型名、token 用量、时刻的个数（应为 3 到 6 个）和每个时刻的起始 tick（应为整数；个别为 `None` 可以接受，要在报告里注明）。
- `example.review.html` 已更新：文件里能搜到 `"commentary"`。有浏览器的话，确认页面顶部出现“文字讲解”卡片，点击其中一个带时间的标题后，时间线上出现一个框。

失败时把错误信息完整记下来。如果是 `Claude API 返回错误 400：…`，冒号后面的内容说明 API 拒绝了哪个请求参数，这正是这一阶段要发现的问题，必须原文保留。同一条命令最多重试一次。

另外读一遍讲解正文，在报告里写下观察（这一项不判通过或失败）：

- 提到的时间是否都在对局时长之内（示例对局是 2:59）；
- 提到的卡牌是否都在数据里出现过（示例卡组是骑士、弓箭手、亡灵、万箭齐发、火球、巨人、火枪手、迷你皮卡）；
- 有没有把推测写成确定的事实，有没有编出数据里没有的交战细节；
- 有没有注意到这是虚构的示例数据。

## 阶段 5：真实对局（需要离线引擎）

前提：阶段 3 通过；这台机器上能启动仓库的离线对战引擎，并且知道它的控制端口。

引擎的安装方法在仓库文档里：Windows 加 MuMu 见 [安装指南](getting_started.md)，Linux 见 [训练指南](training.md) 第 6 节（需要 `/dev/kvm` 和 Android 模拟器）。两条路都需要仓库所有者提供的原版 APK。这份流程的作者没有验证过引擎的安装步骤；装不起来就把本阶段记为 SKIPPED，写明卡在哪一步和缺什么。

引擎就绪后，把 `<端口>` 换成它的控制端口：

```sh
.venv/bin/python -m native_runner.training.v4.cache_builder runs/review/dataset runs/review/cache --ports <端口> --source-count 0 --resident-slots 1
.venv/bin/python -m native_runner.review list runs/review/cache/index.json
.venv/bin/python -m native_runner.review analyze runs/review/cache/index.json --dataset-root runs/review/dataset --output runs/review/out
.venv/bin/python -m native_runner.review summary runs/review/out
```

通过标准：

- `cache_builder` 以 0 退出，最后一行是 JSON 且 `"ok": true`。记录其中的 `completed_replays`、`failed_replays`、`winner_mismatches`、`crowns_mismatches` 和 `elapsed_s`。
- `list` 每个回放一行，每行有一个判定（`faithful`、`unverified`、`partial` 或 `diverged`）。
- `analyze` 以 0 退出，没有 Python 报错栈；每个有决策帧的回放在 `runs/review/out/` 下得到一个 `.review.json` 和一个 `.review.html`。记录每局的用时（命令会打印）。
- `summary` 以 0 退出。把它的完整输出贴进报告。

这是第一次在真实重演的对局上运行这个工具，下面几项是这一阶段真正要回答的问题，请逐项给出结论：

1. 有没有哪个回放让 `analyze` 报错。有的话，贴出完整的报错栈和对应的回放 ID。
2. `summary` 末尾“未能评分的出牌”的比例。预期是百分之几；如果超过 10%，作为问题报告，它可能意味着出牌时间没有对齐。
3. `summary` 末尾“胜方终局局势值高于负方”的比例。没有及格线，它反映局势值是否有意义。
4. `summary` 末尾“漏水估计与来源统计的平均差距”。
5. 卡牌名字是否正常。运行下面的命令，它会列出所有出现过的动作名字；如果有纯数字的名字（例如 `-1` 或 `26000021`），作为问题报告。

   ```sh
   .venv/bin/python -c "import json,glob; names=sorted({a['name'] for f in glob.glob('runs/review/out/*.review.json') for p in json.load(open(f,encoding='utf-8'))['players'] for a in p['actions']}); print(len(names), names)"
   ```

6. 有浏览器的话，打开两个报告页面（一个 `faithful`，一个不是），确认控制台没有报错，保真度横幅和 `list` 的判定一致。

如果设置了 `ANTHROPIC_API_KEY` 且还有请求额度（见执行约定第 4 条），挑一场 `faithful` 的对局生成讲解，把讲解全文贴进报告，并按阶段 4 末尾的几条写下观察：

```sh
.venv/bin/python -m native_runner.review explain runs/review/out/<回放 ID>.review.json --player 0
```

最后，把一份 `faithful` 和一份非 `faithful` 的 `.review.json`，连同 `summary` 的输出，交给仓库所有者。`runs/` 目录不在版本控制里；除非他要求，不要把这些文件提交进仓库。

## 汇报模板

```markdown
# 复盘工具测试报告

- 提交：<git log --oneline -1 的输出>
- 环境：<操作系统> / <Python 版本> / /dev/kvm <有或无> / node <版本或无> / ANTHROPIC_API_KEY <已设置或未设置>

| 阶段 | 结果 | 说明 |
| --- | --- | --- |
| 0 环境 | PASS / FAIL / SKIPPED | |
| 1 自动化测试 | | <通过、失败、跳过的个数> |
| 2 示例复盘和页面 | | <浏览器检查做了还是跳过了> |
| 3 数据集和挑选 | | |
| 4 文字讲解 | | <模型名、输入和输出 token、耗时> |
| 5 真实对局 | | <对局数和各判定的个数> |

## 失败和异常

<每一项：阶段、命令、完整输出>

## 阶段 4 的记录

<讲解全文，以及对讲解的观察>

## 阶段 5 的记录

<cache_builder 的汇总数字、list 的输出、summary 的完整输出、每局用时、第 1 到 6 项的结论、讲解全文和观察>

## 其他观察

<流程里没有问到、但值得仓库所有者知道的事>
```
