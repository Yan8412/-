# 西甲比赛预测

用进球模型估计西班牙足球甲级联赛（La Liga，Primera División）每场比赛的：

1. **全场胜平负**概率（主胜 / 平 / 客胜）
2. **半场胜平负**概率
3. **最可能的三个全场比分**及其概率

概率来自同一张比分分布表，而不是三套互相矛盾的分类器。半场用单独的上半场进球模型（半场样本不够时，退回到按历史半场进球比例缩放全场期望进球）。

这是概率估计，不是投注建议。

## 模型在说什么

每支球队有两个数：

- **进攻**：越高，越容易进球
- **防守**：越高，越不容易让对手进球（参数是“防守强度”，计算时期望进球会减去它）

一场比赛的期望进球大约是：

- 主队进球 ≈ exp(联赛基线 + 主场优势 + 主队进攻 − 客队防守)
- 客队进球 ≈ exp(联赛基线 + 客队进攻 − 主队防守)

进球个数用泊松分布描述，再按 Dixon–Coles（1997）的办法，修正 0-0、1-0、0-1、1-1 这四个低比分的概率（实际比赛里 0-0 和 1-1 比“两个独立泊松”稍微更常见）。把 0 到 10 球的所有比分概率算出来并归一化，就得到一张比分表：

- 主队进球多于客队的格子加总 = 主胜概率，平局格子加总 = 平局概率，其余 = 客胜概率
- 概率最高的三格 = 前三比分

半场是另一套同样结构的参数，只用 API 里的上半场比分来拟合。两套模型互相独立：半场概率自己加总为 1，全场概率自己加总为 1，程序不会强行保证“半场比分不超过全场”。

近期比赛权重更高。权重是 `exp(−ξ × 距训练截止日的天数)`。默认 `ξ = 0.0025`，半衰期大约 277 天，所以最近两三个赛季影响大，更早的赛季逐渐变轻。

进攻和防守还有一个以 0 为中心的正态先验，加在总对数似然上，比赛越多影响越弱，用来稳住小样本。低比分修正 ρ 同样有一个靠近 −0.05 的弱先验，避免只有几十场比赛时把它顶到边界上。

**升班马和样本很少的球队**：没有历史的球队不会被当成豪门，也不会被当成随机数。程序先看训练集里“第一个赛季之后才出现、并且已经踢满约 15 场”的球队，用他们的平均进攻/防守当作升班马先验；估不出来时先验就是联赛平均（参数为 0）。样本少于 `--min-matches`（默认 10 场）的球队，会按“已赛场次 : 先验强度（默认相当于 12 场）”往这个先验收缩。一场都没踢过的新队，直接使用该先验。

`train` 会把每支球队的进攻、防守打印出来，方便核对模型有没有把强队和弱队分开。

## 数据从哪来

[SportMonks Football API v3](https://api.sportmonks.com/v3/football)。西甲的联赛 ID 是 **564**。

程序会：

1. `GET /leagues/564?include=seasons;currentSeason`，按 `starting_at` 取最近若干个赛季（默认 6 个，含当前赛季）
2. `GET /fixtures?filters=fixtureSeasons:{赛季ID}&include=participants;scores;state;round;season`，用游标把该赛季拉完。真实接口的 `next_cursor` 是一条完整 URL，程序只取出里面的 `cursor` 值，并且翻页时不再带 `per_page`（两个一起送会 HTTP 400；把整段 URL 当作 cursor 也会 400）。游标页往往只有 `has_more` 和 `next_cursor`，`has_more` 为 false 就停止。没有游标时跟随 `next_page`，再退回页码
3. 预测某一段日期时，用 `GET /fixtures/between/{开始}/{结束}?filters=fixtureLeagues:564`。这个接口单次最长 100 天，更长的区间会自动拆开

主客队来自 `participants[].meta.location`（`home` / `away`）。比分来自 `scores[].description`：

| description | 含义 |
| --- | --- |
| `1ST_HALF` | 半场比分 |
| `2ND_HALF` | 90 分钟累计比分（全场胜平负用这个） |
| `2ND_HALF_ONLY` | 只统计下半场的进球 |
| `CURRENT` | 当前/最终比分，加时赛也会算进去 |

联赛没有加时。只有在状态为全场结束（`FT` / `state_id = 5`）且缺少 `2ND_HALF` 时，才用 `CURRENT` 代替全场比分。推迟、取消、进行中的比赛不进训练集。

**免费计划不含西甲。** 需要一份覆盖 Spanish Primera División（联赛 ID 564）的订阅，并且要能读到你打算用来训练的那些历史赛季。Token 只从环境变量 `SPORTMONKS_API_TOKEN` 读取，通过查询参数 `api_token` 发送，不会写进代码、日志或本地缓存。

原始 JSON 缓存在 `data/cache/`。同一请求再次运行时直接读缓存。响应里的 `rate_limit.remaining` / `rate_limit.resets_in_seconds` 会被遵守；遇到 HTTP 429 会按 `Retry-After` 或重置时间退避。每页算一次请求，`per_page` 最大 50。

## 安装

需要 Python 3.11、3.12、3.13 或 3.14。`numpy`、`pandas`、`scipy` 的版本选的是这四个版本都有预编译包的发布，避免在 3.14 上从源码编译。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，只填 token，不要加引号：

```bash
SPORTMONKS_API_TOKEN=你的token
```

`.env` 已在 `.gitignore` 里。也可以不建 `.env`，直接 `export SPORTMONKS_API_TOKEN=...`。

可选：`LALIGA_DATA_DIR` 或每个命令的 `--data-dir`，用来改数据目录（默认是当前目录下的 `data/`）。

## 操作台

在本机打开网页。进程只监听 `127.0.0.1`，不会对局域网开放：

```bash
python -m laliga web
```

浏览器访问 http://127.0.0.1:8765 。换端口用 `python -m laliga web --port 8766`。页面文字是简体中文。按钮调用的是上面同一套抓取、训练、回测和预测代码。

| 按钮 | 作用 |
| --- | --- |
| 更新数据 | 向 SportMonks 拉取最近若干个赛季（默认 6），写入本地缓存 |
| 训练模型 | 用本地全部完场比赛拟合，并保存模型 |
| 运行回测 | 按日期走步，和历史频率基准比较。默认最少训练场次是 320；比赛不够时改小这个数字再运行 |
| 预测下一轮 | 预测最近一轮未开赛比赛 |
| 按日期预测 | 预测一段 UTC 日期内的未开赛比赛（含首尾两天） |

点下去之后会进入任务页，日志会往下长。成功时出现「查看数据状态 / 查看回测 / 查看预测」。失败时红字写出原因，例如没设 `SPORTMONKS_API_TOKEN`、HTTP 401、订阅不含西甲（HTTP 403，免费计划不含联赛 564）、HTTP 429 限流。不会静默失败，也不会在失败时填上假比赛。

四个站内页面：

- **操作台**：上面这些按钮，以及当前缓存了多少场比赛
- **预测结果**：开球时间同时给出 UTC 和新加坡时间（Asia/Singapore，UTC+8）、主客队、全场胜平负、半场胜平负、概率最高的三个比分。点球队名进入该场的完整比分概率表
- **回测结果**：模型和历史频率基准的对数损失、Brier、命中率、前三比分命中率
- **数据状态**：接口返回的赛季清单、本地比赛表里的赛季和场次、原始响应缓存文件数、上次训练时间

预测和回测生成之后，页面上有 CSV / JSON 下载。文件还没生成时，直接打开下载地址会看到说明，而不是一份空表。

没有 token、也没有本地缓存时，首页是空状态，并写明下一步要做什么。这里不会自动塞入示例比赛。

示例模式默认关闭。只有加上 `--demo` 才会用合成赛程预填，而且每一页顶部都标明「合成示例数据，不是真实西甲」。这个模式拒绝去请求 SportMonks，避免把真实响应写进示例目录。

```bash
python -m laliga web --demo
```

操作台单次拟合最多 80 步。命令行 `train` 默认 150 步，样本很大时命令行可以多走一些迭代。

页脚三个外链是核对过能打开的地址：[SportMonks 文档](https://docs.sportmonks.com/v3)、[比分字段说明](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/includes/scores)、[项目仓库](https://github.com/Yan8412/-)。

确认安装：

```bash
python -m laliga --help
pytest -q
```

没有 token 时可以先跑离线示例（合成的 6 队联赛，不访问网络）：

```bash
python -m laliga demo
```

合成数据写到 `data/demo/`，不会覆盖你之后 `fetch` 下来的真实比赛。

## 命令

下面都假设已经 `source .venv/bin/activate`，并且在仓库根目录。

### 抓取 / 更新数据

```bash
python -m laliga fetch --seasons 6
```

再次运行会走缓存，并把新抓到的比赛按 `fixture_id` 合并进 `data/processed/matches.csv`。

```bash
python -m laliga fetch --seasons 6 --refresh    # 忽略原始响应缓存，重新请求
python -m laliga fetch --seasons 6 --replace    # 用本次结果覆盖本地比赛表
```

### 训练

用本地全部完场比赛拟合，并写入 `data/models/dixon_coles.json`。终端会打印全场和半场的球队进攻/防守。

```bash
python -m laliga train
python -m laliga train --xi 0.0025 --min-matches 10 --max-iter 150
```

有新结果之后重新 `fetch` 再 `train`。`predict` 如果发现模型比最新完场更旧，会自动按赛前数据重拟合，不必先手动 train；train 的作用是留下一份可查看的参数，并让随后的预测直接复用它。

### 回测

按 UTC 日期走步：评测某一天时，只用该日 00:00 UTC **之前**的完场比赛，不用当天早场，也不用未来。前 `--min-train` 场（默认 320，大约大半个赛季）只用于热身，不计入分数。

```bash
python -m laliga backtest
python -m laliga backtest --min-train 320 --output data/predictions/backtest.json
```

基准是训练集里的历史频率：全场/半场胜平负各加 1 次伪计数后的全局比例（每场比赛用同一组概率），前三比分是训练集里最常见的三个比分。基准不用球队信息。

### 预测

下一轮（同一 `season_id` + `round_id` 里最早一场未开赛比赛所在的轮次；没有轮次号时用最早那天）：

```bash
python -m laliga predict --next
```

某一段日期（含首尾）：

```bash
python -m laliga predict --from 2026-09-26 --to 2026-10-05 \
  --csv data/predictions/week.csv \
  --json data/predictions/week.json
```

本地没有这段赛程、并且环境里有 token 时，会自动调用日期区间接口补齐（`--refresh` 则强制重拉）。只预测状态为未开赛的比赛。

终端表格的概率是百分数，开球时间是 **UTC**（北京时间加 8 小时）。CSV / JSON 里的概率是 0 到 1 的小数，三项之和为 1。JSON 里还有期望进球 `expected_goals_ft` / `expected_goals_ht`。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--xi` | 时间衰减。越大越只看近期 |
| `--min-matches` | 少于此场次就向升班马/联赛先验收缩 |
| `--max-iter` | 单次拟合的迭代上限 |
| `--refit` | 预测时忽略已保存的模型，按开球日前的数据重拟合 |
| `--data-dir` | 数据目录 |

## 怎么读回测

输出分“全部评测比赛”和每个赛季。每一行都有模型和历史频率基准：

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| 全场 / 半场对数损失 | 实际结果那一项概率的负对数，再对比赛取平均。三项里猜得越准、越敢给高概率，损失越低。完全均匀的 1/3 大约是 1.099 | 越低越好 |
| 全场 / 半场 Brier 分数 | 每场比赛上，三项概率与 0/1 结果的平方误差之和，再取平均。范围是 0 到 2 | 越低越好 |
| 全场 / 半场最可能结果命中率 | 概率最高的一项是否正好是实际结果 | 越高越好 |
| 实际比分落在前三的比例 | 真实全场比分是否出现在预测的三个比分里 | 越高越好 |

西甲主胜大约四成出头，平局很多，所以全场“最高概率命中率”经常在五成附近，不必和二元分类的准确率比。比分很分散，前三比分能盖住两到四成已经有信息量；要和同一张表里的历史频率基准比，而不是和 100% 比。

模型应当在对数损失和 Brier 上低于基准。如果只是命中率更高、对数损失却更差，说明概率没有校准好，不要只看命中率。半场进球更少、更吵，半场指标有时只和基准打平，甚至略差；全场对数损失和前三比分更说明模型有没有学到东西。

走步表从第一个够样本的评测日算起，前面的比赛没有进入分数。赛季分表能看出新赛季开头（升班马刚上来、衰减后旧赛季权重还在）和赛季中后段的差别。

## 测试

```bash
pytest -q
```

测试不访问真实的 SportMonks。解析器用文档里的比分结构（含半场、90 分钟、加时里 `CURRENT` 与 `2ND_HALF` 的差别）；HTTP 客户端用假传输检查分页、429、限流和“缓存里不能出现 token”；模型用合成赛程做拟合、梯度核对、未来比赛不能泄漏进训练，以及走步回测对历史频率基准的比较。`demo` / `predict --next` 也在子进程里离线跑通。

操作台测试用 Playwright 打开真实页面，点每一个按钮，并跟着每一个站内链接和页脚外链走一遍。没有 token 时断言空状态和明确报错。带 token 的用例把 `SPORTMONKS_API_BASE` 指到本机的一个 HTTP 服务，响应外形与 SportMonks v3 相同（`data` / `pagination` / `rate_limit`，以及 401、403、429），真正的 `requests` 客户端会去请求它。日常使用不要设置 `SPORTMONKS_API_BASE`。首次跑这组测试需要：

```bash
python -m playwright install --with-deps chromium
```

## 目录

```
laliga/            命令行、SportMonks 客户端、Dixon–Coles 模型、回测、操作台
laliga/templates/  操作台页面
laliga/static/     操作台样式
tests/             离线测试，含操作台点击测试和 SportMonks 外形的本地 HTTP 模拟
data/cache/        原始 API 响应（git 忽略）
data/processed/    matches.csv、seasons.json
data/models/       dixon_coles.json
data/demo/         demo 命令的合成数据
data/web-demo/     python -m laliga web --demo 的合成数据
```

## 本地第一次跑真实数据

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 在 .env 中设置 SPORTMONKS_API_TOKEN
python -m laliga fetch --seasons 6
python -m laliga backtest --output data/predictions/backtest.json
python -m laliga train
python -m laliga predict --next
python -m laliga web
# 或者
python -m laliga predict --from 2026-09-26 --to 2026-10-05 \
  --csv data/predictions/next.csv --json data/predictions/next.json
```

以后每周更新：`python -m laliga fetch`（或预测时加 `--refresh`），然后 `python -m laliga train` 与 `python -m laliga predict --next`。

开发时没有用真实 token 调过接口。分页、比分字段和联赛 ID 按 SportMonks 公开的 v3 文档实现；订阅若不含历史赛季或 `participants` / `scores` include，`fetch` 会把接口返回的错误（以及“免费计划不含西甲”）直接打出来。
