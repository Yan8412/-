# 西甲比赛预测

用进球模型估计西班牙足球甲级联赛（La Liga，Primera División）每场比赛的：

1. **全场胜平负**概率（主胜 / 平 / 客胜）
2. **半场胜平负**概率
3. **最可能的两个全场比分**（前三仍会给出）及其概率
4. **最可能的两个半全场**（九种：1/1、1/X、1/2、X/1、X/X、X/2、2/1、2/X、2/2）

有上半场比分时，这些概率来自**同一张**（半场比分，全场比分）联合分布，所以半场、全场、比分和半全场不会互相矛盾。半场样本不够时，仍退回到按历史半场进球比例缩放全场期望进球，这时半场和全场是分开的。

三条轨道用同一套输出：

- **纯足球预测**：上下半场 Dixon–Coles，可选地按历史 xG 调整速率
- **市场预测**：用赛前 1X2、大小球和亚洲让球反推进球，再走同一套卷积
- **混合**：两侧半场速率的几何平均，以及全场胜平负的对数池。权重只在更早的走步评测日上拟合

这是概率估计，不是投注建议。

## 模型在说什么

每支球队有两个数：

- **进攻**：越高，越容易进球
- **防守**：越高，越不容易让对手进球（参数是“防守强度”，计算时期望进球会减去它）

一场比赛的期望进球大约是：

- 主队进球 ≈ exp(联赛基线 + 主场优势 + 主队进攻 − 客队防守)
- 客队进球 ≈ exp(联赛基线 + 客队进攻 − 主队防守)

进球个数用泊松分布描述，再按 Dixon–Coles（1997）的办法，修正 0-0、1-0、0-1、1-1 这四个低比分的概率（实际比赛里 0-0 和 1-1 比“两个独立泊松”稍微更常见）。

**联合半全场（默认）**。全场进球 = 上半场进球 + 下半场进球。上半场和下半场各拟合一套 Dixon–Coles 速率。下半场的进球用全场比分减去半场比分，残差为负的行丢掉。可选的状态系数 η：领先一方的下半场期望进球乘 exp(−η)，落后一方乘 exp(+η)，半场打平则不变。η 在训练窗口里用网格搜索，并加一个 N(0, 0.2²) 先验，所以短样本会停在 0。η = 0 就是两个半场独立。把两张半场比分表卷积，得到一张联合分布，再从它读出：

- 上半场边际 = 半场胜平负和半场比分（和单独的上半场表一致）
- 全场边际 = 全场胜平负和全场比分。默认看概率最高的两个，前三仍保留
- 九种半全场

比分表的全场上限至少是两侧半场上限之和，避免把 6+6 压进更小的格子，因此半场进球不会超过全场。全场 Dixon–Coles 仍然保存，用来打印球队攻防、给伤停调整速率。页面上的胜平负和比分来自卷积，不来自那张独立的全场表。

半场样本少于门槛时不拟合下半场。半场期望进球 = 全场 × 训练集里的半场比例（没有半场比分时是 0.45），`ht_source` 记为 `scaled_full_time`，半场和全场分开。

负二项只改预测分布的方差（离散度 1 就是泊松），不改攻防参数的极大似然。这是 R 包 goalmodel 里负二项边际的想法，没有把那个包引进来。温度校准把每张半场表按 1/T 重新归一化后再卷积，联合分布仍然自洽。xG 权重用开球之前的历史 xG 做对数攻防，几何混合进半场速率。本场 xG 不进预测。

近期比赛权重更高。权重是 `exp(−ξ × 距训练截止日的天数)`。默认 `ξ = 0.0025`，半衰期大约 277 天，所以最近两三个赛季影响大，更早的赛季逐渐变轻。

进攻和防守还有一个以 0 为中心的正态先验，加在总对数似然上，比赛越多影响越弱，用来稳住小样本。低比分修正 ρ 同样有一个靠近 −0.05 的弱先验，避免只有几十场比赛时把它顶到边界上。

**升班马和样本很少的球队**：没有历史的球队不会被当成豪门，也不会被当成随机数。程序先看训练集里“第一个赛季之后才出现、并且已经踢满约 15 场”的球队，用他们的平均进攻/防守当作升班马先验；估不出来时先验就是联赛平均（参数为 0）。样本少于 `--min-matches`（默认 10 场）的球队，会按“已赛场次 : 先验强度（默认相当于 12 场）”往这个先验收缩。一场都没踢过的新队，直接使用该先验。

`train` 会把每支球队的进攻、防守打印出来，方便核对模型有没有把强队和弱队分开。有下半场模型时，终端还会打印下半场攻防和 η。

## 市场预测和混合

市场轨道不拟合球队强弱。它用这一场自己的赛前盘，反解两支球队的全场期望进球，以及上半场进球占比，使 Dixon–Coles 比分表贴近去水位之后的价格。用到的盘口是全场胜平负、进球大小（市场 80）、亚洲让球（市场 6）。有上半场大小（28）或上半场让球（26）时，占比也可以从盘口里解出来，否则用 0.45。没有大小球也没有让球时，两队期望进球之和会被拉向联赛均值 2.55，因为单靠胜平负分不清总进球高低。

去水位沿用仓库里已有的比例法、幂法和 Shin。二项以上的盘用同一套 Shin 不动点；两项盘（大小、让球）不用 Shin。没有加入 `shin` 这个包。

半全场（29）、正确比分（57）、半场胜平负（31）、半场正确比分（30）、进球单双（12）只用来对照和校准，不参与反推进球。正确比分先按每家博彩自己报出的比分去掉水位，再把概率平均并重新归一，避免把各家不同的比分集合拼成一张大表。报价存在 `data/processed/odds_quotes.json`，`fetch-markets` 每一轮请求结束时写一次，不按场重写整份文件。

开盘只在**同一个选项**真有两个时间戳时才有。接口通常一行一个选项，只有 `latest_bookmaker_update`，没有价格历史。这时开盘不可用，收盘就是这一口。不同选项的时间戳不一样，不算开盘/收盘。同一选项已经有未停盘的报价时，更早的 `stopped` 行丢掉，不算第二口。只有停盘行时才保留。`Tie` 和 `Draw`、队名和 `1` / `2` 先归一成同一选项，再做这个判断。

`latest_bookmaker_update` 早于开球的报价保留。不早于开球的一律丢掉，包括 `stopped` 为真、以及 `winning` 为 false 的行。实盘里每一条滚球报价都带 `winning`，多数还是 `false`，不能用这个字段把结算碰过的赛前盘和滚球分开。没有时间戳的仍当作赛前盘保留。显式的滚球标记（`inplay` / `is_live`，或盘口说明里写了 in-play）一律丢掉，哪怕时间戳还在开球前。某个盘筛完没有报价，这个盘就空着，不沿用上一轮留下的数字。预测用的是这样筛过之后的最后一口。

亚洲让球有两种写法，按每家博彩自动判断，不写死公司编号。一种是主客各自带自己的盘口（主 −1、客 +1），客队盘口要取反才和主队配成一对。另一种是两边都写主队盘口（两边都是 −0.5），不能再把客队取反。判断标准是能配成的盘口更多，并且主队打穿概率随盘口变容易而上升。配完之后如果汇总结果仍不单调，就丢掉两边里家数更少的那一条。四分盘写成 `-1,-1.5` 或 `-1/-1.5` 时取平均（−1.25）。大小球若 `total` 为空、`name` 是 `3.5`，盘口从 `name` 读。半全场优先读 `name`（`Home/Away`、`Draw/Home`）。`name` 为空时，把 `label` 里用 ` - ` 分开的队名对上这场的主客队：`Draw` / `Tie` 是 X，`Athletic Club` 和 `Athletic Bilbao` 这类对照表里的别名算同一队。正确比分若 `label` 是 `2`、`name` 是 `2-1`，`name` 是胜方视角，客胜要翻成主客比分 1-2；`Tie` 和 `Draw` 都是平局，比分必须打平。

混合有两层，都只在更早的评测日上按全场对数损失选权重：

- 参数层：四个半场速率的几何平均。权重是市场一侧的权重
- 概率层：原有的全场胜平负对数池。权重是足球一侧的权重

## 按准确率升级

每次 `predict` 的终端都会打印纯足球、市场和混合三条，包括 `--no-archive`。市场和混合也打印前两比分和半全场前两档。比分网格与 `compare-tracks` 选权重时相同：上半场最多 5 球，全场最多 8 球，温度 1。注册表里的混合市场权重（例如 0.65）会写在混合那一行上；权重为 0 时会写明混合与纯足球相同。这些行不进 `latest_next.json`。存档仍把纯足球追加到 `data/archive/predictions.jsonl`，并带上模型版本、联合分布参数、前两比分和半全场。有赛前盘且没有 `--no-archive` 时，市场和混合各再追加一行，`extra.track` 分别是 `market` 和 `blend`。历史命中率只看纯足球那一行，市场行不会把足球行换掉。

`upgrade` 把最近一段完场比赛留作样本外（按 UTC 日切）。optuna 只在更早的比赛上走步，搜索 ξ、L2、ρ 先验、η 缩放、xG 权重、负二项离散度和下半场速率缩放。温度和混合权重也只在这段调参窗口上冻结。样本外只用调参截止时拟合的一个模型计分，不能回头改参数。

注册表在 `data/models/registry.json`。还没有冠军时，第一次成功的搜索成为冠军，样本外数字只作记录。已经有冠军、且两边都没有完整的混合分数时，挑战者的全场对数损失至少要低 0.005，并且 RPS 不能变差超过 0.002，才会替换。两边都在全部样本外场次上有混合分数时，改看混合全场对数损失（同样至少低 0.005）；纯足球对数损失变差不超过 0.01，RPS 变差不超过 0.002，半全场对数损失（两边都有时）变差不超过 0.01。样本外场次少于 `--min-promote` 时不更换已有冠军。`upgrade --rollback` 换回上一版。中文记录写在 `data/predictions/accuracy_history.md`。

每日 `auto_update.ps1` 不传 `--ignore-champion`，所以下一轮训练和预测会用上新冠军。每周用 `auto_upgrade.ps1`（Windows 任务计划程序），或在任何系统上运行 `python -m laliga upgrade`。

## 用了哪些库

| 库 | 决定 |
| --- | --- |
| optuna | 使用。升级命令的贝叶斯搜索。没装时退回三组固定参数 |
| shin | 不加入。去水位已经是纯 Python，和 1X2 的 Shin 同一不动点 |
| penaltyblog | 不加入。它的 Dixon–Coles 和本仓库是同一似然，比分分布已有梯度测试。Pi rating 不产出联合比分 |
| soccerdata | 不加入。football-data.co.uk 的 SP1 用标准库直接下载，队名对照已经在仓库里 |
| lightgbm | 不加入默认依赖，也不进冠军。可选的泊松目标半场模型会多一套重依赖；上下半场速率仍由 Dixon–Coles 给出 |
| xgboost | 保留原有对照，不进联合分布的冠军 |

R 包 goalmodel 和 implied 只移植想法：负二项预测边际，以及多种去水位。不运行 R。

## 数据从哪来

[SportMonks Football API v3](https://api.sportmonks.com/v3/football)。西甲的联赛 ID 是 **564**。

程序会：

1. `GET /leagues/564?include=seasons;currentSeason`，按 `starting_at` 取最近若干个赛季（默认 3 个，含当前赛季）。很多订阅（包括用来核对过这个程序的那一份）只能列出 2024/2025 及以后的三个赛季；接口列出更少时按实际数量抓取，不会因为填写的数字更大而失败
2. `GET /fixtures?filters=fixtureSeasons:{赛季ID}&include=participants;scores;state;round;season`，用游标把该赛季拉完。真实接口的 `next_cursor` 是一条完整 URL，程序只取出里面的 `cursor` 值，并且翻页时不再带 `per_page`（两个一起送会 HTTP 400；把整段 URL 当作 cursor 也会 400）。游标页往往只有 `has_more` 和 `next_cursor`，`has_more` 为 false 就停止。没有游标时跟随 `next_page`，再退回页码
3. 预测某一段日期时，用 `GET /fixtures/between/{开始}/{结束}?filters=fixtureLeagues:564`。这个接口单次最长 100 天，更长的区间会自动拆开

赛前赔率和 xG 不在上面这条默认请求里，所以每天的 `fetch` 和操作台「更新数据」不会变慢。需要它们时单独回填（见下文）。

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
| 更新数据 | 向 SportMonks 拉取最近若干个赛季（默认 3；接口列出更少就抓那么多），写入本地缓存 |
| 训练模型 | 用本地全部完场比赛拟合，并保存模型 |
| 运行回测 | 按日期走步，和历史频率基准比较。默认最少训练场次是 320；比赛不够时改小这个数字再运行 |
| 预测下一轮 | 预测最近一轮未开赛比赛 |
| 按日期预测 | 预测一段 UTC 日期内的未开赛比赛（含首尾两天） |

点下去之后会进入任务页，日志会往下长。成功时出现「查看数据状态 / 查看回测 / 查看预测」。失败时红字写出原因，例如没设 `SPORTMONKS_API_TOKEN`、HTTP 401、订阅不含西甲（HTTP 403，免费计划不含联赛 564）、HTTP 429 限流。不会静默失败，也不会在失败时填上假比赛。

五个站内页面：

- **操作台**：上面这些按钮，以及当前缓存了多少场比赛
- **预测结果**：开球时间同时给出 UTC 和新加坡时间（Asia/Singapore，UTC+8）、主客队、全场胜平负、半场胜平负、概率最高的三个比分，以及半全场前两档。有赛前盘时，同一张表下面会多一行市场预测；注册表里的混合权重大于 0 时再多一行混合。点球队名进入该场的完整比分概率表
- **历史记录**：开球前存档的预测和实际比分对照，累计命中率，和「永远猜主胜」「历史频率基准」比较；另有单独标注的「回测重建」
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
python -m laliga fetch --seasons 3
```

再次运行时，已经踢完的赛季走缓存；联赛赛季清单和当前（未结束）赛季每次都重新请求，所以每天的 `fetch` 能拿到新赛果。新抓到的比赛按 `fixture_id` 合并进 `data/processed/matches.csv`。

```bash
python -m laliga fetch --seasons 3 --refresh    # 忽略原始响应缓存，重新请求
python -m laliga fetch --seasons 3 --replace    # 用本次结果覆盖本地比赛表
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

### XGBoost 对照

Dixon–Coles 仍是 `train`、`predict`、操作台和每日更新使用的模型。梯度提升树只作为对照，放在同一次走步里和 Dixon–Coles、历史频率基准比。它不进默认的 `backtest`，所以日常更新不会因为多训一套树而变慢。

对照命令只读本地已经下载的 `matches.csv`，不访问 SportMonks：

```bash
python -m laliga compare --min-train 320 --output data/predictions/model_comparison.json
```

终端会打印全场对数损失、Brier、命中率、场次、校准，以及「每场对数损失之差」的均值和 95% bootstrap 区间，并把同一份结果写入上面的 JSON。没有本地完场比赛时命令会失败并说明原因，不会编造数字。操作台的「回测结果」页在这个文件存在时显示这张表；文件不存在时只提示上面的命令，不显示占位数字。示例模式（`--demo`）不显示这份对照。

树是固定的浅设置：深度 3、学习率 0.05、最多 200 棵、种子 7。每个训练窗口内部按时间留出最后一段做早停，评测日不参与早停，也不做针对回测期的网格搜索。

不用赔率时的特征（全部来自开球前）：

- 近 5 场积分、进球、失球；主队的主场近况和客队的客场近况；本赛季至今的积分和进失球；已赛场次；休息天数（最多按 30 天计）
- 近 5 场半场进球、失球
- 若比赛表里有 `home_xg` / `away_xg`：更早比赛的 xG。本场 xG 是赛后数据，不会当成这场的特征
- 截至该日 00:00 UTC 的 Dixon–Coles 攻防、期望进球和胜平负概率（全场和半场）

同一 UTC 日的比赛互相看不见。这和 Dixon–Coles 走步一致：评测某一天时，当天早场的结果也不进训练。

赔率是可选的。列可以是十进制赔率 `odds_home`、`odds_draw`、`odds_away`，或已经是概率的 `implied_home`、`implied_draw`、`implied_away`。有可用赔率时，命令会多训练一个「含赔率」模型，并把隐含概率归一化（去掉水位），同时加一行「赛前赔率（去水位）」基准：只用这场开赛前的市场隐含概率，不看球队、不看 xG。没有这些列，或最早一个评测窗口里还没有完整赔率时，含赔率的树不运行，输出里会写明原因。赔率基准只统计评测集里有完整赔率的比赛，场次可以比其他模型少。含赔率的结果回答的是「盘口之外还剩多少信息」，不要和不用赔率的对数损失直接当成同一个问题。有完整赛前赔率、并且评测期里样本外比赛已经够热身时，同一份输出还会给出 Dixon–Coles 和去水位赔率的混合，见下一节。

### 回填赛前赔率和 xG

默认 `fetch` 仍只请求 `participants;scores;state;round;season`。本地 `matches.csv` 有比赛之后，用下面这条命令补赛前全场胜平负和每队 xG。它不进每日更新，也不进操作台「更新数据」：

```bash
python -m laliga fetch-markets
python -m laliga compare --min-train 320 --output data/predictions/model_comparison.json
```

第一条按已经存好的 `fixture_id` 请求 SportMonks v3，原始 JSON 同样进 `data/cache/`：

- xG：`GET /fixtures/multi/{ids}?include=xGFixture`，每批最多 50 场。只用 type 5304（Expected Goals）。`home_xg` / `away_xg` 是各方期望进球，`home_xga` 是客队 xG，`away_xga` 是主队 xG。没有 xG 的格子留空。未开赛的比赛不请求 xG，因为这是赛后数据。
- 赔率：`GET /fixtures/multi/{ids}?include=odds&filters=markets:1`。market id 1 是全场胜平负（Match Winner）。不请求 `inplayOdds`。每家博彩保留赛前最后一条报价。时间戳用 `latest_bookmaker_update`（没有则用 `updated_at`、`created_at`）。早于开球的保留；没有时间戳的当作赛前盘保留。不早于开球的一律丢掉，包括 `stopped` 和 `winning: false`，避免滚球价。三边都齐的博彩才计入。三列 `implied_*` 是这些 `1/小数赔率` 的平均，再除掉水位（三项和为 1）。`odds_*` 是这些概率的倒数，所以是去水位之后的价格，不是某一家的原始报价。`raw_implied_*` 保留除掉水位之前的那组平均，Shin 和幂去水位要用它。已经有 `odds_*` 但没有 `raw_implied_*` 的完场比赛，下次 `fetch-markets` 会再请求一次。

订阅没有 Odds & Predictions 或 Pressure Index & xG 时，对应请求返回 HTTP 403，命令会写明缺哪个附加包并停掉这一路，另一路照常写回，不会抛出未处理的异常。某一场没有数据就留空。已经填好的完场比赛下次会跳过；上次确认没有数据的完场比赛也会跳过。未开赛比赛的赔率每次都会重拉，因为收盘价还在变。`--refresh` 忽略缓存并重新请求全部。新抓下来的比分不会把已经回填的赔率或 xG 盖成空值。

1X2 请求成功之后，同一条命令会再要一次扩展盘：亚洲让球 6、进球单双 12、上半场让球 26、上半场进球 28、半全场 29、半场正确比分 30、半场胜平负 31、正确比分 57、进球大小 80。扩展盘这一轮会在日志里写进度（`回填扩展盘`）。扩展盘 403 只记一句话，不把已经写入的 1X2 清掉，也不会把 `parser_version` 标成当前版本，下次还会再试。解析结果在 `data/processed/odds_quotes.json`。1X2 这一轮和扩展盘这一轮各写一次文件，不在每场比赛后重写。缺字段的盘留空，不会编造。同一选项没有两个时间戳时，文件里的开盘为空，探测会写明开盘不可用。`--refresh` 按这场替换整份记录，这次响应里没有的盘不会留下。不带 `--refresh` 时，记录上的 `parser_version` 不是当前解析器（或根本没有这个字段）会用缓存里的原始赔率再解析一次。两轮都成功之后才写入当前 `parser_version`。

想知道当前 token 实际返回了哪些盘、哪些博彩公司、能不能分出开盘和收盘、以及 xG 字段，用探测命令（需要真实 token，本仓库的测试不调用它）。探测只看已完场比赛，未开赛的不计入；时间都标 UTC：

```bash
python -m laliga probe
python -m laliga probe --days 21 --limit 6 --json data/predictions/probe.json
```

输出只列市场编号、博彩公司 id 和 xG 是否出现。token 不会打印。

xG 进 XGBoost 时只作为**更早比赛**的滚动均值（近 5 场 xG 和 xGA）。本场的 `home_xg` / `away_xg` / `home_xga` / `away_xga` 不是这场的特征。本场赛前赔率可以是这场的特征。

若某次赛程响应里已经带了 `xgfixture`（或文档示例里的 `expected` 数组）或赛前 `odds` / `premiumOdds`，解析器会按同样的规则写入比赛表。滚球赔率仍然不读。

`xgboost==3.2.0` 在依赖里。这个版本提供 Windows 的 `py3-none-win_amd64` 轮子，可在 Python 3.13 上安装，不必从源码编译。Windows 上若导入失败，先安装微软的 Visual C++ 可再发行组件。

### Dixon–Coles 与赔率的混合

上面的 `compare` 命令在本地已经有赛前赔率时，会多算全场混合，用来看「Dixon–Coles 加进去之后，能不能低于去水位赔率本身」。没有单独的命令，也不进 `train`、`predict`、操作台按钮或每日更新。半场没有赔率，混合只评全场。

三种混合都是低维的，权重只在 0 和 1 附近的小网格上选：

- 线性：`w × Dixon–Coles + (1 − w) × 赔率`
- 对数：概率正比于 `Dixon–Coles` 的 `w` 次方乘以赔率的 `(1 − w)` 次方，再归一化
- 对数线性叠加：`softmax(a × log(Dixon–Coles) + b × log(赔率))`，`a` 和 `b` 都不小于 0。没有结果类别的截距，所以不能单独把主胜概率抬高。两个模型如果在同一档主胜概率上一起偏低，这三种权重补不上

`w`（以及叠加的 `a`、`b`）按评测日走步拟合。某一天只用更早评测日的样本外 Dixon–Coles 概率和那些场的赛前赔率；当天的结果，包括当天早场，不进入当天的权重。至少先累计 60 场这样的历史才开始计分，所以混合的 `n` 小于全部评测场次。表里会同时给出这一批上的 Dixon–Coles 和赔率（行名带「混合同一批」），配对区间也用这一批。要看的是「线性混合 − 赛前赔率（去水位）」以及对数混合、对数线性叠加各自减赔率的那几行。负值表示混合更好。

终端和 JSON 里的 `blend` 会给出权重路径的起点、中位数和终点。天数为偶数时，中位数是中间两天的算术平均；叠加的 `a` 和 `b` 分开取中位数，不一定是同一天的一对。JSON 和终端里另有事后最优固定权重，标成样本内上界（`in_sample` / `hindsight_upper_bound`），并给出它在计分比赛上的对数损失。它用这些比赛的结果本身来选一个不变的权重，不是走步结果，不能当成混合有没有打败赔率的证据。同时有两者的比赛少于 60 场时，不输出混合行，并写明原因。主胜概率分箱在 JSON 的 `calibration.*.home_probability_bins`，混合和「混合同一批」的 Dixon–Coles、赔率都有。

去水位不只有比例法。`raw_implied_*` 是开赛前多家 `1/小数赔率` 的平均，还带着水位。在此之上再算：

- 比例（multiplicative）：除以三项之和。这一行仍叫「赛前赔率（去水位）」
- 幂：解 `sum(q^k) = 1`。有水位时 `k > 1`，强队的概率会高于比例法
- Shin：纯 Python 固定点，不依赖需要编译的扩展，Windows 上的 Python 3.13 可以直接跑
- 加法，以及赔率比 `p = q / (c + q − c·q)`

每种方法各自一行，带全场对数损失、Brier、RPS、命中率和校准（含主胜概率分箱）。RPS 把主胜、平、客胜按这个顺序当作有序结果：两档累积概率的平方误差之和再除以 2，0 最好，1 最差。配对 bootstrap 同时给出对数损失之差和 RPS 之差，用的是同一次重抽样。每种方法都和比例法比，也和 Dixon–Coles 比。某方法在一部分场次上没有解（加法可能把冷门减成非正数）时，这一行的 `n` 更小。

已经只存了去水位概率、没有 `raw_implied_*` 的完场比赛，这几种方法会和比例法重合，输出里会说明，并且不再单列。补原始隐含概率：

```bash
python -m laliga fetch-markets
```

混合用的去水位方法也按更早评测日的对数损失走步选择，打平时留在比例法。当天和以后的结果不参与选择。事后最优权重只在已经选定的方法上重拟合 `w` 或 `a`、`b`，不会在计分比赛上改选 Shin 或幂。收盘赔率不进入混合。

### 更长的 football-data 历史

2012/13 之后的西甲可以从 football-data.co.uk 的 SP1 CSV 导入。它写到单独的 `data/processed/history.csv`，并缓存原始 CSV。每日 `fetch`、操作台「更新数据」和 `walk_forward` 都不读这个文件。

```bash
python -m laliga import-history --since 2012
python -m laliga compare --min-train 320 --output data/predictions/model_comparison.json
python -m laliga compare --history football-data --since 2012 --min-train 320 --output data/predictions/model_comparison_history.json
```

第一条下载 `https://www.football-data.co.uk/mmz4281/{赛季}/SP1.csv`（例如 `1213` 到 `2627`），需要能访问该站的网络。站点会拒绝一部分没有普通浏览器 User-Agent 的请求。缓存命中时不再下载。`--refresh` 忽略缓存。

导入的是赛果、半场比分，以及 Pinnacle 的赛前盘和收盘盘。PSH/PSD/PSA 是站点在周末前的周五下午、或中场周的周二下午采集的价格：不是开盘第一口，也不是收盘。公平比较用这一组，缺了才用同一时点的 Bet365（B365H/D/A）。PSCH/PSCD/PSCA 是收盘，预测时还没有，所以单独成行并标明「预测时不可用」，不进入混合。市场平均（Avg 或更早赛季的 BbAv）和最高价（Max 或 BbMx）会写入历史表，不单独作为模型行。CSV 的 `Time` 是英国时间（Europe/London）。赛季起始年取自文件名（`1213` 是 2012/13），不从开球月份推断，所以 2020 年 7 月打完的 2019/20 仍算 2019/20。

大小球 2.5 和亚洲让球也读进来，列可以缺。赛前快照优先 Pinnacle（`P>2.5` / `P<2.5`、`PAHH` / `PAHA`、让球线 `AHh`），否则 Bet365，再否则市场平均。收盘列是 `PC>2.5`、`B365C>2.5`、`AHCh` 这一组，只存下来，不拿去预测。`AHh` 是主队让球，可以为 0 或负数。这些列让 `compare-tracks` 在 2024/25 之前也能做市场轨道，不必等 SportMonks 的扩展盘。

2024/25 之后若 SportMonks 和 football-data 是同一天、同一对已对照的球队，比分、半场、fixture id 和球队 id 以 SportMonks 为准。全场比分不一致时记下来，并保留 SportMonks 的比分。队名对照在 `laliga/data/teams.py`，重音和「de」这一类词会先归一化。football-data 也覆盖的赛季里如果还有对不上的 SportMonks 队名，导入直接停止，不会把同一场写成两行。更早、football-data 没有的赛季仍保留原 id。更早赛季没有 xG。长历史对照会在终端和 JSON 里写出评测期每一行公平赛前盘的来源场次（Pinnacle PSH、Bet365、SportMonks，或没有）。Pinnacle 的 PSH 在 2025/26 中途会断，后面的行会落到 Bet365 或 SportMonks。

第二条是原来的 SportMonks 对照，每天重拟合 Dixon–Coles，并训练 XGBoost。第三条读 `history.csv`。不写 `--output` 时，长历史写到 `predictions/model_comparison_history.json`，避免盖掉操作台读的那份 SportMonks 对照。长历史默认每 7 个 UTC 日重拟合一次 Dixon–Coles（`--refit-every 1` 可以改回每天），并且默认不训练 XGBoost（`--xgboost` 打开，`--no-xgboost` 在 SportMonks 对照里关掉）。`--since 2012` 从那个赛季的起始年留到评测里。评测期会跨多个赛季；终端在多于一个赛季时打印分赛季的对数损失和 RPS，并打印主胜概率 0.6–0.8 这一档的校准。`n_folds` 是计分的评测日，`n_refits` 是 Dixon–Coles 的拟合次数。

在这台 Linux 环境里，4000 场、36 支球队的一次全场加半场拟合大约 0.06 秒，大约 25 次迭代就停了，没有用满 `max_iter` 150。长历史默认大约每个有比赛的星期拟合一次，而不是每个评测日都拟合，并且不训树。笔记本上会更慢；这不是在你的电脑上跑完 `compare` 的实测。`--max-iter 40` 可以限制单次拟合的迭代上限，默认仍然是 150。

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

终端表格的概率是百分数，开球时间是 **UTC**（北京时间加 8 小时）。CSV / JSON 里的概率是 0 到 1 的小数，三项之和为 1。JSON 里还有期望进球 `expected_goals_ft` / `expected_goals_ht`。`--json` 的字段（每日脚本写的 `latest_next.json`）不会变，写入时先写临时文件再改名。`--publish` 另外写操作台读的 `predictions/latest.json` 和 `latest.csv`。

### 预测存档和历史记录

每次 `predict`（以及操作台的两个预测按钮）都会给每场比赛追加一行到 `data/archive/predictions.jsonl`：fixture id、开球 UTC、主客队、全场和半场胜平负、前三比分和概率、期望进球、模型版本、生成时间。联合模型还会在附加字段里记下前两比分、半全场、η 和参数。文件只追加，不改写。生成时间不早于开球时间的预测会被拒绝，比赛表里已经不是未开赛状态的比赛也会被拒绝。临时看一眼、不想存档时加 `--no-archive`。终端仍会打印纯足球、市场和混合，`--no-archive` 只是不写存档。`latest_next.json` 的字段集合不变，通知程序不用改。前两比分和半全场写在操作台的 `latest.json` 和终端输出里。

同一场比赛开球前生成过多次时，历史记录用开球前**最后一次**，也就是当时最后能看到的那组数字。赛果不写进存档，每次从 `matches.csv` 读取再对照：

- 全场胜平负命中：概率最高的一项等于全场结果
- 半场胜平负命中：同上，用半场比分；没有半场比分时不计
- 前三比分命中：全场比分在预测的三个比分里

对照的基准：永远猜主胜；历史频率基准（只用这次预测之前的完场比赛，胜平负按频率给概率，前三比分是那时最常见的三个比分）。还给出全场对数损失和 RPS。延期或取消的比赛单列，不计入命中率。

```bash
python -m laliga history                     # 终端汇总和最近 20 场
python -m laliga history --json data/predictions/history.json
python -m laliga backfill-archive            # 可选：回测重建本赛季已赛比赛
```

`backfill-archive` 给存档开始之前已经踢完的本赛季比赛补算：每个比赛日只用当天 00:00 UTC 之前的完场比赛重新拟合，和 `backtest` 同一个切法。结果写到单独的 `data/archive/backtest_reconstruction.jsonl`，标成「回测重建」，单独统计，不并入实时命中率；再运行会覆盖上一次重建。操作台的「历史记录」页显示两部分，并有「回测重建本赛季」按钮和 CSV / JSON 下载。

### 每日更新和新鲜度检查

`auto_update.ps1`（仓库根目录）是笔记本上的每日任务：`fetch` → `check-freshness` → `train` → `predict --next`（写 `latest_next.json`，并存档）→ `history` → `backtest`，每次运行在 `logs/auto_update.log` 追加一行汇总。部署和验证步骤见 [docs/DEPLOY_WINDOWS.md](docs/DEPLOY_WINDOWS.md)。

```bash
python -m laliga check-freshness        # 退出码 0 = OK，2 = STALE
python check_freshness.py               # 同一个检查，给脚本按路径调用
LALIGA_FRESHNESS_NOW=2026-10-20T20:00:00Z python -m laliga check-freshness
python -m laliga doctor                 # 部署后自检
```

STALE 的条件：某场比赛开球超过 48 小时仍是未开赛（或卡在进行中状态）；或者最近一场完场已超过 10 天、并且一场未开赛比赛都没有。最近完场超过 10 天但下一轮已经排出（国家队比赛日）是 OK；上个赛季已经踢完（无未开赛比赛且完场接近 380 场，或赛季清单标记已结束）、新赛程还没公布（休赛期）也是 OK。延期、取消的比赛不算 STALE。脚本遇到 STALE 会用 `--refresh` 再抓一次；仍然 STALE 就退出码 2，不重训，也不覆盖上一次的预测文件。

### 伤停和首发（可选，默认关闭）

SportMonks 文档写明 `lineups`（首发 type 11、替补 type 12）和 `sidelined`（伤病、停赛、存疑）包含在所有基础计划里，Starter 也有，不需要附加包。正式首发大约开球前 1 小时才公布；更早的预测首发需要 Expected Lineups 附加包，这里不用。开发时没有 token，接口覆盖率要在笔记本上用真实 token 确认。

```bash
python -m laliga fetch-availability     # GET /fixtures/multi/{ids}?include=lineups;sidelined.sideline;metadata
python -m laliga compare-availability --min-train 320
python -m laliga predict --next --availability
```

特征只用开球前能知道的信息。球员的「首发占比」是球队在更早日期最近 10 场联赛里他首发的比例。「缺阵主力」是这场伤停名单上球员首发占比之和：伤病和停赛按 1 计，存疑按 0.5 计；开始日期晚于比赛日、或结束日期早于比赛日的记录不算。长期伤员首发占比接近 0，不会再被重复扣分。调整只有两个系数：

- 主队期望进球 × exp(−a × 主队缺阵主力 + d × 客队缺阵主力)，客队对称
- 半场期望进球乘同样的系数，ρ 和比分表不变

`compare-availability` 用和 `backtest` 相同的严格走步：每个评测日的 Dixon–Coles 只用之前的完场比赛，a、d 只用更早评测日的样本外比赛拟合，累计不到 150 场时不调整。输出历史频率基准、Dixon–Coles、加伤停调整、以及加首发缺口调整的全场对数损失、RPS、Brier、命中率、前三比分命中率和场次，配对 bootstrap 区间，写到 `data/predictions/availability_comparison.json`，系数写到 `data/models/availability.json`。首发缺口用的是正式首发，每日脚本运行时通常还没有，这一行只说明开球前 1 小时再跑一次的价值。历史伤停记录是事后读取的，接口没有说明是不是赛前录入，历史评估可能偏乐观。

只有当伤停调整在样本外明显更好时（n ≥ 300，对数损失之差和 RPS 之差的 95% 区间都在 0 以下），`recommend_default` 才是 true；这时再给每日任务加 `-Availability`。否则生产仍然是 Dixon–Coles。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--xi` | 时间衰减。越大越只看近期 |
| `--min-matches` | 少于此场次就向升班马/联赛先验收缩 |
| `--max-iter` | 单次拟合的迭代上限 |
| `--refit` | 预测时忽略已保存的模型，按开球日前的数据重拟合 |
| `--data-dir` | 数据目录 |

## 三条预测对照

足球、市场、混合放在同一次走步里。市场概率只用该场自己的赛前盘。混合权重只用更早评测日的全场对数损失，当天和之后的赛果不参与。

```bash
python -m laliga fetch-markets
python -m laliga compare-tracks --min-train 320 --output data/predictions/tracks_comparison.json
python -m laliga compare-tracks --history football-data --since 2012 --min-train 320 --refit-every 7 \
  --output data/predictions/tracks_comparison_history.json
```

终端和 JSON 都有纯足球、有赔率场次上的纯足球、市场、参数几何混合、概率对数混合，以及「市场（与混合同一批比赛）」和「纯足球（与混合同一批比赛）」。市场场次可以比混合多，因为混合要等足够的历史才开始计分；后两行把市场和足球放到混合已经计分的同一批比赛上。表里还有半全场前两档的命中率。选权重、计分和 `predict` 里的市场/混合用同一网格：上半场最多 5 球，全场最多 8 球，温度 1。混合权重摘要的天数是不同评测日，不是不同权重值的个数。`--refit-every` 默认 0：SportMonks 每天重拟合，football-data 每 7 天。走步时标准错误会打印当前评测日和混合进度。直接盘（半全场、半场胜平负、正确比分、单双）单独报对数损失，而且只统计报过价的场次。操作台「回测结果」页在 `tracks_comparison.json` 存在时显示这张表；文件不存在时只提示上面的命令。

## 升级、回退、探测

```bash
python -m laliga upgrade --trials 12 --holdout 80 --min-promote 40
python -m laliga upgrade --history football-data --since 2012 --trials 12 --holdout 80 --refit-every 14
python -m laliga upgrade --rollback
python -m laliga probe --days 21 --limit 6
python -m laliga predict --next
python -m laliga train --ignore-champion   # 不用注册表冠军，仍用命令行上的 ξ
```

Windows 每周任务是仓库根目录的 `auto_upgrade.ps1`，调用的就是上面的 `upgrade`。它不改 `latest_next.json`。下一次 `auto_update.ps1` 训练或预测时会读到新冠军。

解析器按实盘里看到的字段来写：开球时刻及之后的报价全部丢掉（`winning: false` 不能当成赛前盘）；半全场在 `name` 为空时用 `label` 里的队名；正确比分在 `label` 为 1/2/Draw/Tie 时按胜方视角翻转；亚洲让球按每家博彩判断盘口写法；同一选项已有未停盘报价时忽略更早的 `stopped` 行。这台机器没有 token，所以还不能在全部完场比赛上核对：丢掉开球后的行之后还有多少场会变成没有报价，队名半全场能否覆盖那 510 场只有博彩公司 2 的记录，以及停盘重复行过滤能否清掉那 22 场假开盘。

## 怎么读回测

输出分“全部评测比赛”和每个赛季。每一行都有模型和历史频率基准：

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| 全场 / 半场对数损失 | 实际结果那一项概率的负对数，再对比赛取平均。三项里猜得越准、越敢给高概率，损失越低。完全均匀的 1/3 大约是 1.099 | 越低越好 |
| 全场 / 半场 Brier 分数 | 每场比赛上，三项概率与 0/1 结果的平方误差之和，再取平均。范围是 0 到 2 | 越低越好 |
| 全场 / 半场 RPS | 主胜、平、客胜按这个顺序的累积概率平方误差，再除以 2。范围是 0 到 1。对照表里有全场 RPS | 越低越好 |
| 全场 / 半场最可能结果命中率 | 概率最高的一项是否正好是实际结果 | 越高越好 |
| 实际比分落在前三的比例 | 真实全场比分是否出现在预测的三个比分里 | 越高越好 |

西甲主胜大约四成出头，平局很多，所以全场“最高概率命中率”经常在五成附近，不必和二元分类的准确率比。比分很分散，前三比分能盖住两到四成已经有信息量；要和同一张表里的历史频率基准比，而不是和 100% 比。

模型应当在对数损失和 Brier 上低于基准。如果只是命中率更高、对数损失却更差，说明概率没有校准好，不要只看命中率。半场进球更少、更吵，半场指标有时只和基准打平，甚至略差；全场对数损失和前三比分更说明模型有没有学到东西。

走步表从第一个够样本的评测日算起，前面的比赛没有进入分数。赛季分表能看出新赛季开头（升班马刚上来、衰减后旧赛季权重还在）和赛季中后段的差别。

## 测试

操作台测试需要 Chromium。第一次先安装浏览器（网页测试不过时就是缺这一步）：

```bash
python -m playwright install chromium
```

Linux 上如果还缺系统库，改用 `python -m playwright install --with-deps chromium`。

```bash
pytest -q
```

测试不访问真实的 SportMonks，也不下载 football-data。解析器用文档里的比分结构（含半场、90 分钟、加时里 `CURRENT` 与 `2ND_HALF` 的差别）；HTTP 客户端用假传输检查分页、429、限流和“缓存里不能出现 token”；模型用合成赛程做拟合、梯度核对、未来比赛不能泄漏进训练，以及走步回测对历史频率基准的比较。联合分布的测试检查半场边际、九种半全场加总为 1、前两比分和前三比分的长度，以及 `latest_next.json` 的字段没有变。市场轨道用假报价检查开盘/收盘、大小球和让球，并用已知 λ 反推进球。三条预测的混合权重测试检查当天赛果不进入当天的权重。升级测试检查样本外比赛不进入调参、第一次写入冠军、更差的挑战者不升、以及回退。XGBoost 对照在合成赛程上和同一次 Dixon–Coles 走步对齐，并检查本场比分、本场 xG、同一天早场和未来比赛不会进入赛前特征。混合权重的测试检查当天和未来的结果不会进入当天的权重或去水位方法的选择，并确认事后最优权重标成样本内。队名对照和历史合并用一份写在测试里的 CSV。`demo` / `predict --next` 也在子进程里离线跑通，并检查 `latest_next.json` 的字段没有变。预测存档的测试检查开球后不能写入、开球前多次生成时取最后一次、比分改期提前时的处理、命中和基准只用预测之前的比赛。新鲜度检查用 `LALIGA_FRESHNESS_NOW` 固定时钟。伤停调整用一份测试里生成的阵容和伤停数据，检查系数能找回真实效应、没有效应时接近 0，以及最后一天的结果不影响更早日期的概率。

操作台测试用 Playwright 打开真实页面，点每一个按钮，并跟着每一个站内链接和页脚外链走一遍。没有 token 时断言空状态和明确报错。带 token 的用例把 `SPORTMONKS_API_BASE` 指到本机的一个 HTTP 服务，响应外形与 SportMonks v3 相同（`data` / `pagination` / `rate_limit`，以及 401、403、429），真正的 `requests` 客户端会去请求它。日常使用不要设置 `SPORTMONKS_API_BASE`。首次跑这组测试需要：

```bash
python -m playwright install --with-deps chromium
```

## 目录

```
laliga/            命令行、SportMonks 客户端、联合半全场模型、市场反推进球、升级注册表、回测、操作台
laliga/templates/  操作台页面
laliga/static/     操作台样式
tests/             离线测试，含操作台点击测试和 SportMonks 外形的本地 HTTP 模拟
data/cache/        原始 API 响应，以及 football-data 的 SP1 CSV（git 忽略）
data/processed/    matches.csv、seasons.json、odds_quotes.json，以及 import-history 写出的 history.csv
data/models/       dixon_coles.json、registry.json（冠军参数，可回退）
data/predictions/  预测（latest_next.json、latest.json）、回测、history.json、model_comparison.json、tracks_comparison.json、accuracy_history.md
data/archive/      predictions.jsonl（只追加的开球前预测存档，含市场和混合行）、backtest_reconstruction.jsonl（回测重建）
data/processed/    另有 fetch-availability 写的 lineups.csv、sidelined.csv、availability_fixtures.csv
docs/              DEPLOY_WINDOWS.md：更新 D:\laliga-predictor 的步骤
auto_update.ps1    每日任务（Windows PowerShell）
auto_upgrade.ps1   每周调参（Windows PowerShell）；其他系统用 python -m laliga upgrade
check_freshness.py 新鲜度检查（同 check-freshness）
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
python -m laliga fetch --seasons 3
python -m laliga backtest --output data/predictions/backtest.json
python -m laliga train
python -m laliga predict --next
python -m laliga web
# 或者
python -m laliga predict --from 2026-09-26 --to 2026-10-05 \
  --csv data/predictions/next.csv --json data/predictions/next.json
```

以后每周更新：`python -m laliga fetch`（或预测时加 `--refresh`），然后 `python -m laliga train` 与 `python -m laliga predict --next`。Windows 上每日任务是 `auto_update.ps1`，每周任务是 `auto_upgrade.ps1`。每周更新不要跑 `compare`。想把赔率和 xG 补进本地表、再和 Dixon–Coles、市场、混合比较时：

```bash
python -m laliga fetch-markets
python -m laliga compare --min-train 320 --output data/predictions/model_comparison.json
```

更早的赛季和另一套去水位不进每周更新：

```bash
python -m laliga import-history --since 2012
python -m laliga compare --history football-data --since 2012 --min-train 320 --output data/predictions/model_comparison_history.json
```

开发时没有用真实 token 调过接口。分页、比分字段和联赛 ID 按 SportMonks 公开的 v3 文档实现；订阅若不含历史赛季或 `participants` / `scores` include，`fetch` 会把接口返回的错误（以及“免费计划不含西甲”）直接打出来。
