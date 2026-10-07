# 黄金价格推动力模型 Gold Force

完整的模块化 Python 研究项目：官方网络数据采集 → 原始快照 → 观测台账 → 发布时间对齐 → 周/季特征 → 滚动 Ridge 与时间序列调参 → 各方贡献分解 → 诊断与 HTML 报告。

**本项目估计统计归因，不把相关性宣称为因果。附带结果全部为合成示例，不能当成真实市场分析或策略收益。**

## 1. 安装与第一遍运行

需要 Python 3.10 或以上；推荐 Python 3.12。在解压后的 `gold_force_project` 目录打开 PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
.venv\Scripts\python.exe -m gold_force demo --root runs/my_demo --strict-vintages
.venv\Scripts\python.exe -m pytest -q
```

Linux / WSL：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m gold_force demo --root runs/my_demo --strict-vintages
.venv/bin/python -m pytest -q
```

打开 `runs/my_demo/attribution/report.html` 或 `runs/my_demo/forecast/report.html`。网页和图表不依赖在线 CDN。也可直接打开压缩包已有的 `examples/demo_weekly/attribution/report.html`，但应先解压整个目录以保留 PNG 的相对路径。

季频示例：

```bash
python -m gold_force demo --frequency quarterly --root runs/my_quarterly_demo --strict-vintages
```

## 2. 文件结构与各模块职责

| 路径 | 功能 |
|---|---|
| `gold_force/data/http.py` | HTTPS 下载、重试、缓存、下载链接发现、SHA256、抓取记录 |
| `gold_force/data/cftc.py` | CFTC Socrata API 分页、年度 ZIP、COMEX Gold 088691 过滤、净头寸归一化 |
| `gold_force/data/spdr.py` | 当前 GLD 官方页面下载链接发现、XLSX/CSV 表头和单位字段选择 |
| `gold_force/data/fred.py` | FRED 实际利率、广义美元指数、VIX、S&P 500 数据采集 |
| `gold_force/data/wgc.py` | WGC 下载发现、工作簿检查、长表/横向季度表映射 |
| `gold_force/data/imports.py` | CSV、TSV、TXT、XLSX、JSON、只读 SQLite 导入 |
| `gold_force/data/ledger.py` | 统一台账、单位/重复值/非有限值校验、数据质量汇总 |
| `gold_force/features.py` | 周/季特征、对数收益、同比差、按已发布时间选取信息、过期数据过滤 |
| `gold_force/models/rolling.py` | 滚动 Ridge、按时间顺序 CV、训练窗口标准化、分组贡献和评分 |
| `gold_force/models/diagnostics.py` | OLS-HAC、FDR、VIF、异方差/自相关诊断、基准比较 |
| `gold_force/models/uncertainty.py` | 最新期区块 bootstrap，估计贡献分配稳定性 |
| `gold_force/report.py` | CSV、JSON、PNG、独立 HTML 报告 |
| `gold_force/pipeline.py` | 分析流程及成功/失败记录 |
| `gold_force/cli.py` | 命令行入口 |
| `configs/` | 周频、季频、WGC 映射配置 |
| `tests/` | 数据错误、时间泄漏、分解一致性等测试 |
| `examples/` | 已跑通的合成台账、配置和报告 |

## 3. 默认真实数据版本覆盖什么

| 研究对象 | 输入 | 解释边界 |
|---|---|---|
| ETF 投资者 | GLD 官方黄金持仓变化，吨 | 单只基金需求代理；不是全球 ETF，也不是投资者交易流水；费用售金等也会影响持仓 |
| 黄金收益目标（默认） | GLD 收盘价的对数收益 | 明确标记的黄金市场代理，包含基金费用、折溢价等差异，不是现货/LBMA 每盎司金价 |
| Managed Money | CFTC 净头寸占 OI 的百分比变化 | 类别头寸相关性；不能识别单个基金或每笔交易 |
| Swap Dealers | 同上 | 不能直接称为“银行自营看多/看空”，可能反映客户对冲 |
| Producer/Merchant | 同上 | 商业套保/头寸变化，不等同于矿商金条净销售 |
| Nonreportable | 同上 | 小型未报告交易者代理，不能当成全部散户实物需求 |
| 市场活跃度 | OI 的对数变化 | OI 增加本身没有净多方向：每张合约同时有多方和空方 |
| 宏观 | 广义美元指数、10 年实际利率、VIX、S&P 500 | FRED 的 DTWEXBGS **不是 ICE DXY**；无需冒充同一指数 |
| 央行/实物/供应 | WGC 季度净购买、金条金币、矿产、回收、ETF | 需要有效 WGC 下载或授权导入；按季度建模 |

默认留出 CFTC `Other Reportables` 类别作为参照，因为全部类别净头寸和恒为零。每一类别的系数是相对参照组、控制其他变量后的统计关系，不是能唯一识别的独立冲击。

GLD 与全球 ETF 变量存在包含关系，因此代码不允许把两者同时作为平行“ETF 独立贡献”。若取得真实全球周度净流量，改成 `global_etf_flow` 并移除 `gld_tonnes_change`。

## 4. 真实采集与运行

先打开 `configs/weekly.yaml` 设置开始/结束日期和根目录。日期默认截至本项目创建日前的完整日期，不会自动把未来日期填满。

```bash
python -m gold_force fetch --config configs/weekly.yaml
python -m gold_force run --config configs/weekly.yaml --mode attribution
python -m gold_force run --config configs/weekly.yaml --mode forecast
```

`run` 会先采集再建模；同一请求默认缓存 24 小时。`--refresh` 强制重新请求。默认研究预测会按假设发布延迟对齐，但仍使用当前版本历史值。

严格研究预测：

```bash
python -m gold_force run --config configs/weekly.yaml --mode forecast --strict-vintages
```

**如果历史文件今天才下载，严格模式通常得不到足够的历史训练样本。这是预期行为，不能去掉告警后称为无未来信息回测。**

如果某来源下载失败，命令以失败退出；已经成功采集的数据和下载记录仍保存，查看 `source_status.json`。不会自动把失败来源删除并给出一张看似完整的报告，也不会用合成数据补上真实数据。

### GLD 当前接口与访问失败

代码先从 `https://www.spdrgoldshares.com/usa/gld/` 发现 `historical-archive` 链接，再下载当前 XLSX 接口。初次请求曾返回 403，后续通过下载客户端成功获取。真实工作簿的说明页表示历史 LBMA Gold Price 已移除，数据表包含 Closing Price 和 Tonnes of Gold。

因此默认 `target_series: gld_price`、`sources.spdr.price_mode: gld_close`，研究的是 **GLD 收盘收益作为黄金市场代理**。代码区分 `gld_price`（USD/share）与 `gold_price`（USD/oz），不会混用；基金费用、折溢价和收盘时间都会与现货收益产生差异。

若你有合法授权的黄金价格台账，导入 `gold_price`（`usd_per_troy_oz`），把 `target_series` 改为 `gold_price`，把 `sources.spdr.price_mode` 改为 `holdings_only`。这样 SPDR 只提供持仓、独立金价源提供目标。若旧授权工作簿确实包含金价列，也可用 `price_mode: benchmark` 并指定精确列名；缺少金价列就报错。

旧 `GLD_US_archive_EN.csv` 地址在验证时返回 PDF，不能继续当 CSV 使用，本项目不采用这个旧地址。

你可从官方页面正常下载工作簿，放到 `inputs/GLD_archive.xlsx`，然后在周频配置中启用：

```yaml
spdr:
  enabled: true
  file: inputs/GLD_archive.xlsx
  price_mode: gld_close
  assumed_delay_days: 2
```

如果实际表头不同：

```bash
python -m gold_force inspect-xlsx inputs/GLD_archive.xlsx
```

再在 `sources.spdr.columns` 中填入实际列名。解析器要求所选价格列和持仓吨数列各唯一，不会把 GLD 股票收盘价当成每盎司金价。

### CFTC 特殊发布日

默认 CFTC 假设报告日后 10 天可知，明确标注为 `assumed_cftc_delay`；这比常规周五发布时间更迟，但仍不保证覆盖政府停摆、数据供应商事故等长期延迟。正确做法是导入完整真实发布日历：

```csv
observation_date,available_at
2024-01-02,2024-01-05T20:30:00Z
2024-01-09,2024-01-12T20:30:00Z
```

配置 `sources.cftc.release_calendar: inputs/cftc_releases.csv`。日历必须覆盖采集到的全部报告，否则停止。

## 5. WGC 从网络发现到季频模型

WGC 页面公开下载链接可能要求账户、访问许可或直接返回 403。代码不绕过登录。先尝试：

```bash
python -m gold_force discover-wgc --kind demand
python -m gold_force discover-wgc --kind etf
python -m gold_force download --url "从上一步得到的官方HTTPS地址" --output inputs/GDT_official.xlsx
python -m gold_force inspect-xlsx inputs/GDT_official.xlsx
```

如果发现/下载返回 403，从官方页面下载后继续。本项目创建时 WGC 官方需求与 ETF 下载都返回 403，不能宣称已自动取得全球历史数据。

有两种完整的解析路径：

1. **官方横向表直接映射**：把 `configs/wgc_rows.example.yaml` 中的行列索引、sheet、精确标签和发布日历改成你当前文件的值，复制到 `sources.wgc`。索引全部从 0 开始；示例位置不能直接当真实位置。精确标签校验阻止工作簿换版后悄悄读错行。季度行需要全球合计吨数，不能把国家行和全球总行一起相加。
2. **规范长表**：把官方季度表整理到 `inputs/wgc_quarterly_normalized.xlsx` 的 `data` sheet，字段为：

```text
date,available_at,cb_purchase_tonnes,bar_coin_tonnes,etf_demand_tonnes,mine_supply_tonnes,recycling_tonnes
```

`date` 可以是 `2010Q1` 或季度末日期；`available_at` 必须是这份数据实际发布的带时区时间。每行一个季度。当前下载工作簿可能修订历史季度，发布日列不能证明历史数值版本已经复原。

然后运行：

```bash
python -m gold_force run --config configs/quarterly.yaml --mode attribution
```

季频默认对央行、金条金币、矿产、回收使用 `本季 - 上年同季` 的吨数差，减少固定季节性和慢趋势；ETF 净需求可负，使用当季吨数。你可在 `features.py` 调整变换，再重新评估，不能只挑让结果显著的一种规格。

## 6. 通用输入台账

统一字段：

| 字段 | 含义 |
|---|---|
| `series` | 稳定变量名，例如 `gold_price`、`cb_purchase` |
| `observation_date` | 数据所属日，周总量为周五、季总量为季度末 |
| `available_at` | 实际发布时间或明确声明的估计发布时间，UTC ISO 时间 |
| `value` | 数值 |
| `unit` | 明确单位，代码对已知系列强制校验 |
| `source` | 数据来源，不可把代理数据标成全球数据 |
| `vintage_at` | 这一个具体数值版本首次被你获得/验证的时间 |
| `availability_basis` | 例如真实发布日历、假设延迟、合成首次发布 |

```csv
series,observation_date,available_at,value,unit,source,vintage_at,availability_basis
gold_price,2024-01-05,2024-01-05T23:00:00Z,2045.2,usd_per_troy_oz,verified_market_source,2024-01-05T23:05:00Z,verified_release
cb_purchase,2024-03-31,2024-04-30T12:00:00Z,290,tonnes,WGC,2024-04-30T12:05:00Z,verified_release
```

这里只展示字段形式，数值不是本项目真实分析输入。

配置导入：

```yaml
imports:
  - file: inputs/custom_observations.csv
  - file: inputs/observations.sqlite
    query: SELECT series,observation_date,available_at,value,unit,source,vintage_at,availability_basis FROM observations
```

重复 series/date 会报错，必须显式解决数据修订或重叠来源。关闭已经由导入提供的下载来源；不要静默覆盖数据。支持 `.csv/.tsv/.txt/.xlsx/.xlsm/.json/.sqlite/.db/.sqlite3`；不支持任意文本自然语言抽取、旧 `.xls` 或远程 SQL 凭据。

直接分析已有台账而不重新下载：

```bash
python -m gold_force analyse --config configs/weekly.yaml --ledger inputs/observations.csv --mode both
```

## 7. 数学定义与输出解释

目标为 `r_t = 100 × log(P_t/P_(t-1))`。默认真实配置的 P 是 GLD 收盘价（黄金市场代理）；合成示例的 P 是合成黄金价；导入授权金价并切换 target_series 后，P 才是你的直接金价序列。

每个滚动训练窗口中拟合均值、标准差，`z_jt = (x_jt - mean_train)/std_train`。Ridge 估计 `r_t = intercept + sum(beta_j × z_jt) + residual`。

`贡献_j = beta_j × z_jt`；组贡献是该组所有特征贡献之和。`force_score = 100 × 组贡献 / sum(abs(各组贡献))`；分母为 0 时全部得分为 0。

**贡献是相对训练窗口均值的偏离贡献，不是该因素造成的全部收益。** 例如资金流本周为正，但小于历史均值，可能得到负贡献。基准截距单列，未解释残差单列。

贡献、截距、残差可以精确加总为对数收益百分点；普通简单收益应使用 `100 × (exp(r/100)-1)`，不能再把各项单独指数化后直接加总。

正负 force_score 的绝对值之和为 100；带符号之和不一定为 100。它不是某方占实际涨幅的百分比，也没有提供跨周/季可直接相加的统一吨数。

`__beta_z` 表示变量增加 1 个训练期标准差时，对模型输出的变化；`__beta_raw` 是原始单位斜率。例如 ETF 吨数变化多 100 吨，在该模型局部对应 `100 × beta_raw` 对数收益百分点，而不是已验证的因果影响。

## 8. 两类分析严格区分

### attribution：事后解释

使用本期已实现的变量解释本期收益，系数只用之前的窗口拟合。当前版本修订历史可能参与计算；这是回顾性、窗口外解释，不是可交易样本外回测。季度央行数据发布后才可能解释所属季度。

### forecast：按发布信息对齐的历史研究预测

每期起点为上一完整周五/季度末纽约时间 23:59。每个特征选择截止此时已发布的最新完整周期，超过最大允许年龄就缺失。训练时只纳入截止起点已知的历史收益。周五基准价格不是 23:59 可成交价格，所以输出仅研究收益预测，**没有交易、滑点、保证金、执行收益回测**。

这个版本产生完整历史预测序列；没有连接券商或实时订单，没有声称持续运行的实时行情系统。

严格模式将信息可用时间设为 `max(available_at, vintage_at)`。当前历史下载不能假装当年已经获知这个版本；缺乏历史首次发布快照时严格模式会拒绝历史预测。台账只允许每个系列/观测日一个明确版本；要做全面 point-in-time 修订数据库，还需接入 ALFRED 等历史版本并按每个预测起点重建台账。本项目保留原始抓取快照，但没有自动拼出多版本历史数据库。

FRED 实际利率、VIX、股指默认延迟 2 天；广义美元指数默认 10 天，考虑 H.10 的周度批量发布。所有这些都是明确的研究假设，不是逐条精确发布日复原。

缺失周期不会无限填充。一个价格周缺失时，下周差分也缺失，不把两周涨幅冒充一周涨幅。月/季总量不会复制成数十个独立周/日样本。

## 9. 如何判断模型是否有用

- `MAE/RMSE`：误差大小，单位为对数收益百分点。
- `r2`：窗口外拟合优于目标自身均值到什么程度，可以为负。
- `skill_vs_rolling_mean` 和 `skill_vs_zero`：是否优于历史均值/零收益预测，负值代表更差。
- `direction_accuracy`：方向命中率，需同时对比市场偏向正收益的基准，不能单独当策略能力。
- `__beta_z` 随时间变化：观察某一方的统计关系是否稳定。
- VIF 与条件数：检查共线性。高度共线下单方归因很不稳定，正则化不等于识别因果。
- 贡献区间：最新期的 moving-block pairs bootstrap 反映系数和分配的不稳定性；固定已选 alpha，未覆盖模型选择、数据修订、所有非平稳性或因果不确定性，不是收益预测区间。
- OLS-HAC 是另外一个全样本描述性分析；`q_value_bh` 控制多重检验。没有给 Ridge 系数编造 p-value，也不能据此证明 Granger 因果或结构冲击。

附带周频合成例子：事后解释 R² 约 0.372，历史预测 R² 约 -0.070，展示了解释与预测的区别。合成数据的生成规则特意包含同期关系，不是实际黄金绩效。

## 10. 输出文件

| 文件 | 内容 |
|---|---|
| `data/processed/observations.csv` | 统一观测台账 |
| `data/raw/` | 原始下载快照与 URL/哈希/抓取时间 |
| `source_status.json` | 每个数据源的成功或失败 |
| `download_manifest.json` | 本次实际使用的下载与缓存记录 |
| `data_quality.json` | 覆盖范围、单位、缺口、来源和时间假设 |
| `analysis_status.json` | 模型运行结果与失败原因 |
| `attribution/force_history.csv` | 每期因素贡献、组贡献、系数、均值/标准差、残差 |
| `attribution/latest_force.csv` | 最新一期归因与 bootstrap 区间 |
| `attribution/feature_availability_audit.csv` | 特征所属时间、发布时间、版本时间 |
| `attribution/design_matrix.csv` | 实際训练/测试输入 |
| `attribution/skipped_periods.csv` | 未估计时期及原因 |
| `attribution/ols_hac_descriptive.csv` | OLS-HAC 统计量与 FDR |
| `attribution/vif.csv` | 共线性 |
| `attribution/metrics.json` | 误差、基准比较和限制 |
| `attribution/report.html` | 报告与图表 |

`forecast/` 下输出同类文件，但解释为按当时信息构造的研究预测。两个频率分别输出，不能把季频贡献硬塞进同一个周频涨幅分解。

## 11. 下一步扩展位置

- 全球周度 ETF：新增台账 `global_etf_flow`（真实周总量，吨，周五所属日期）并替换 GLD 特征。
- IMF/央行月度储备：新增独立月频模型；需要保留真实发布滞后与修订版本。不要把“储备变化”直接与季度净购买混用。
- LBMA/CME：在数据层增加授权下载器，并把成交量、持仓、需求分开，不把总 turnover 当净买入方向。
- 市场状态：在 `features.py` 增加只由起点已知信息定义的交互项，再用时间 CV 验证；本版未加入主观贴标签的危机分类。
- 因果推断：有有效识别假设后另加事件研究/IV/结构模型，不能仅把这个回归命名为因果。
- 与你的股票配对交易工程整合：复用台账、原始快照与 CV 模块；交易层仍需独立实现成交价、成本、资金、保证金等规则。

## 12. 官方数据说明（核验日期 2026-10-07）

- GLD 产品与当前历史下载：https://www.spdrgoldshares.com/usa/gld/
- CFTC API 数据集：https://publicreporting.cftc.gov/resource/72hh-3qpy.json
- CFTC 历史 ZIP：https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
- CFTC 发布日历：https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm
- WGC ETF：https://www.gold.org/goldhub/data/gold-etfs-holdings-and-flows
- WGC 季度供需：https://www.gold.org/goldhub/data/gold-demand-by-country
- FRED DFII10：https://fred.stlouisfed.org/series/DFII10
- FRED DTWEXBGS：https://fred.stlouisfed.org/series/DTWEXBGS
- FRED VIX：https://fred.stlouisfed.org/series/VIXCLS
- FRED SP500：https://fred.stlouisfed.org/series/SP500

数据源条款和访问权限以其当前官方页面为准。压缩包不附带第三方完整真实历史数据，只附带明确标记的合成示例和验证状态。
