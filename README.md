# Harness_TFM · 路由选择空间 Pilot

先回答一个具体问题：**TabPFN-3.5、TabICLv2、LimiX-2 是否在不同数据集上各有优势，验证反馈能否帮助选中合适模型？**

当前交付是三阶段实验框架：3 个模型 × 6 个数据集 × 3 个固定划分 = **54 个正式评测单元**。每个单元分别生成 validation/test 预测，统一记录耗时、资源与失败，最后生成对照表和报告。完整协议见 [docs/protocol.md](docs/protocol.md)，实现说明见 [docs/architecture.md](docs/architecture.md)。

目前接入的是三个 TFM 和验证集选择器。LLM 路由、自迭代、跨模型 ensemble 尚未实现；它们可以复用 adapter、decision context 和预测文件，不需要推翻实验执行层。ToolCard 是静态描述，尚未作为路由输入。

## 1. 本地目录

```text
D:\TFM_proposal\
├── harness_mvp\              # 本 Git 仓库：代码、配置、文档、小型检查
└── tfm_runtime\              # 前两步已准备，重型文件不上传 GitHub
    ├── envs\tfm\             # 已安装的共享 Python 环境
    ├── sources\LimiX\        # 固定 commit + 缓存路径补丁
    ├── weights\              # 四份权重
    ├── data\                 # 六数据集与 18 组索引
    ├── cache\、tmp\          # 缓存、临时文件，均在 D 盘
    ├── verification\         # 接通与框架检查证据
    └── pilot_runs\core-v1\   # 正式实验开始后才创建
```

本机运行入口会复用 `tfm_runtime/env.ps1` 和现有环境，不需要重装三个模型。Windows 下运行目录和输出目录强制在 D 盘。新机器配置见 [docs/environment.md](docs/environment.md)。

## 2. 从不消耗模型算力的检查开始

在 PowerShell 中：

```powershell
cd D:\TFM_proposal\harness_mvp
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 plan
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 doctor
```

`plan` 只列出 54 个计划单元。`doctor` 核对软件版本、4 份权重、LimiX 源码/配置、6 数据集和 18 组索引，不启动模型推理。后者会读取约 3GB 权重计算哈希，等待一会儿是正常的。

## 3. 正式运行：由用户在第四步执行

以下命令会使用 GPU 执行正式实验。工程检查使用合成数据的 `smoke` 命令，详见 [docs/running.md](docs/running.md)。

```powershell
# P1：credit-g、concrete 的 S0，共 6 单元
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 run --stage P1 --gpus 0

# P2：其余 4 数据集 S0，新增 12 单元
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 run --stage P2 --gpus 0

# P3：六数据集 S1/S2，新增 36 单元
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 run --stage P3 --gpus 0

# 冻结验证集选择，然后读取测试标签评分并生成报告
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 summarize
```

有三张可用卡时，将全部阶段的 `--gpus 0` 统一换成 `--gpus 0,1,2`。同一数据集/划分的三模型会在同一卡上串行运行。每个单元单独子进程，20 分钟硬上限，超时终止整个子进程树。

同一命令可以重复执行：哈希相同且文件完整的成功项直接复用；失败项保留，需要显式 `--retry-failed` 才新增尝试。代码、配置、环境、硬件变化时必须选新输出目录，避免混用结果。不要在实验进行中编辑代码。

## 4. 模型、数据与指标

| 模型 | 固定配置 |
| --- | --- |
| TabPFN-3.5 | tabpfn 9.2.0，官方 V3_5 标准配置；记录 auto 解析值 |
| TabICLv2 | tabicl 2.2.0，8 个内部成员，分类/回归分别使用本地权重 |
| LimiX-2 | 官方源码 2.0.0；noretrieval_v2 配置，分类 32、回归 8 |

| 数据集 | OpenML Task | 类型 | 行数 / 特征 |
| --- | ---: | --- | ---: |
| credit-g | 363626 | 二分类 | 1000 / 20 |
| qsar-biodeg | 363696 | 二分类 | 1054 / 41 |
| website_phishing | 363707 | 多分类 | 1353 / 9 |
| splice | 363702 | 多分类 | 3190 / 60 |
| concrete_compressive_strength | 363625 | 回归 | 1030 / 8 |
| healthcare_insurance_expenses | 363675 | 回归 | 1338 / 6 |

采用 OpenML 官方外层 repeat=0/1/2、fold=0、sample=0；每个外层训练集再留 20% validation。**不使用 fit+val 重训，不合并 val/test 查询。**

主指标：二分类 AUC，多分类 log-loss，回归 RMSE。跨数据集汇总用同分区常量基线归一化损失，先平均划分再平均数据集。比较三个固定模型、RandomSelect 损失期望、ValSelect，以及明确标为事后上限的 Oracle。SBS/VBS/G_type 使用先平均划分后选数据集赢家的口径。

这是使用 TabArena 数据和划分的简化开发实验；不是完整官方排行榜复现，也不能以 54 个单元声称有 54 个独立任务。

## 5. 输出在哪、交给学长看什么

默认在 `D:\TFM_proposal\tfm_runtime\pilot_runs\core-v1\`：

```text
run-manifest.json                     # 冻结的代码/配置/环境/硬件/数据身份
units/<dataset>/s0/<model>/
├── result.json                       # 最近一次结果/失败状态
└── attempt-001/
    ├── request.json、worker.log、result.json
    └── validation.npz、test.npz       # 预测+原始行号，不存查询标签
analysis/
├── report.md                         # 四类汇总表与解释边界
├── metrics.csv、coverage.csv
├── selection_decisions.csv、selections/<hash>.json
├── method_summary.csv、selection_space.json
├── fixed_costs.csv、search_costs.csv、cost_summary.csv
├── pairwise_advantages.csv、summary.json
└── relative_gap.png、quality_cost.png
```

模型接口没有 validation/test 标签参数。选择器只接收 validation 损失；选择落盘后才计算 test 分数。这是代码层面的访问边界，不是操作系统安全隔离，底层 parquet 仍包含完整目标列。测试成绩不写回 ToolCard 或共享 memory。

完整性不足时报告展示失败/缺失数量，质量仅在三模型共同成功的组内比较，不填补失败成绩。耗时区分实际全部运行开销和由阶段计时推算的选择流程成本；后者不是在线部署实测。

## 6. 代码从哪里读

```text
configs/          实验参数 + 锁定资产身份；不存权重本体
harness/
  cli.py          命令入口
  config.py       阶段、模型、数据配置
  data.py         固定划分、保留列类型、fit 目标映射
  adapters.py     三个 TFM 的唯一接入点
  worker.py       单模型单划分执行、预测、同步计时
  runner.py       GPU 分配、超时、进程隔离、断点恢复
  routing.py      可替换选择接口；当前为 ValSelect
  metrics.py      主指标和常量基线
  evaluation.py   冻结选择、最终评分、CSV/报告/图
toolcards/        静态模型说明，暂无成绩记忆
scripts/          PowerShell 入口和资产准备脚本
tests/            CPU 合成检查，不比较真实模型性能
docs/             协议、架构、操作和验收记录
```

先读 `config.py → adapters.py → worker.py → runner.py → routing.py → evaluation.py`。修改选择逻辑通常只涉及 `routing.py`；增加模型主要涉及 adapter、ToolCard、配置和对应检查。

工程检查命令：

```powershell
. D:\TFM_proposal\tfm_runtime\env.ps1
& D:\TFM_proposal\tfm_runtime\envs\tfm\Scripts\python.exe -m unittest discover -s tests -v
```

检查完成状态与证据见 [docs/verification.md](docs/verification.md)。GitHub 的 CPU 检查不安装 TFM、不下载权重、不启动正式实验。
