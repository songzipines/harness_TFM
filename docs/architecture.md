# 框架与修改位置

## 1. 当前闭环

```text
pilot.json + assets.lock.json
          ↓
plan / doctor：固定实验单元并检查输入
          ↓
runner：分配 GPU、固定随机顺序、建立 attempt、启动子进程
          ↓
worker：读取 fit X/y 和查询 X → adapter.fit → 分别预测 val/test
          ↓
保存预测、哈希、时间、资源、实际配置；失败也记录
          ↓
evaluation 第一阶段：validation 标签+预测 → Router.select → 选择落盘
          ↓
evaluation 第二阶段：读取 test 标签 → 固定模型/选择器/Oracle 评分
          ↓
CSV + report.md + 两张图
```

本轮不用 LangChain/LangGraph。固定阶段实验用普通 Python 就足够；LLM 将来只需实现相同的 `Router.select(context)`，无需接管 GPU 子进程和指标计算。

## 2. 为什么拆成这些模块

| 变化 | 主要修改处 | 已有部分可复用 |
| --- | --- | --- |
| 替换 TFM 或模型 API 改动 | adapters.py、锁定配置、ToolCard | 数据划分、runner、指标、报告 |
| 改路由依据 | routing.py 的 DecisionContext / Router | 现有冻结预测可离线回放 |
| 加预算决策 | context 中增加可见成本与预算，再实现 Router | 超时和逐阶段成本记录 |
| 加跨模型 ensemble | 新增组合器，读取同一行号/类别顺序的预测 | adapters、预测格式、独立评分层 |
| 自迭代 / ToolCard 更新 | 独立增加离线更新与发布流程 | 尝试记录、身份哈希、来源信息 |
| 改数据、fit 上下文、模型配置 | 新协议/新输出目录 | 可复用框架，但旧预测不能当新实验 |

`DecisionContext` 当前只有数据集标识、划分、任务类型和验证损失。以后可增加 fit 摘要、语义、冻结 ToolCard、预算；不要加入 test 标签或 test 分数。六数据集随后可作为开发任务，不能再冒充未见最终评测任务。

## 3. 重要约定

- 特征保持 DataFrame 的列类型，不能把混合表格摊成 object ndarray。曾在接通阶段发现该转换会使 LimiX 把数值列误判为类别。
- fit 的类别元数据去掉未出现水平；查询类别值保留，让模型官方预处理处理未知值。目标映射仅由 fit 学得。
- 分类概率按 fit 目标类别顺序对齐，检查形状、有限值、范围与行和。log-loss 统一截断并归一化。
- validation/test 各自调用 predict。官方内部无监督处理可能使用当前查询 X；LimiX 的 `all_data` 变换明确记录这一点，不将其描述为完全归纳式。
- TabPFN/TabICL 可能延迟到 fit 才加载权重；阶段名表示 API 边界，不能把构造器时间当作完整加载时间。
- LimiX 的 fit 只保存上下文，每次 predict 都包含训练上下文处理，这部分时间完整计入 predict_s。
- 每单元一个子进程；超时杀进程树。显存由子进程释放，不需要常驻三个模型。
- 每次尝试有独立目录；同代码/配置/环境/硬件下才恢复。成功预测必须通过 SHA256 才可复用。
- 同一输出目录用锁防止两个 runner 或 summarize 同时写入。异常断电后可能留下锁，需先检查其中主机/PID 确实不再运行，再移除。

## 4. 保存与评分边界

预测 npz 只有 `row_ids`、`prediction`。分类 prediction 为 N×K，回归为 N；查询标签不存入预测。类别映射存于单元 result.json。数据、行索引、权重和代码身份在 run-manifest 中。

评分程序先生成不可变命名的 `selections/<hash>.json`，随后才读取 test 目标并算分。重新汇总会产生当前覆盖相应的 selection hash，旧选择文件保留。

这只是一套可审查的工程边界，不提供恶意代码防护：模型工作进程仍能访问包含完整 y 的本地 parquet。未来若研究要求强隔离，可预先导出无 test-y 的输入文件，再让评分进程独占标签目录。

## 5. 没有自动做的研究决定

本轮不根据 P1/P2 成绩换数据、减内部成员或改主指标，不自动启动 P4。失败是否源于模型能力、硬件上限还是 adapter 错误，需要诊断；修复代码后应换输出目录，并作废/重跑所有受影响项，不能只补差成绩。

P0 官方成绩缓存分析保留为协议中的独立工作，本框架没有实现其下载/分析。当前也没有跨任务训练的 LLM、Jev 或可学习路由器。
