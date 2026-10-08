# Harness + TFM MVP

这个仓库用于实现论文项目的第一个最小闭环：由 LLM 读取任务与数据摘要，选择并调用一个表格基础模型（TFM），接收运行结果，再生成一份简短报告。

当前仓库只有目录骨架和本文档，尚未包含任何可执行代码、配置文件或实验结果。

## 第一阶段目标

先用最小规模验证整个 Harness 流程能够跑通：

1. 加载一个小型公开表格数据集。
2. 生成结构化的数据集摘要。
3. 将任务、数据摘要和可用 ToolCard 交给 LLM。
4. LLM 返回结构化动作，选择 TabPFN。
5. Harness 调用 TabPFN 完成训练与预测。
6. 将指标和运行信息返回给 LLM。
7. LLM 判断任务完成，并输出最终报告。

计划采用的最小组合：

- 数据集：scikit-learn Breast Cancer Wisconsin
- TFM：TabPFN v2
- LLM：DeepSeek Flash
- Harness：普通 Python 状态循环，LLM 接入使用 LangChain 的结构化输出能力
- 任务：二分类
- 划分：固定随机种子的训练集/测试集划分

第一阶段只验证闭环和接口，不以获得有说服力的科研结论为目标。

## 预期流程

```text
用户需求与数据
      ↓
DatasetCard：数据集的结构化摘要
      ↓
LLM Router：决定下一步动作
      ↓
Tool Registry：找到并调用 TabPFN 工具
      ↓
Observation：返回指标、耗时和错误信息
      ↓
LLM Router：继续调用或结束
      ↓
运行记录与最终报告
```

LLM 只负责决策和解释；数据加载、模型调用、指标计算和文件保存由确定性的程序完成。原始表格数据默认不发送给 LLM，只发送统计摘要和字段信息。

## 当前目录结构

```text
harness_mvp/
├── README.md
├── prompts/          # 后续存放 Router 提示词
├── toolcards/        # 后续存放模型能力与限制的结构化描述
├── harness/          # 后续存放 Harness 核心逻辑
│   └── tools/        # 后续存放 TFM 适配器和工具注册逻辑
└── runs/             # 后续存放每次运行的结果与轨迹
```

## 核心接口草案

实现时优先固定数据结构，让每个模块可以独立替换。

### DatasetCard

描述当前任务和数据，不包含原始样本。建议至少包括：

- 数据集名称与任务类型
- 样本数、特征数和类别数
- 数值/类别特征数量
- 缺失值情况
- 类别分布
- 训练集与测试集规模

### ToolCard

描述一个 TFM 能做什么以及何时适合调用。建议至少包括：

- 工具名称与版本
- 支持的任务类型
- 输入要求
- 数据规模或特征类型限制
- 资源需求
- 调用参数
- 输出字段
- 已知失败情形

### Action

LLM 每轮只能返回预定义动作：

- `run_model`：选择工具并给出受约束的参数
- `finish`：结束循环并生成报告

### Observation

工具运行后返回给 Harness 和 LLM 的结构化结果，建议包括：

- 运行状态
- 验证指标
- 运行耗时
- 预测结果保存位置
- 错误类型与错误摘要

### Harness State

保存一轮任务的完整状态：任务描述、DatasetCard、可用 ToolCard、动作历史、Observation 历史和终止原因。

## 计划输出

一次成功运行最终应在 `runs/` 下产生：

- `report.md`：面向用户的简短结果报告
- `result.json`：指标、参数和运行元数据
- `predictions.csv`：测试集预测结果
- `trace.jsonl`：LLM 决策、工具调用和观察结果的逐步轨迹

这些文件目前尚未创建。

## 第一周验收标准

- 使用一个命令启动完整流程。
- LLM 能基于 DatasetCard 和 ToolCard 返回合法的结构化动作。
- Harness 能根据动作调用 TabPFN v2。
- TabPFN 能在 Breast Cancer Wisconsin 数据集上完成一次二分类预测。
- 指标由程序计算，并作为 Observation 返回给 LLM。
- LLM 能结束循环并生成报告。
- 一次运行可以通过固定随机种子复现。
- 失败时留下可定位问题的运行轨迹。

## 暂缓实现的能力

以下能力保留扩展位置，但不进入第一个可运行版本：

- 多个 TFM 之间的真实路由
- 多模型 ensemble
- adaptive portfolio construction
- ToolCard 自动更新
- Harness 自主迭代和经验记忆
- 大规模数据集与严格论文实验

后续增加新模型时，应通过新的工具适配器和 ToolCard 接入，避免修改 Harness 主循环。这样可以反复调整路由规则、提示词和模型组合，而不需要重写整个项目。
