# 运行与结果交接

## 本机快捷入口

```powershell
cd D:\TFM_proposal\harness_mvp
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 plan --stage P1
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 doctor
```

`plan` 不推理；`doctor` 不推理但会读取全部权重与数据做核对。入口只改变当前进程及子进程的缓存路径，不修改系统 PATH 或全局环境。

可直接使用 Python，注意先加载已有环境设置：

```powershell
. D:\TFM_proposal\tfm_runtime\env.ps1
& D:\TFM_proposal\tfm_runtime\envs\tfm\Scripts\python.exe -m harness plan
```

参数顺序：全局 `--config` 和 `--output` 放在子命令之前；如 `python -m harness --output D:\TFM_proposal\tfm_runtime\pilot_runs\new run --stage P1 --gpus 0`。

## 只试新接口：合成检查

下面只生成 128 fit、16 validation、16 test 的合成表格，含一列类别特征，不使用正式六数据集。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 --output D:\TFM_proposal\tfm_runtime\verification\my-new-smoke smoke --model tabpfn --task multiclass
```

`--model` 可为 tabpfn/tabicl/limix，`--task` 可为 binary/multiclass/regression。每次使用新目录，保留旧检查证据。这个直接检查命令不带 runner 的 20 分钟终止器，仅适合上述微型数据。

## 正式实验

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 run --stage P1 --gpus 0
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 run --stage P2 --gpus 0
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 run --stage P3 --gpus 0
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 summarize
```

P1/P2/P3 分别新增 6/12/36 单元，不自动包含前序阶段。也可 `run --stage all` 执行全部尚未完成项。三卡使用 `--gpus 0,1,2`；所有阶段保持同一组设备。一次单元只占一张卡。

20 分钟从子进程启动计算，包含 Python/库导入、权重加载、数据核对、fit、validation/test 预测及落盘。下载不发生在执行期间。核心上限 18 GPU 小时，不是实测预计时长；失败重跑另计，协议预留诊断额度 2 GPU 小时，请根据每次 attempt 墙钟控制追加投入。

本机只有 6GB 显存，框架不会为适应显存偷偷削减样本/内部成员。完整六数据集和 4090 性能需正式运行后确认。

## 恢复与失败

- 正常重复命令会显示 REUSE 或 KEEP FAILURE，成功项不重算，失败默认不重试。
- 原因已处理且代码/配置未变时，使用 `run --stage P1 --gpus 0 --retry-failed` 追加新尝试，旧 attempt 保留。
- 代码/配置/环境/主机/GPU 集变化：指定新 `--output`。旧目录不覆盖；不能拼接不同配置的结果作为同一实验。
- 查看 `units/<dataset>/s<split>/<model>/attempt-xxx/worker.log` 与 traceback.txt；latest result.json 有状态和错误类型。
- OOM/timeout/error 都保留实际开销，不能记成快速成功。
- 中断/断电可能留下 `.run.lock`；先查看其主机/PID，确认没有活动运行，再处理锁。不要同时启动两个指向相同输出目录的运行器。

## 汇总

`summarize` 可在任意阶段后运行，报告明确写出缺失数量；正式结论应等核心覆盖完成。质量表只比较三个模型都成功的相同数据集/划分，同时完整导出 coverage.csv。

四张主要表在 report.md：逐数据集质量、选择收益、选择空间、成本失败。详细指标可以用 Excel 打开带 BOM 的 CSV。误差条是三个划分间标准差，不是置信区间；小任务数量不能支持强泛化结论。

成本 CSV 分别保存实际完整单元墙钟和阶段求和估算；固定模型 test 推理发生在 validation 之后，可能受预热/缓存影响，因此冷启动值仅为估算。后续部署成本需完整流程实测。

交接结果时，打包完整的本次输出目录（run-manifest、units、analysis），只传 report.md 不足以独立复核。权重与原数据不必重复传，但需保留同哈希可访问副本。不要把大结果目录提交 Git。
