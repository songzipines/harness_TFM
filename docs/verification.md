# 第三步验收记录

日期：2026-10-21。验收机器：Windows 11、RTX 3060 Laptop 6GB，沿用步骤 1/2 的共享环境。服务器和 4090 尚未验证。

## 已完成的检查

| 检查 | 结果 | 本地证据（仓库外） |
| --- | --- | --- |
| CPU 框架检查 | 13 项通过 | `../tfm_runtime/logs/framework-tests-final.log` |
| 冻结资产预检 | 4 份权重、源码/配置、6 数据集、18 组索引通过 | `../tfm_runtime/verification/framework-doctor.json` |
| 独立官方划分复核 | 18 组通过 | `../tfm_runtime/verification/data_checks.json` |
| PowerShell 包装入口 | plan --stage P1 正确列出 6 单元 | `../tfm_runtime/logs/powershell-entry-plan.log` |
| 代码与配置语法 | 18 个 Python 文件及 JSON 可解析 | 本轮本地检查；CI 继续执行 CPU 用例 |

13 项 CPU 检查覆盖：54 单元与阶段数量、验证选择及平局规则、常量基线/主指标、概率类别列对齐及无效输出、实际精度配置导出、SBS/VBS 口径、写锁、进程超时、混合列类型与 fit 类别边界、篡改数据/索引拒绝、全流程调度/恢复/报告、失败显式重试与尝试保留/身份变化拒绝、预测损坏拒绝。

其中调度集成测试使用微型合成表格和模拟工作进程，**不调用真实 TFM，不是 54 次正式评测**。该测试同时构造“验证偏好 PFN、测试偏好 LimiX”的情形，核对选择不会跟随测试答案。

## 新 adapter 的真实 GPU 接通检查

每项合成数据为 128 fit / 16 validation / 16 test，6 列中有 1 列类别、5 列数值。fit 保持 categorical，查询类别列转成 object 值，与正式输入边界一致。模型内部成员保持冻结配置，不为检查削减。

| 模型 | 二分类 | 多分类 | 回归 |
| --- | --- | --- | --- |
| TabPFN-3.5 | 通过 | 通过 | 通过 |
| TabICLv2 | 通过 | 通过 | 通过 |
| LimiX-2 | 通过 | 通过 | 通过 |

九项最终检查的原始证据在 `../tfm_runtime/verification/framework_final/<model>_<task>/`，各项有 `smoke-result.json`、validation/test 预测、实际配置和代码身份。计分仅为检查输出有效，不能把这些合成分数当成模型排名。

初次检查的三个 LimiX 调用在 Codex 工具沙箱中遇到 Triton 编译器子进程的 Windows DLL 重定位错误；在正常本机进程复核成功。初次失败记录保留于 `verification/framework_v1/limix_*/`，不计为有效模型性能结果。没有为通过检查修改 LimiX 权重或推理逻辑。

最终验收期间还发现并修复 runner 将 OOM 状态覆盖成一般 error 的分支，已由显式失败/重试 CPU 用例复验。随后补全 TabICL 实际推理配置字段导出；相关检查见 `verification/framework_metadata/tabicl_multiclass/`。这些调度/元数据变动不调整模型预测参数。

## 未执行的内容

- 六数据集正式模型评测：**0 / 54**。默认 `pilot_runs/core-v1` 尚未创建。
- P0 官方缓存成绩分析：未执行。
- P4 扩展任务：未准备或启动。
- 4090/Linux 服务器整机验收、在线完整路由成本：未验证。
- LLM 路由、ensemble、自迭代有效性：未实验。

用户在第四步执行正式实验后，再用保存预测生成质量/成本结论；不能把工程接通、CPU 模拟或文件存在当成研究结果。
