# AGENTS.md

本仓库：Polymarket 温度预测系统（Phase 1，高精度物理概率模型）。当前执行规格以《项目执行文件 v5.9.1(细化版).md》为准（v5.9 的更新版，无冲突）；需求来源《项目方案：Polymarket 温度市场量化投注系统 (v2.2).md》。

## 规格更新流程（触发即执行）

当用户出现以下任一情形时，**执行 `docs/spec-update-process.md` 的完整流程**：

- 提供新版本《项目执行文件》（如 v5.8）或《项目方案》，并要求"走流程"/"走标准流程"/"规格更新"
- 说"grill with docs"、"to spec"、或"更新规格文档"

流程四步：文档审读 → grill-with-docs（grilling + domain-modeling，产出 ADR/术语表）→ to-spec/to-tickets（生成 spec + 拆分 ticket + 发布命令）→ 冻结规则校验。发布命令生成后**交给用户执行**，agent 不代跑 gh。

## 硬性规则

- **已完成的任务是冻结的历史记录**（如 task 文件中的 Task 1.1），内容永不改写；只更新未完成任务的描述与相关设计文档。
- 设计文档更新采用**版本标注**（"已对齐 vX.Y"），不重写既有内容。
- 项目内部文档版本号由 agent 决定（核心架构变更 → 次版本 +1；细节修正 → 修订号 +1），并向用户报告；用户提供的输入文档版本号由用户定。
- 用 `gh` 代跑（如 issue 验收/关闭）遇认证或权限报错时，**不得反复重试**：先请用户完成认证后再重跑一次；持续报错则停止。

## 代码生成约束

- 修改文件时避免重写整个文件，始终优先使用 `edit` 工具进行细粒度、精确的 diff 修改。
- 若文件创建或重构的规模很大，将任务拆分为多个步骤/函数分步完成。
- 工具调用前后的注释与说明保持简洁。

## 数据管道验证工作流（Task 1.2 GEFS，硬性）

- **每张 ticket 完成时必跑真实冒烟**：`RUN_NETWORK_TESTS=1 python -m pytest tests/unit/data_acquisition/test_gefs_fetcher.py::test_network_reforecast_single_message -q`。ticket 不涉及网络路径时可豁免，但须在交付说明中注明理由。**mock 全绿 ≠ 真实网络可用**（T01 与 T02 均已实证：mock 绿但真实 AWS 路径坏）。
- **外部契约必须钉成契约测试**：GRIB idx 搜索正则、真实解码变量名（tmax/tmin）、网格方向/分辨率等，必须用「观测到的真实数据字面量」写成快速单元测试（参考 `TestBuildSearch`），mock 不得覆盖这些契约。
- **mock 仅 mirror 已验证的契约**：`MockHerbie` / `make_fake_forecast_ds` 只反映真实联网验证过的行为，不得凭空假设。

## 其他

- 开发协作、沟通模式（提问/执行/确认）见 `DEVELOPER_GUIDELINES.md`。
