# 规格更新标准流程（Spec Update Process）

当用户提供新版本《项目执行文件》或《项目方案》并要求走流程时，按以下四步执行。**每一步有明确完成标准**，未达标不进入下一步。

---

## ① 文档审读

通读新版本《项目执行文件 vX.Y》+《项目方案 vX.Y》，对照 `specs/` 目录现有文档与旧版本执行文件，产出**冲突清单**：

- 新版本覆盖了哪些旧协议（逐条列出）
- 哪些协议保留不变
- 哪些内容与既有文档矛盾、需要同步更新

> 原则：**新版本优先**；有冲突以新版本为准。

**✅ 完成标准**：冲突清单完整列出，且明确每一条的处理（覆盖/保留/待确认）。清单呈现给用户确认。

---

## ② grill-with-docs

运行 `/grilling` 会话（设计树逐轮追问）+ `/domain-modeling`（产出 ADR 与术语表）。

**执行要求**：
- 每轮只问当前 frontier 的问题（前提已解决的决策），编号提问并附推荐答案
- 一轮问完整 frontier，等用户回答后再进下一轮
- 事实查找由 agent 完成（读文件、查数据、算天文时间等），不向用户索要可自查的事实
- 决策实时记录；涉及架构/口径的决策产出 **ADR**（`docs/adr/ADRNNN-标题.md`），术语变更更新术语表（`CONTEXT.md`）

**✅ 完成标准**：设计树 frontier 清空（无悬而未决的决策）；每个决策有记录（ADR 或决策日志）；术语表与最终决策一致。

---

## ③ to-spec / to-tickets + 本地文档更新

### 3.1 生成 spec
将收敛的决策合成为 spec（模板见附录 A），内容包括 Problem Statement / Solution / User Stories / Implementation Decisions / Testing Decisions / Out of Scope。

### 3.2 拆分 tickets（to-tickets）
将 spec 拆成**垂直切片 ticket**（tracer bullet）：
- 每个 ticket 是可独立演示的完整纵向切片，声明 **Blocked by**（依赖哪些前置 ticket）
- 无依赖的 ticket 可立即开始（frontier）
- 先呈现拆分方案给用户确认（粒度、依赖关系），批准后再生成发布命令

### 3.3 生成发布命令（交给用户执行）
生成现成的 gh 命令（见附录 B），包含：spec issue + 各 ticket issue + `ready-for-agent` 标签。**agent 不代跑 gh**——用户执行后反馈 issue 编号，agent 记录编号回填到本地文档。

### 3.4 同步更新本地 specs/ 文档
按冻结规则（第④步）把决策同步进 `specs/` 相关文档：未完成任务的描述、设计文档的口径、TDD 计划等。**已完成的任务内容不动**。

**✅ 完成标准**：用户已执行发布命令并反馈全部 issue 编号；本地 `specs/` 文档已同步且可追溯到决策记录。

---

## ④ 冻结规则校验

**已完成的工作是冻结的历史记录，永不改写**。校验：

- `specs/implementation-tasks-phase1.md` 中已完成任务（如 Task 1.1）的正文**零改动**——只允许更新其状态标记或在其后新增说明
- 设计文档的既有内容不重写——更新采用顶部**版本标注**（"已对齐 vX.Y"）+ 定点修正
- 版本号：项目内部文档由 agent 决定（核心架构变更 → 次版本 +1；细节修正 → 修订号 +1），向用户报告；用户提供的输入文档版本号由用户定

**✅ 完成标准**：`git diff` 检查确认不含对已完成任务内容的任何改动。

---

## 附录 A：Spec 模板（to-spec）

```markdown
## Problem Statement
用户视角的问题。

## Solution
从用户视角的解决方案。

## User Stories
1. As an <actor>, I want a <feature>, so that <benefit>
（长列表，覆盖所有方面）

## Implementation Decisions
- 将构建/修改的模块、接口、技术澄清、架构决策、模式变更、API 契约
（不写具体文件路径和代码片段，会过时）

## Testing Decisions
- 好测试的标准（只测外部行为）、哪些模块被测、测试先例

## Out of Scope
本 spec 不做的事。

## Further Notes
补充说明。
```

## 附录 B：发布命令（gh CLI，用户执行）

### 一次性设置（每仓库一次）
```bash
# 1. 启用 Issues：GitHub 网页 → 仓库 Settings → Features → Issues 打勾
# 2. 创建标签：
gh label create ready-for-agent -R oasislin/Poly2-Phase1 --color "0E8A16" --description "Spec/ticket 可供实现 Agent 抓取"
gh label create triage          -R oasislin/Poly2-Phase1 --color "FBCA04" --description "待分诊"
gh label create blocked         -R oasislin/Poly2-Phase1 --color "B60205" --description "被阻塞"
```

### 发布 spec issue
```bash
gh issue create -R oasislin/Poly2-Phase1 --title "Spec: <标题>" --body "<spec 正文>"
# 返回编号，如 #12；随后打标签：
gh issue edit 12 -R oasislin/Poly2-Phase1 --add-label ready-for-agent
```

### 发布 ticket issues（按依赖顺序，无阻塞的在前）
```bash
# 无依赖的 ticket：
gh issue create -R oasislin/Poly2-Phase1 --title "Ticket 01: <标题>" --body "<正文> Blocked by: None — can start immediately"
gh issue edit <编号> -R oasislin/Poly2-Phase1 --add-label ready-for-agent

# 有依赖的 ticket：
gh issue create -R oasislin/Poly2-Phase1 --title "Ticket 02: <标题>" --body "<正文> Blocked by: #<前置编号>"
gh issue edit <编号> -R oasislin/Poly2-Phase1 --add-label ready-for-agent
```

### 查看待办（frontier）
```bash
gh issue list -R oasislin/Poly2-Phase1 --label ready-for-agent
```

### Ticket 正文模板（to-tickets）
```markdown
## Parent
（若源自某 spec issue，注明其编号）

## What to build
从用户视角的端到端行为，而非分层实现清单。

## Acceptance criteria
- [ ] 标准 1
- [ ] 标准 2

## Blocked by
- 前置 ticket 引用，或 "None — can start immediately"
```
