# Case Migration Notes v0.9

## Runtime relationship

v0.9 **在正式运行中取代 v0.8**。v0.8 及更早版本只作为归档快照、回归基线和迁移审计保留。详见 `versioning_policy_v0.9.md`。

## Data changes

- v0.8: 102 ACTIVE + 3 DEPRECATED。
- v0.9: 110 ACTIVE + 3 DEPRECATED。
- 新增 ACTIVE: HD-106—HD-113，共8条。
- 新增反案例关系：15组。
- 所有当前记录统一写入 `version=0.9`；旧快照保留此前状态。

## New direct samples

- HD-106：单一最高领导赞助/保护的更替风险。
- HD-107：阶段完成后依赖下降与权力/利益重定价。
- HD-108：最近关键贡献变化后的名位重排。
- HD-109：危机功劳与下一阶段岗位适配分离。
- HD-110/111：继任者重新授权的设计与执行。
- HD-112：前任保护性承诺在继任结构下未持续。
- HD-113：组织明确承接代理人承诺。

## Source-quality corrections

- HD-112 拆成两条《资治通鉴》源引用：卷199记录太宗临终嘱托，卷200记录高宗时期处置，避免把跨十年的材料伪装成一个原文段落。
- HD-110/111 保留用户 EPUB 中“李世”姓名丢字问题，证据状态为 `SOURCE_VERIFIED_TEXT_NAME_OCR_ISSUE`；不静默修补成底本没有显示的文字。

## Retrieval changes

新增高精度特征：
- `sponsor_dependency_risk`
- `phase_change_reallocation`
- `dynamic_contribution_regrading`
- `phase_role_fit_reallocation`
- `successor_mandate_reset`
- `predecessor_commitment_continuity`
- `organizational_ratification_of_agent_commitment`

定向召回：10/10 Top-1；旧机制回归：18/18。
