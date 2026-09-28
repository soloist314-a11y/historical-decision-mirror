# 输出约定

deep 输出依次包含：现实决策结构、关键变量、历史主案例、反案例或缺口、跨案例机制、条件式决策树、显性与隐性代价及未来选择空间、历史无法回答的变量。

每个历史案例包含 case_id、T0 信息、可选方案、实际选择、分期结果、代价、来源锚点、当前核验状态与类比失效条件。不得把推演填入史实字段。

quick 可以合并段落，但不能删除证据状态和关键差异。audit 先列用户命题，再列支持、反证、错误类比风险和仍需核验的现实事实。

结构化审计记录为：

```json
{
  "decision": "allow",
  "structural_score": 0.8,
  "role_match": true,
  "mechanism_match": true,
  "critical_differences": ["示例差异，须按实际填写"],
  "evidence_status": "verified",
  "hindsight_used_as_t0": false,
  "modern_rights_overtransfer": false,
  "invented_user_values": false,
  "contrast_status": "verified"
}
```

decision 可为 allow、abstain、request_information。structural_score 是上游评分，必须可追溯，不能凭此示例复制数值。evidence_status 可为 verified、unavailable、mismatch、unverified；contrast_status 可为 verified、missing、unverified。缺失对照只允许保留有限假设，不作强结论。

runtime.py 的大写状态是候选包内部状态，不能直接复制到最终审计字段。REQUIRES_CURRENT_SOURCE_READ 转为 unverified；NOT_APPLICABLE 表示未进入案例核验，最终如需审计记录仍用 unverified 并解释未选案例；MISSING 转为 missing。CANDIDATES_REQUIRE_REVIEW 不能转为 verified，须实际回读和审查后才能升级。

口语输入可能未触发旧检索器的词面条件。宿主在 audit 模式仍须回应用户命题、指出逻辑问题和未知事实，不得只复述“无法召回”。若重新表达结构以检索，应记录原输入、规范化结构与依据，不注入预期人物；缺少现实事实时停止历史案例类比。
