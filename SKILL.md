---
name: historical-decision-mirror
description: 使用有来源的历史决策节点分析现实选择、组织权力、资源依赖与承诺，或审计错误类比。用户调用“历史推演”“历史抉择镜鉴”“用史记分析”“用资治通鉴推演”时进入；普通历史知识问答不需要此技能。
metadata:
  version: 1.0.0
---

# 历史抉择镜鉴

唯一用户入口：`/历史推演 [quick|deep|audit] <现实问题>`。省略模式默认 deep。自然语言请求进入同一流程，不增加另一套推演规则。

运行只加载当前包的 110 个 ACTIVE 节点；3 个 DEPRECATED 节点只保留谱系。数据源于 v0.9，来源与逐文件迁移映射见 `data/migration_manifest.json`。旧证据标志不是本次来源读取证明。

## 调用

用 `python3 scripts/runtime.py '<用户原话>'` 进入统一候选检索流程。脚本只识别显式入口和常见自然语言触发，不替代宿主的语义理解；对明确的自然语言历史决策请求，宿主可将问题等价地送入 `/历史推演 deep <原问题>`。保留原话，不改写成预期历史答案。读 [决策模型](references/decision-model.md)，将用户事实、假设、未知和价值判断分开。未知字段留空，不猜测价值排序。

只读取 `data/decision_nodes.jsonl` 与 `data/contrast_pairs.jsonl`。按角色、权力、资源、依赖、退出与时间条件建立结构映射，不能仅因人物名或结局相似而认定可迁移。脚本仅返回候选，始终要求宿主完成六道准入检查；不把候选排名直接当成最终推演。

输出前必须读 [类比边界](references/analogy-policy.md) 和 [证据规则](references/evidence-policy.md)。结构不足则请求关键事实或停止历史类比；原文不可读取则标为未核验。反案例缺失时明确披露，不补造。

按命中机制补充读取：退出、代理、资源、继任、联盟等读 [基础机制](references/mechanisms-foundation.md)；授权撤权、合作锁定、沉没成本、人才读 [治理机制](references/mechanisms-governance.md)；交接、承诺和贡献变化读 [连续性机制](references/mechanisms-continuity.md)。案例重建读 [案例模型](references/case-model.md)，对照选择读 [反案例规则](references/contrast-policy.md)。不用一次加载全部案例和参考。

- quick：压缩篇幅，保留结构映射、证据状态、一个可靠对照或缺失说明、关键未知。
- deep：完整重建决策时点信息、选项、主案例与对照、条件树和显隐性代价。
- audit：以用户已有判断为待检验命题，检查反证、事后偏见、制度越界和未声明的价值假设，不代替用户另作决定。

按 [输出约定](references/output-schema.md) 呈现。结构化审计记录可用 `python3 scripts/check_analogy.py record.json` 检查；该检查不能证明自由文本推演正确。

## 维护

修改数据或算法后运行 `python3 scripts/run_evals.py`。检查说明见 [统一评测](evals/README.md)，版本规则见 [版本治理](references/versioning-policy.md)，来源见 [源注册表](references/source-registry.md)。只有完整迁移并完成统一测试，才能提升 LATEST 与发布正式标签。
