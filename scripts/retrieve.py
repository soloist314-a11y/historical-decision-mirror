#!/usr/bin/env python3
import json, pathlib, argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
BASE=pathlib.Path(__file__).resolve().parent
CASES=BASE.parent/'data/decision_nodes.jsonl'
EVIDENCE_MULT={'SOURCE_VERIFIED':1.0,'SOURCE_VERIFIED_TEXT_NAME_OCR_ISSUE':0.9,'SEED_UNVERIFIED':0.72}

FEATURE_WEIGHTS={
 'platform_transition':3.4,'authority_gap':3.3,'portable_capability':3.0,'dependency_lockin':2.9,
 'autonomy_exit':2.8,'option_value_core':2.6,'resource_asymmetry':2.2,'short_long_tradeoff':1.7,'timing_window':1.5,
 'reversible_exchange':4.0,'commitment_enforceability':3.4,'bargaining_power':2.8,
 'dual_authority':4.2,'succession_governance':3.5,'power_retention_after_handoff':3.6,
 'reward_role_separation':4.2,'role_fit':3.1,'incentive_design':2.6,
 'selection_evidence':2.4,'conflict_of_interest':2.0,
 'strategic_concession':3.8,'common_threat_coalition':3.4,'post_failure_recovery':3.6,'wait_for_trigger':3.2,'delegation_control':3.8,
 'internal_conflict_deescalation':4.0,'powerful_stakeholder_rule_enforcement':4.0,
 'multi_source_warning':4.2,'mandate_clarity':4.2,'information_verification':4.2,
 'policy_correction':4.0,'de_facto_power_recognition':4.1,'dissent_safety':4.0,
 'coercion_signal':4.0,'fast_legitimization_window':4.0,'neutral_boundary':3.8,
 'principal_agent_drift':4.5,'control_resource_concentration':4.35,'crisis_resource_concentration':4.0,
 'succession_ambiguity':4.5,'succession_capture':4.25,'information_gatekeeper_capture':4.5,'deception_attack':4.25,
 'reward_allocation':4.3,'alliance_fracture':4.3,'non_self_enforcing_commitment':4.4,'crisis_core_preservation':4.3,
 'stop_condition_verification':4.4,'exit_on_strategic_misalignment':4.35,'specific_consideration_commitment':4.35,
 'agent_betrayal_under_conflict':4.6,'second_power_center':4.5,'status_preserving_deconcentration':4.4,
 'revocation_cost':4.5,'partner_asset_lockin':4.6,'asset_recovery_after_lockin':4.6,
 'sunk_cost_escalation':4.7,'project_stop_loss':4.7,'leadership_postmortem_stop':4.6,
 'professional_dissent':4.6,'expert_refusal_cost':4.5,'critical_talent_retention':4.6,
 'failure_accountability_separation':4.5,'talent_underrecognition_loss':4.5,'external_talent_fast_regrade':4.4,
 'relationship_confidentiality_leak':4.9,'insider_external_influence':5.0,'external_force_internal_governance':4.9,
 'strategic_chokepoint':4.7,'leadership_transition_continuity':4.6,'authority_commitment_gap':4.9,
 'fairness_signal':4.7,'voluntary_handoff':4.6,'post_success_exit':4.7,'late_exit_after_trust_collapse':4.8,
 'failure_attribution_retest':4.8,'silence_as_false_consensus':4.9,'frontline_stop_resistance':5.0,
 'bad_news_suppression':4.9,'organizational_reputation_commitment':5.0,
 'sponsor_dependency_risk':5.2,'phase_change_reallocation':5.2,'dynamic_contribution_regrading':5.3,
 'phase_role_fit_reallocation':5.25,'successor_mandate_reset':5.35,'predecessor_commitment_continuity':5.35,
 'organizational_ratification_of_agent_commitment':5.4,
}

def flat(r):
 s=[r.get('case_title',''),r.get('decision_question',''),r.get('actual_choice',''),r.get('actor_role',''),
    ' '.join(r.get('tags',[])),' '.join(r.get('structural_match_keys',[])),r.get('reversibility',''),r.get('exit_options','')]
 s += list(map(str,r.get('structures',{}).values()))
 s += list(map(str,r.get('costs',{}).get('structural_inferred',[])))
 s += list(map(str,r.get('known_information_at_time',[])))
 return ' '.join(s)

def tags(r): return set(r.get('tags',[]))
def has_any(text, vals): return any(v in text for v in vals)
def has_tag(r, vals): return bool(tags(r)&set(vals))

def case_features(r):
 t=flat(r); f=set()
 if has_tag(r,['平台选择','合作选择','独立与并入','跳槽','弱平台','独立空间','平台风险']) or has_any(t,['旧平台','新平台','强平台','平台不给','亡楚归汉','离魏入秦']): f.add('platform_transition')
 if has_tag(r,['授权','控制权','名位与实权','正式权力','授权边界','战略自主权','责任权利匹配']) or has_any(t,['不给核心权力','不给机会','不授核心权力','核心授权','兵权才能兑现','正式控制权']): f.add('authority_gap')
 if has_tag(r,['人才流动','边际价值','跳槽']) or has_any(t,['能力可迁移','跨组织迁移','自主能力可迁移','军事战略能力可跨组织迁移','改革与强国方案','个人能力']): f.add('portable_capability')
 if has_any(t,['依赖','路径依赖','沉没成本','控制范围','全部议价权','单点']) or has_tag(r,['路径依赖','依赖']): f.add('dependency_lockin')
 if has_tag(r,['退出','退出止损','独立空间','独立与并入','平台选择','跳槽']) or has_any(t,['主动退出','离开','亡楚归汉','离魏','独立身份']): f.add('autonomy_exit')
 if has_tag(r,['选择权','可逆性','退路','退出条件','议价权']) or has_any(t,['选择权','退出窗口','保留退路','可逆','议价权','回旋']): f.add('option_value_core')
 if has_tag(r,['平台资源','制度资源','边际价值','资源整合']) or has_any(t,['稀缺资源','核心资产','资源强','资源弱','个人能力','掌握关键']): f.add('resource_asymmetry')
 if has_tag(r,['长期主义','长期战略','短期KPI','路径依赖']) or has_any(t,['短期','长期','边际收益下降','未来选择权']): f.add('short_long_tradeoff')
 if has_tag(r,['机会窗口','时机','先发窗口']) or has_any(t,['窗口','时机','阶段性机会']): f.add('timing_window')
 # Transaction / commitment cluster
 if has_tag(r,['可逆性','交付条件','退出条件']) or has_any(t,['核心资产','交付','撤回','归璧','先交出']): f.add('reversible_exchange')
 if has_tag(r,['可信承诺','承诺','信用']) or has_any(t,['承诺不可执行','强制执行','违约','履约','安全承诺','可信承诺']): f.add('commitment_enforceability')
 if has_tag(r,['议价权','谈判']) or has_any(t,['议价权','谈判能力','负在','负秦']): f.add('bargaining_power')
 # Succession/governance cluster
 if has_tag(r,['双重权威','创始人退出']) or has_any(t,['双重权威','形式交班','交班后仍','原领导仍握','主父']): f.add('dual_authority')
 if has_tag(r,['继承','接班','继承冲突','治理边界']) or has_any(t,['继承人','接班','传位','交班']): f.add('succession_governance')
 if has_tag(r,['创始人退出','双重权威']) or has_any(t,['保留实权','仍握高威望','重新分配权力','传位后']): f.add('power_retention_after_handoff')
 # Reward / role design cluster
 if has_tag(r,['功劳奖励','职位设计']) or has_any(t,['官以任能','爵以酬功','奖励与岗位混同','功劳很大但未必适合']): f.add('reward_role_separation')
 if has_tag(r,['用人','识人','关键岗位','能力证据']) or has_any(t,['适合更高管理','岗位能力','任能','候选人']): f.add('role_fit')
 if has_tag(r,['激励','激励设计','功劳奖励']) or has_any(t,['激励','奖励','奖赏']): f.add('incentive_design')
 if has_tag(r,['能力证据','行为证据','第三方验证']) or has_any(t,['绩效证据','交叉验证','能力可以在任务中验证']): f.add('selection_evidence')
 if has_tag(r,['利益冲突','举亲']) or has_any(t,['关系密切','裙带']): f.add('conflict_of_interest')
 if has_tag(r,['战略让步','强者加码','诱导过度扩张']) or has_any(t,['暂让','索地','购买时间']): f.add('strategic_concession')
 if has_tag(r,['联盟形成','共同威胁','择交']) or has_any(t,['共同威胁','相亲','择交','联盟成熟']): f.add('common_threat_coalition')
 if has_tag(r,['失败后恢复','战后恢复','止损']) or has_any(t,['新败','重大失败后','士气沮丧']): f.add('post_failure_recovery')
 if has_tag(r,['触发条件','等待']) or has_any(t,['条件成熟','可以战矣','等待对手','六十余日']): f.add('wait_for_trigger')
 if has_tag(r,['授权风险','代理执行']) or has_any(t,['临时交出军务','代理将领','执行层改写']): f.add('delegation_control')
 if has_tag(r,['内部冲突','冲突降级']) or has_any(t,['两虎共鬥','共同外部威胁','个人名位冲突']): f.add('internal_conflict_deescalation')
 if has_tag(r,['规则执行','权贵冲突']) or has_any(t,['强势利益相关者要求例外','依法处理','规则一致性']): f.add('powerful_stakeholder_rule_enforcement')
 if has_tag(r,['多源预警','关键任命']) or has_any(t,['多个独立来源','独立反证','赵括']): f.add('multi_source_warning')
 if has_tag(r,['授权边界','战略自主权','责任权利匹配']) or has_any(t,['臣如前','接受任命的明确条件','战略自主']): f.add('mandate_clarity')
 if has_tag(r,['信息核验','单点信息源']) or has_any(t,['单一代理报告','交叉核验','信息通道被污染']): f.add('information_verification')
 if has_tag(r,['政策纠偏','过度泛化']) or has_any(t,['一刀切','撤销逐客','单一负面事件']): f.add('policy_correction')
 if has_tag(r,['既成权力','现实约束']) or has_any(t,['无法强制约束','名义权力与实际','既成事实']): f.add('de_facto_power_recognition')
 if has_tag(r,['直谏','心理安全','坏消息']) or has_any(t,['有人敢说真话','真实信息通道','尖锐反馈']): f.add('dissent_safety')
 if has_tag(r,['胁迫','不奖励威胁']) or has_any(t,['威胁可改变','奸谋得成','胁迫可换取']): f.add('coercion_signal')
 if has_tag(r,['窗口期','快速合法化','正式身份']) or has_any(t,['正式认可','窗口高度短暂','机会一失']): f.add('fast_legitimization_window')
 if has_tag(r,['中立','边界一致性','不选边']) or has_any(t,['对称中立','互相要求你站队','应之亦然']): f.add('neutral_boundary')
 # v0.6 mechanism-gap features
 if has_tag(r,['代理人失控','远程治理','本地化']) or has_any(t,['逐渐不听命','不能制','囚其使者','脱离控制','代理人本地化']): f.add('principal_agent_drift')
 if has_tag(r,['控制资源集中','军财一体','区域权力']) or has_any(t,['强兵富资','地盘、军权和财富','同时掌握地盘','资源和权限集中']): f.add('control_resource_concentration')
 if has_tag(r,['组织危机','资源集中','全员动员']) and has_any(t,['一次性下注','最终反攻','资源极限','集中投入']): f.add('crisis_resource_concentration')
 if has_tag(r,['继承模糊','派系竞争','多中心权力']) or has_any(t,['未显言','多个候选人','矫绍遗命','继任规则不清']): f.add('succession_ambiguity')
 if has_tag(r,['继任设计','沙丘之谋','矫诏']) or has_any(t,['改嗣','继承文件','遗诏','少数人控制继任']): f.add('succession_capture')
 if has_tag(r,['信息垄断','坏消息屏蔽']) or has_any(t,['不坐朝廷见大臣','事皆决于赵高','单一信息入口','信息过滤']): f.add('information_gatekeeper_capture')
 if has_tag(r,['信息欺骗','反间','诈降']) or has_any(t,['反间','制造错误判断','虚假信息','诈为','诈降']): f.add('deception_attack')
 if has_tag(r,['利益分配','奖励公平','相对公平','明确对价']) or has_any(t,['分配利益','功人','功狗','未有分地','一王一侯']): f.add('reward_allocation')
 if has_tag(r,['联盟解体','信任破裂']) or has_any(t,['转向田荣','遂有却','联盟裂解','盟友转投']): f.add('alliance_fracture')
 if has_tag(r,['非自执行协议','履约机制']) or has_any(t,['缺少持续约束','没有第三方执行','协议后立即','自执行承诺']): f.add('non_self_enforcing_commitment')
 if has_tag(r,['核心保全','业务连续性','危机撤离']) or has_any(t,['保全中枢','不可替代核心','业务连续性','一网打尽']): f.add('crisis_core_preservation')
 if has_tag(r,['错误止损','停止条件','不可逆让步']) or has_any(t,['真实停止条件','东帝','满足诉求','不买到停战']): f.add('stop_condition_verification')
 if has_tag(r,['退出止损','价值冲突','平台风险','身份切割']) or has_any(t,['有异志','退出成本会','提前切割','低冲突退出']): f.add('exit_on_strategic_misalignment')
 if has_tag(r,['明确对价','联盟履约']) or (has_any(t,['未有分地','具体分配','明确分地']) and has_any(t,['会师','履约','进兵'])): f.add('specific_consideration_commitment')

 # v0.7 second-order mechanism features
 if has_tag(r,['代理背叛']) and (has_tag(r,['利益冲突','人身安全','替代联盟']) or has_any(t,['不自安','私怨','内应','转为'])): f.add('agent_betrayal_under_conflict')
 if has_tag(r,['第二权力中心','多中心权力']): f.add('second_power_center')
 if has_tag(r,['权力去集中','保留荣誉']): f.add('status_preserving_deconcentration')
 if has_tag(r,['撤权失败','资产回收']): f.add('revocation_cost')
 if has_tag(r,['合作方锁定','关键资产授权']): f.add('partner_asset_lockin')
 if has_tag(r,['资产回收']) and (has_tag(r,['合作方锁定','路径依赖']) or has_any(t,['索还','回收','逐之'])): f.add('asset_recovery_after_lockin')
 if has_tag(r,['沉没成本升级','失败后加码']): f.add('sunk_cost_escalation')
 if has_tag(r,['项目停损','转向闸门']): f.add('project_stop_loss')
 if has_tag(r,['领导者认错','公开复盘','战略纠偏']): f.add('leadership_postmortem_stop')
 if has_tag(r,['错误指令','专业异议']): f.add('professional_dissent')
 if has_tag(r,['专业拒绝','拒命代价']): f.add('expert_refusal_cost')
 if has_tag(r,['关键人才留退','人才保留']): f.add('critical_talent_retention')
 if has_tag(r,['失败问责','领导责任','保留团队']): f.add('failure_accountability_separation')
 if has_tag(r,['人才流失','认知锁定']) or (has_any(t,['反复推荐','实战证明','救其性命']) and has_any(t,['不用','待如初','流失'])): f.add('talent_underrecognition_loss')
 if has_tag(r,['人才引入','快速定级','空降人才']) or (has_any(t,['外部人才','转入']) and has_any(t,['多源推荐','快速定级','同于旧臣'])): f.add('external_talent_fast_regrade')
 # v0.8 third-order mechanisms
 if has_tag(r,['私人关系冲突','信息泄漏','关键知情人']): f.add('relationship_confidentiality_leak')
 if has_tag(r,['内外串联']) and (has_tag(r,['利益冲突','反间','信息欺骗']) or has_any(t,['外部','宠臣','受贿','私见'])): f.add('insider_external_influence')
 if has_tag(r,['授人以柄','外部合作方']) and has_any(t,['外部','独立强制','强军','引兵','进入核心治理']): f.add('external_force_internal_governance')
 if has_tag(r,['关键资源单点','先发窗口','制度资源']) or has_any(t,['关键入口资源','先占优势','网络效应','若不时定']): f.add('strategic_chokepoint')
 if has_tag(r,['领导更换','关键人才连续性','交接治理']): f.add('leadership_transition_continuity')
 if has_tag(r,['组织承诺','权力更替']) and has_any(t,['履约资源','实际分配权','名义权威','控制权变化']): f.add('authority_commitment_gap')
 if has_tag(r,['公平信号','组织信用']) and has_any(t,['可观察','人人自坚','群臣皆喜','私人偏好']): f.add('fairness_signal')
 if has_tag(r,['主动逊位','权责分层']) or (has_tag(r,['成功后交权']) and has_any(t,['保留荣誉','顾问'])): f.add('voluntary_handoff')
 if has_tag(r,['主动退出','成功后交权']) and has_any(t,['足矣','退出窗口','淡出']): f.add('post_success_exit')
 if has_tag(r,['退出窗口','信任崩溃']) and has_any(t,['不信','不肯听','请归','角色有效性']): f.add('late_exit_after_trust_collapse')
 if has_tag(r,['归因偏差','项目复盘']) and has_any(t,['局部归因','战略假设','修复局部']): f.add('failure_attribution_retest')
 if has_tag(r,['沉默治理','心理安全']) and has_any(t,['无敢言','沉默','共识']): f.add('silence_as_false_consensus')
 if has_tag(r,['一线沉没成本','停损执行','完成偏差']): f.add('frontline_stop_resistance')
 if has_tag(r,['坏消息抑制','指标失真']) or has_any(t,['不以实','报表越来越','报喜不报忧','系统性瞒报','信息过滤而不是问题改善']): f.add('bad_news_suppression')
 if has_tag(r,['组织信用','信誉成本']) and has_any(t,['公开承诺','依赖投入','国家声听','食言']): f.add('organizational_reputation_commitment')
 # v0.9 leadership / contribution / commitment continuity features
 if has_tag(r,['赞助人依赖','个人保护']) and has_tag(r,['领导更替','退出窗口']): f.add('sponsor_dependency_risk')
 if has_tag(r,['阶段变化','权力重定价']) and (has_tag(r,['依赖下降','利益再分配']) or has_any(t,['阶段结束','依赖结构变化','重新配置'])): f.add('phase_change_reallocation')
 if has_tag(r,['动态贡献','功劳排序']) or (has_any(t,['最近一次关键危机','贡献更大']) and has_any(t,['名位','重新排序','让'])): f.add('dynamic_contribution_regrading')
 if has_tag(r,['阶段角色适配','能力模型变化']) or (has_any(t,['常态治理','岗位并不匹配','能不如']) and has_tag(r,['功劳与岗位分离'])): f.add('phase_role_fit_reallocation')
 if has_tag(r,['继任关系设计','重新授权','继任者授权']) or (has_any(t,['继任者','新君']) and has_any(t,['重新任用','重新授权','建立自己的授权'])): f.add('successor_mandate_reset')
 if has_tag(r,['前任承诺连续性','承诺承接']): f.add('predecessor_commitment_continuity')
 if has_tag(r,['组织承接个人承诺']): f.add('organizational_ratification_of_agent_commitment')
 return f

def query_features(q):
 f=set()
 # Platform is triggered only by explicit platform/migration language, not generic 'organization'.
 if has_any(q,['平台','换平台','加入','转投','离开现有','依附']): f.add('platform_transition')
 if has_any(q,['控制权','实权','管理权','自主权','授权不足','不给权']): f.add('authority_gap')
 if (has_any(q,['客户','项目获取','资源整合','可迁移能力','个人能力']) or ('能力' in q and has_any(q,['换平台','加入','转投','离开现有','自建体系']))) and has_any(q,['自建','迁移','平台','独立','换平台','转投']): f.add('portable_capability')
 if has_any(q,['依赖','沉没成本','锁定','路径依赖']): f.add('dependency_lockin')
 if has_any(q,['自建','独立体系','离开现有','转投']) or ('退出' in q and has_any(q,['平台','现有体系','合作关系','组织关系','合伙关系'])): f.add('autonomy_exit')
 if has_any(q,['选择权','可逆','退路','退出能力','议价权']): f.add('option_value_core')
 if has_any(q,['资源','客户','项目','履约体系','核心资产']): f.add('resource_asymmetry')
 if '短期' in q and has_any(q,['长期','未来']): f.add('short_long_tradeoff')
 if has_any(q,['窗口','时机','先发','延迟']): f.add('timing_window')
 # Precise clusters
 if has_any(q,['先交付','交付后','可逆交付','核心资产']) and has_any(q,['不可替代','失去','退出','追回','议价权','承诺']): f.add('reversible_exchange')
 if has_any(q,['承诺','违约','强制执行','履约保证','担保']) : f.add('commitment_enforceability')
 if has_any(q,['议价权','谈判权','谈判能力']): f.add('bargaining_power')
 if has_any(q,['双重权威','原负责人仍','原领导仍','名义交班','交班后仍','创始人仍']): f.add('dual_authority')
 if has_any(q,['继承人','接班','交班','继承','传位']): f.add('succession_governance')
 if has_any(q,['保留实权','仍掌握','重新分配权力','实际影响力']) and has_any(q,['交班','继承','接班','原负责人','原领导']): f.add('power_retention_after_handoff')
 if has_any(q,['功劳','功臣']) and has_any(q,['岗位','管理权','授权','职位','奖励','爵赏']): f.add('reward_role_separation')
 if has_any(q,['适合','岗位能力','管理岗位','任职能力','胜任']): f.add('role_fit')
 if has_any(q,['奖励','激励','奖赏','爵赏']): f.add('incentive_design')
 if has_any(q,['能力证据','绩效证据','行为证据','第三方验证']): f.add('selection_evidence')
 if has_any(q,['利益冲突','裙带','举亲','亲属']): f.add('conflict_of_interest')
 if has_any(q,['暂时让步','战略让步','先让一步','对方不断加码','强势索取','购买时间']): f.add('strategic_concession')
 if has_any(q,['共同威胁','第三方结盟','形成联盟','共同敌人','联合其他受压方']): f.add('common_threat_coalition')
 if has_any(q,['刚失败','重大失败','新败','士气低','失败后']): f.add('post_failure_recovery')
 if has_any(q,['等待条件','等到','触发条件','何时再行动','择机','对方疲惫']): f.add('wait_for_trigger')
 if has_any(q,['代理人','授权给下属','临时交权','我不在现场','下属违令','执行失控']): f.add('delegation_control')
 if has_any(q,['内部冲突','同级冲突','内斗','核心成员冲突']) and has_any(q,['共同目标','共同敌人','外部威胁','大局']): f.add('internal_conflict_deescalation')
 if has_any(q,['权贵','强势股东','大客户','老板亲属','特殊关系']) and has_any(q,['规则','例外','违规','制度']): f.add('powerful_stakeholder_rule_enforcement')
 if has_any(q,['多个独立意见','多人预警','多源预警','独立反证']) and has_any(q,['任命','选人','换人','候选人']): f.add('multi_source_warning')
 if has_any(q,['复任','接任','接受职位','负责结果']) and has_any(q,['自主权','授权边界','策略权','决策权','权责']): f.add('mandate_clarity')
 if has_any(q,['单一信息源','单一报告','被收买','利益冲突的信息','交叉核验','复核信息']): f.add('information_verification')
 if has_any(q,['一刀切','单一事件','个案'] ) and has_any(q,['政策','规则','全面禁止','全面清退','纠偏']): f.add('policy_correction')
 if has_any(q,['名义上有权','实际控制','既成事实','无法强制','事实控制']) and has_any(q,['承认','授权','让步','合作']): f.add('de_facto_power_recognition')
 if has_any(q,['直言','坏消息','尖锐反馈','不敢说真话','心理安全','报喜不报忧']): f.add('dissent_safety')
 if has_any(q,['威胁','恐吓','攻击','胁迫']) and has_any(q,['撤人','罢免','换人','让步','奖励威胁']): f.add('coercion_signal')
 if has_any(q,['窗口期','转投','归附','刚加入','脱离旧体系']) and has_any(q,['正式身份','快速确认','立即任命','尽快确认','合法化']): f.add('fast_legitimization_window')
 if has_any(q,['双方都要我站队','两边都拉我','保持中立','对称中立','不选边','边界一致']): f.add('neutral_boundary')
 # v0.6 mechanism-gap features: use precise conjunctions to avoid generic matches
 if has_any(q,['代理人','区域负责人','分公司负责人','本地负责人']) and has_any(q,['区域','本地','分公司','几年','长期','逐渐']) and has_any(q,['不听','脱离控制','失控','本地化','自立','逐渐不受控']): f.add('principal_agent_drift')
 if has_any(q,['同时掌握','集中掌握','一个人掌握','一人掌握']) and has_any(q,['地盘','区域','现金流','财务','人事','客户','团队','军权','核心资源']): f.add('control_resource_concentration')
 if has_any(q,['资源已接近极限','只剩一次','最后窗口','孤注一掷','一次集中行动']) and has_any(q,['全员','集中资源','动员','危机']): f.add('crisis_resource_concentration')
 if has_any(q,['继承人未定','继任人未定','继任规则不清','接班人不明确','多个候选人']) and has_any(q,['各自掌握','各带团队','各有资源','派系','下注','观察']): f.add('succession_ambiguity')
 if has_any(q,['继任','继承','接班']) and has_any(q,['控制文件','控制遗嘱','控制遗诏','少数人控制','改规则','矫诏','被清算']): f.add('succession_capture')
 if has_any(q,['不直接见','过滤所有信息','唯一入口','单一幕僚','一个人过滤','信息入口']) and has_any(q,['核心管理层','大臣','团队','议程','决策']): f.add('information_gatekeeper_capture')
 if has_any(q,['假消息','虚假信息','离间','反间','误导信息','造谣']) and has_any(q,['核心副手','核心团队','竞争对手','猜忌','信任']): f.add('deception_attack')
 if has_any(q,['怎么分','利益分配','奖金怎么分','收益怎么分','分配公平','功劳分配','相对公平']) or (has_any(q,['前台','后台','供应链','平台']) and has_any(q,['贡献','奖金','奖励'])): f.add('reward_allocation')
 if has_any(q,['盟友翻脸','联盟解体','合作伙伴内讧','转投竞争','长期合伙人']) and has_any(q,['分配','让权','资源','争吵','翻脸','转投']): f.add('alliance_fracture')
 if has_any(q,['协议签了','承诺签了','合同约定','和约']) and has_any(q,['没有担保','没有保证金','没有第三方','没有司法','无法强制','激励变了','不履约']): f.add('non_self_enforcing_commitment')
 if has_any(q,['管理中枢','核心团队','不可替代核心','业务连续性','灾备','一锅端']) and has_any(q,['危机','摧毁','保全','备份','撤离']): f.add('crisis_core_preservation')
 if has_any(q,['只要我','只要你','提出条件','停止条件']) and has_any(q,['停战','停止攻击','不再攻击','停止冲突','就停']) and has_any(q,['确认','真实','是否真','不能确认','验证']): f.add('stop_condition_verification')
 if has_any(q,['平台','组织','公司']) and has_any(q,['合规风险','战略风险','重大风险','价值冲突','方向失控']) and has_any(q,['还没爆雷','尚未公开','退出成本','提前退出','是否退出']): f.add('exit_on_strategic_misalignment')
 if has_any(q,['口头答应','答应支持','约好支持','承诺支持']) and has_any(q,['收益边界','对价','分配没写清','利益没写清','兑现条件']): f.add('specific_consideration_commitment')

 # v0.7 second-order mechanisms: require conjunctions to reduce semantic overreach
 if has_any(q,['核心代理人','关键代理人','核心下属','重要代理人']) and has_any(q,['威胁','不公平','利益冲突','安全风险','关系破裂']) and has_any(q,['替代联盟','竞争对手','另一方','转投','背叛']): f.add('agent_betrayal_under_conflict')
 if has_any(q,['第二权力中心','事实独立','独立山头','另一个中心','尾大不掉']) or (has_any(q,['地方负责人','事业部负责人','功臣','亲信']) and has_any(q,['权力过大','事实控制','高度独立','自成体系'])): f.add('second_power_center')
 if has_any(q,['撤权','降权','收回运营权','减少权限']) and has_any(q,['保留荣誉','保留头衔','保留待遇','保留股权','不否定功劳']): f.add('status_preserving_deconcentration')
 if has_any(q,['撤权','收回权限','换负责人','解除控制','收回资产']) and has_any(q,['成本很高','会冲突','会反弹','实际撤不动','引发大冲突','收不回来']): f.add('revocation_cost')
 if has_any(q,['合作方','伙伴','盟友']) and has_any(q,['关键资产','客户','数据','渠道','区域','团队','控制资源']) and has_any(q,['交给','借给','授权','共同对抗','共同敌人']) and has_any(q,['锁定','以后难收回','收不回来','依赖']): f.add('partner_asset_lockin')
 if has_any(q,['合作方','伙伴','盟友']) and has_any(q,['已经经营','多年','建立团队','掌握客户','形成体系']) and has_any(q,['回收','索回','收回资产','撤回授权','收回授权']) and has_any(q,['拖延','拒绝','冲突','收不回来']): f.add('asset_recovery_after_lockin')
 if has_any(q,['重大失败','第一次失败','已经失败','项目失败']) and has_any(q,['已经投入','沉没成本','不能认输','怕丢面子','声誉','不能承认失败']) and has_any(q,['继续加码','追加投入','加倍投入','再投','扩大投入']): f.add('sunk_cost_escalation')
 if has_any(q,['项目','行动','投资','业务']) and has_any(q,['拖了','久攻不下','长期不达标','没达里程碑','持续失败','预算超支']) and has_any(q,['停损','止损','转向','暂停','何时停','继续还是停']): f.add('project_stop_loss')
 if has_any(q,['公开承认','承认判断错','公开复盘','模型错了','此前判断错误']) and has_any(q,['停止新投入','停止扩张','不再追加','资源转回','回到基本盘']): f.add('leadership_postmortem_stop')
 if has_any(q,['老板','上级','领导','强势上级']) and has_any(q,['要求改变','临时要求','错误指令','要求执行','急令']) and has_any(q,['专业判断','专业负责人','已批准方案','证据表明','替代方案']) and has_any(q,['反对','异议','坚持','不执行','不同意']): f.add('professional_dissent')
 if has_any(q,['强势老板','强势上级','最高领导']) and has_any(q,['专家','专业负责人','关键人才']) and has_any(q,['高概率失败','不可行','判断会失败']) and has_any(q,['拒绝','不接','不执行']) and has_any(q,['职业代价','被撤职','严重代价','得罪']): f.add('expert_refusal_cost')
 if has_any(q,['关键人才','核心人才','稀缺人才']) and has_any(q,['准备走','离职','要走','流失']) and has_any(q,['长期观察','实绩','不可替代','稀缺能力','重新定级','破格']): f.add('critical_talent_retention')
 if has_any(q,['项目失败','重大失败']) and has_any(q,['领导战略判断','上层判断','领导决策']) and has_any(q,['执行团队','项目团队','下属']) and has_any(q,['开掉','问责','保留','责任区分']): f.add('failure_accountability_separation')
 if has_any(q,['员工','人才','关键成员']) and has_any(q,['实战证明','绩效已证明','多个核心人推荐','反复推荐']) and has_any(q,['旧标签','一直不用','低估','不提拔','不给机会']) and has_any(q,['离职','要走','流失']): f.add('talent_underrecognition_loss')
 if has_any(q,['外部人才','外来人才','新加入','转入人才','空降']) and has_any(q,['已验证实绩','实绩','多名推荐','多源推荐']) and has_any(q,['快速定级','高岗位','破格','快速授权','试用']): f.add('external_talent_fast_regrade')
 # v0.8 precise query mechanisms
 if has_any(q,['私人关系','朋友','亲属关系','旧交']) and has_any(q,['机密','敏感信息','行动计划','泄密','提前告诉']) and has_any(q,['外部','竞争方','另一方','合作方']): f.add('relationship_confidentiality_leak')
 if has_any(q,['外部竞争','竞争对手','外部方']) and has_any(q,['收买','贿赂','利益输送','串联','内应']) and has_any(q,['内部人','核心人员','高层身边','信息中介','代理人']): f.add('insider_external_influence')
 if has_any(q,['内部矛盾','内部冲突','公司内斗','股东冲突']) and has_any(q,['引入外部','外部强者','强势资本','大平台','外部团队']) and has_any(q,['控制','退出','反客为主','授人以柄','独立资源']): f.add('external_force_internal_governance')
 if has_any(q,['唯一入口','关键入口','稀缺牌照','核心资质','制度资源','独占资源','关键控制点']) and has_any(q,['先占','窗口','别人也能拿','网络效应','必须先拿']): f.add('strategic_chokepoint')
 if has_any(q,['新领导','换领导','领导更换','新老板','换届']) and has_any(q,['旧团队','关键人才','核心负责人','前任留下']) and has_any(q,['换掉','留任','过渡','流失','连续性']): f.add('leadership_transition_continuity')
 if has_any(q,['原领导答应','上一任承诺','之前承诺','组织承诺','合同承诺']) and has_any(q,['新掌权','控制权变化','实际掌权者','换老板','换股东']) and has_any(q,['还认不认','不兑现','重谈','承接']): f.add('authority_commitment_gap')
 if has_any(q,['大家怀疑不公平','担心偏私','分配不透明','奖励还没分完']) and has_any(q,['先奖励','示范','最不喜欢的人','可信信号','稳定预期']): f.add('fairness_signal')
 if has_any(q,['主动交权','主动退位','退出一线','交给接班人']) and has_any(q,['保留顾问','保留荣誉','保留股权','保留身份','不清零']): f.add('voluntary_handoff')
 if has_any(q,['成功后','功成','阶段目标完成']) and has_any(q,['主动退出','淡出','不再扩权','交权','减少权力']) and has_any(q,['选择权','风险','生活','边际收益']): f.add('post_success_exit')
 if has_any(q,['信任破裂','已经不信任','建议不被采纳','实际没权']) and has_any(q,['还要不要留','退出太晚','离开窗口','责任还在']): f.add('late_exit_after_trust_collapse')
 if has_any(q,['重大失败','项目失败','第一次失败']) and has_any(q,['归因','甩锅','只怪执行','只怪供应链','只修局部']) and has_any(q,['战略假设','整体方向','重新验证','复盘']): f.add('failure_attribution_retest')
 if has_any(q,['没人反对','全票通过','都不说话','会议沉默','没有异议']) and has_any(q,['不敢说','怕领导','心理安全','虚假共识','真实共识']): f.add('silence_as_false_consensus')
 if has_any(q,['总部已停','总部已经停','老板已经叫停','董事会停损','董事会已经停损','最高层已停止','最高层已经停止','已经叫停项目']) and has_any(q,['一线负责人','项目经理','现场负责人']) and has_any(q,['继续做','不甘心','只差一步','无功而返','拒绝停','仍想继续']): f.add('frontline_stop_resistance')
 if has_any(q,['报表越来越好','数据漂亮','坏消息上不来','报喜不报忧','瞒报','不敢报坏消息']) and has_any(q,['现实恶化','领导惩罚','信息失真','独立核验']): f.add('bad_news_suppression')
 if has_any(q,['公开承诺','已经收对价','对方已经投入','组织信用','信誉']) and has_any(q,['后来反悔','解除承诺','战略变了','不想履约','领导判断变化']) and has_any(q,['声誉','长期合作','重谈','退出条件']): f.add('organizational_reputation_commitment')
 # v0.9 precise query mechanisms
 if has_any(q,['创始人','老板','最高领导','一把手']) and has_any(q,['个人提拔','个人保护','一直保我','私人支持','罩着我']) and has_any(q,['退休','换届','要走','即将离任','新老板','新领导']) and has_any(q,['降低依赖','提前退出','降低暴露','交接前','是否还安全']): f.add('sponsor_dependency_risk')
 if has_any(q,['项目完成','阶段结束','阶段目标完成','阶段目标已经完成','阶段目标已完成','完成阶段目标','战争结束','创业阶段结束']) and has_any(q,['依赖下降','依赖明显下降','依赖显著下降','不再那么需要','边际价值下降','贡献结构变化']) and ((has_any(q,['重新分配','重谈权限','重谈利益','重定价','调整权力'])) or (has_any(q,['重新谈','重谈','调整','重新调整']) and has_any(q,['权限','利益','收益','权力','分配']))): f.add('phase_change_reallocation')
 if has_any(q,['早期贡献','前期贡献','最近贡献','后期贡献','贡献变了','贡献变化']) and has_any(q,['奖金','收益','分配','股权','名位','排序']) and has_any(q,['重新','调整','重排','动态']): f.add('dynamic_contribution_regrading')
 if ((has_any(q,['危机中立大功','项目初期立大功','特殊阶段立功','功劳很大'])) or (has_any(q,['危机','特殊阶段']) and has_any(q,['立了大功','立大功','重大功劳']))) and has_any(q,['下一阶段','常态运营','日常管理','新阶段']) and has_any(q,['能力不匹配','不适合','岗位不匹配','不胜任','胜任']): f.add('phase_role_fit_reallocation')
 if has_any(q,['继任者','新领导','新老板','接班人']) and has_any(q,['前任核心团队','前任留下的高管','核心高管','关键人才']) and has_any(q,['重新任命','重新授权','重新确认','由新领导确认','建立自己的授权']): f.add('successor_mandate_reset')
 if has_any(q,['上一任承诺','前任承诺','上一任明确承诺','前任明确承诺','上一任明确保护','前任明确保护','明确承诺保护','旧领导答应']) and has_any(q,['新领导','新老板','继任者','换届','控制权变化']) and has_any(q,['还有效','有效吗','是否有效','是否承接','认不认','保护是否还在','旧承诺']): f.add('predecessor_commitment_continuity')
 if has_any(q,['业务负责人','谈判代表','销售负责人','项目负责人','代理人']) and has_any(q,['已经答应','作出承诺','承诺不追责','承诺价格','承诺条件']) and has_any(q,['对方已依赖','对方已经行动','对方也基于承诺行动','基于承诺行动','基于承诺已行动','已按承诺行动','基于承诺投入','信赖']) and has_any(q,['总部','公司','组织']) and has_any(q,['认不认','承接','兑现','反悔']): f.add('organizational_ratification_of_agent_commitment')
 return f

def structural_score(qf,cf):
 d=sum(FEATURE_WEIGHTS[x] for x in qf) or 1.0
 return sum(FEATURE_WEIGHTS[x] for x in qf&cf)/d

def role_multiplier(q,r):
 ts=tags(r); role=r.get('actor_role',''); title=r.get('case_title','')
 if 'platform_transition' in query_features(q) and 'portable_capability' in query_features(q):
  if has_tag(r,['平台选择','合作选择','跳槽','人才流动','边际价值']): return 1.0
  if any(x in role for x in ['人才','策士','功臣','战略执行者']): return 0.95
  if has_tag(r,['独立与并入','独立空间']): return 0.88
  if any(x in title for x in ['高平陵','袁绍因家庭','毛遂']): return 0.68
  return 0.80
 return 1.0

def rank(q,topn=10):
 rows=[json.loads(l) for l in CASES.read_text(encoding='utf-8').splitlines() if l.strip()]
 rows=[r for r in rows if r.get('lifecycle_status')=='ACTIVE']
 docs=[flat(r) for r in rows]
 vec=TfidfVectorizer(analyzer='char',ngram_range=(2,4),sublinear_tf=True)
 X=vec.fit_transform(docs+[q]); sims=cosine_similarity(X[-1],X[:-1]).ravel()
 qf=query_features(q); out=[]
 for r,sem in zip(rows,sims):
  cf=case_features(r); ss=structural_score(qf,cf)
  # exact structural features dominate; text similarity is a tiebreaker, not the core engine.
  raw=0.88*ss+0.12*float(sem)
  rm=role_multiplier(q,r); em=0.84+0.16*EVIDENCE_MULT.get(r.get('evidence_status'),0.65)
  score=raw*rm*em
  out.append({'case_id':r['case_id'],'title':r['case_title'],'score':round(score,4),'structural_score':round(ss,4),
              'semantic_score':round(float(sem),4),'matched_features':sorted(qf&cf),'case_features':sorted(cf),
              'role_multiplier':rm,'evidence_status':r.get('evidence_status'),'evidence_confidence':r.get('evidence_confidence'),
              'source_refs':r.get('source_refs',[]),'fatal_mismatch_conditions':r.get('fatal_mismatch_conditions',[])})
 out.sort(key=lambda x:x['score'],reverse=True)
 return {'query':q,'query_features':sorted(qf),'ranking':out[:topn]}

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('query'); ap.add_argument('--topn',type=int,default=10); ap.add_argument('--out')
 a=ap.parse_args(); res=rank(a.query,a.topn); s=json.dumps(res,ensure_ascii=False,indent=2)
 if a.out: pathlib.Path(a.out).write_text(s+'\n',encoding='utf-8')
 print(s)
