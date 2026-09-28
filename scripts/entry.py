#!/usr/bin/env python3
"""Parse the public invocation; semantic reality parsing belongs to the agent."""
import json
import re
import sys

MODES = {"quick", "deep", "audit"}


def parse_request(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入现实问题")
    problem = text.strip()
    mode = "deep"
    explicit = re.match(r"^/历史推演(?=\s|$)", problem)
    if explicit:
        problem = problem[explicit.end():].strip()
        parts = problem.split(None, 1)
        if parts and parts[0] in MODES:
            mode = parts[0]
            problem = parts[1].strip() if len(parts) > 1 else ""
        elif parts and re.fullmatch(r"[A-Za-z_-]+", parts[0]):
            raise ValueError("未知模式；请使用 quick、deep 或 audit")
    else:
        natural = re.match(r"^(?:调用)?历史抉择镜鉴[：:]\s*", problem)
        if natural:
            problem = problem[natural.end():].strip()
        elif not re.search(r"历史推演|用(?:《)?(?:史记|资治通鉴)(?:》)?(?:分析|推演)", problem):
            raise ValueError("未识别为历史决策请求；语义触发由宿主 Agent 判断")
    if not problem:
        raise ValueError("入口之后缺少现实问题")
    result = dict(mode=mode, raw_problem=text, problem=problem)
    for key in ("decision_subject", "decision_question", "primary_goal", "time_horizon"):
        result[key] = None
    for key in ("current_options", "constraints", "irreversible_factors", "known_facts",
                "assumptions", "unknown_variables", "user_value_judgments"):
        result[key] = []
    return result


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("用法：python3 scripts/entry.py '<用户原话>'")
        print(json.dumps(parse_request(sys.argv[1]), ensure_ascii=False, indent=2))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
