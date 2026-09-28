#!/usr/bin/env python3
"""Shared entry -> retrieval -> contrast -> evidence gate for every mode.

This prepares candidates for the skill host. It never invents a final decision.
"""
import argparse
import json
from pathlib import Path
from entry import parse_request
from retrieve import rank
from contrast import by_case


def prepare(text):
    request = parse_request(text)
    result = rank(request["problem"], 5)
    candidates = [r for r in result["ranking"] if r["structural_score"] > 0]
    supported = bool(result["query_features"] and candidates)
    primary = candidates[0] if supported else None
    pairs = by_case(primary["case_id"]) if primary else []
    return {
        "version": "1.0.0", "request": request,
        "status": "CANDIDATES_REQUIRE_REVIEW" if supported else "ABSTAIN_INSUFFICIENT_STRUCTURE",
        "query_features": result["query_features"],
        "candidates": candidates if supported else [],
        "contrast_candidates": pairs,
        "contrast_status": "CANDIDATES_REQUIRE_REVIEW" if pairs else "MISSING",
        "evidence_status": "REQUIRES_CURRENT_SOURCE_READ" if supported else "NOT_APPLICABLE",
        "required_gates": ["decision", "evidence", "role", "structure", "institutional_context", "hindsight"],
        "conclusion": None,
        "notice": "候选分数不证明角色同构或原文支持；宿主须读取来源、审计失配并按模式推演。" if supported else "结构特征不足，不能可靠类比。",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.request), ensure_ascii=False, indent=2))
    except ValueError as exc:
        parser.error(str(exc))
