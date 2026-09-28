#!/usr/bin/env python3
"""One honest entry point: missing migration is BLOCKED, never silently skipped."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from entry import parse_request
from check_analogy import check_record

ROOT = Path(__file__).resolve().parents[1]


def run(component=False, epub_ops=None, cache=None, refresh=False):
    results = []

    def record(name, ok, detail=None):
        results.append(dict(id=name, status="PASS" if ok else "FAIL", detail=detail))

    inputs = [
        ("/历史推演 quick 是否继续合作？", "quick", "是否继续合作？"),
        ("/历史推演 deep 是否退出？", "deep", "是否退出？"),
        ("/历史推演 audit 我的判断成立吗？", "audit", "我的判断成立吗？"),
        ("/历史推演 是否退出？", "deep", "是否退出？"),
        ("调用历史抉择镜鉴：是否退出？", "deep", "是否退出？"),
        ("用资治通鉴推演我该如何面对换届", "deep", "用资治通鉴推演我该如何面对换届"),
    ]
    for index, (value, mode, problem) in enumerate(inputs):
        out = parse_request(value)
        record("entry_" + str(index), out["mode"] == mode and out["problem"] == problem
               and out["raw_problem"] == value and out["user_value_judgments"] == []
               and out["primary_goal"] is None)
    for index, value in enumerate(["", "/历史推演", "/历史推演 quick", "/历史推演 turbo 问题", "今天天气如何"]):
        try:
            parse_request(value)
            record("entry_reject_" + str(index), False)
        except ValueError:
            record("entry_reject_" + str(index), True)
    baseline = dict(decision="allow", structural_score=0.8, role_match=True,
                    mechanism_match=True, critical_differences=["现代组织有独立制度约束"],
                    evidence_status="verified", hindsight_used_as_t0=False,
                    modern_rights_overtransfer=False, invented_user_values=False,
                    contrast_status="verified")
    record("positive_contract", check_record(baseline) == [])
    fixtures = json.loads((ROOT / "evals/negative_analogy.json").read_text(encoding="utf-8"))
    for fixture in fixtures:
        errors = check_record(dict(baseline, **fixture["patch"]))
        record(fixture["id"], set(errors) == set(fixture["expected_errors"]), errors)
    for name, value in [("nan", float("nan")), ("infinity", float("inf")), ("bool", True)]:
        record("reject_score_" + name, "invalid_score" in check_record(dict(baseline, structural_score=value)))
    for field in baseline:
        missing = baseline.copy()
        del missing[field]
        record("required_" + field, bool(check_record(missing)))
    abstain = dict(baseline, decision="abstain", structural_score=0, role_match=False,
                   mechanism_match=False, evidence_status="unavailable", contrast_status="missing",
                   critical_differences=[])
    record("honest_abstention", check_record(abstain) == [])
    record("invalid_root", check_record([]) == ["invalid_record"])
    if not component:
        from retrieve import rank
        from contrast import by_case, from_query
        from runtime import prepare
        from validate_database import validate, load_lines
        database = validate()
        record("database_structure", not database["errors"], database)
        active = {r["case_id"] for r in load_lines(ROOT / "data/decision_nodes.jsonl") if r["lifecycle_status"] == "ACTIVE"}
        for item in json.loads((ROOT / "evals/retrieval.json").read_text()):
            out = rank(item["query"], 5)
            ids = [r["case_id"] for r in out["ranking"]]
            record("retrieval:" + item["test_id"], ids[0] == item["expected_top1"], {"top3": ids[:3], "expected": item["expected_top1"]})
        regression = json.loads((ROOT / "evals/regression.json").read_text())
        for item in regression["tests"]:
            ids = [r["case_id"] for r in rank(item["query"], 3)["ranking"]]
            record("regression:" + item["test_id"], bool(set(ids) & set(item["expected_top3_any"])), ids)
        for item in json.loads((ROOT / "evals/contrast.json").read_text()):
            ids = [p["pair_id"] for p in by_case(item["case_id"], 10)]
            record("contrast:" + item["test_id"], item["expected_pair"] in ids, ids)
        for item in json.loads((ROOT / "evals/end_to_end.json").read_text()):
            out = from_query(item["query"], 5)
            ids = [p["pair_id"] for p in out["contrast_candidates"]]
            record("end_to_end:" + item["test_id"], out["primary"]["case_id"] == item["expected_primary"]
                   and item["expected_pair"] in ids and out["contrast_confidence"] == "OK")
        ids = [r["case_id"] for r in rank("领导更替与组织承诺", 200)["ranking"]]
        record("no_deprecated_leak", len(ids) == len(active) and set(ids) == active)
        for mode in ("quick", "deep", "audit"):
            packet = prepare("/历史推演 " + mode + " 同事名字里有个信字，他是不是就像韩信？")
            record("name_only_abstention:" + mode, packet["status"] == "ABSTAIN_INSUFFICIENT_STRUCTURE"
                   and not packet["candidates"] and not packet["contrast_candidates"] and packet["conclusion"] is None)
            packet = prepare("/历史推演 " + mode + " 创始人一直个人提拔并保护我，但他马上退休，新老板跟我没关系，我是否应在交接前降低依赖")
            record("mode_pipeline:" + mode, packet["request"]["mode"] == mode
                   and packet["candidates"][0]["case_id"] == "HD-106"
                   and packet["evidence_status"] == "REQUIRES_CURRENT_SOURCE_READ" and packet["conclusion"] is None)
        # All source-state failures must fail closed at the conclusion gate.
        for state in ("unverified", "unavailable", "mismatch"):
            record("source_gate:" + state, "evidence_not_verified" in check_record(dict(baseline, evidence_status=state)))
        if epub_ops:
            from validate_evidence import validate as validate_sources
            evidence = validate_sources(epub_ops, cache, refresh)
            (ROOT / "reports/source_anchors.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
            for item in evidence["results"]:
                record("source_anchor:" + item["case_id"] + ":" + str(item["ref_index"]), item["status"] == "ANCHOR_MATCH")
    failed = sum(item["status"] == "FAIL" for item in results)
    manifest = json.loads((ROOT / "evals/manifest.json").read_text(encoding="utf-8"))
    pending = [s["id"] for s in manifest["suites"] if s["status"] != "implemented"]
    migration = json.loads((ROOT / "data/migration_status.json").read_text(encoding="utf-8"))
    if not migration["imported"]:
        pending.append("v0.9_artifact_import")
    status = "FAIL" if failed else "BLOCKED" if pending and not component else "PASS"
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  scope="component" if component else "full_with_source_anchors" if epub_ops else "offline", status=status,
                  executed=len(results), passed=len(results)-failed, failed=failed,
                  pending_suites=pending, source_recheck=bool(epub_ops),
                  release_allowed=(status == "PASS" and bool(epub_ops) and not component), results=results)
    output = ROOT / "reports" / ("component.json" if component else "full.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    return 1 if failed else 2 if status == "BLOCKED" else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", action="store_true")
    parser.add_argument("--epub-ops")
    parser.add_argument("--cache")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if bool(args.epub_ops) != bool(args.cache):
        parser.error("--epub-ops 和 --cache 必须同时提供")
    raise SystemExit(run(args.component, args.epub_ops, args.cache, args.refresh))
