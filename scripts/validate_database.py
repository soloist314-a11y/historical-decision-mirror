#!/usr/bin/env python3
"""Offline structural validation; source rereading is a separate suite."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(root=ROOT):
    rows = load_lines(root / "data/decision_nodes.jsonl")
    pairs = load_lines(root / "data/contrast_pairs.jsonl")
    sources = {s["source_id"] for s in json.loads((root / "data/source_registry.json").read_text())["sources"]}
    errors = []
    required = ["case_id", "version", "case_title", "source_refs", "evidence_status", "decision_actor",
                "decision_question", "actual_choice", "choice_rationale", "results", "costs", "structures",
                "option_value_effect", "tags", "structural_match_keys", "evidence_anchors",
                "evidence_confidence", "lifecycle_status", "superseded_by"]
    all_ids = {r["case_id"] for r in rows}
    active = {r["case_id"] for r in rows if r.get("lifecycle_status") == "ACTIVE"}
    for key, count in collections.Counter(r["case_id"] for r in rows).items():
        if count != 1:
            errors.append("duplicate_case:" + key)
    for row in rows:
        cid = row["case_id"]
        for key in required:
            if key not in row:
                errors.append(cid + ":missing:" + key)
        if row.get("lifecycle_status") not in {"ACTIVE", "DEPRECATED"}:
            errors.append(cid + ":invalid_lifecycle")
        for target in row.get("superseded_by", []):
            if target not in all_ids:
                errors.append(cid + ":missing_successor:" + target)
        if cid not in active:
            if not row.get("superseded_by"):
                errors.append(cid + ":deprecated_without_successor")
            continue
        for key in ("decision_actor", "actual_choice", "source_refs", "evidence_anchors", "fatal_mismatch_conditions"):
            if not row.get(key):
                errors.append(cid + ":empty:" + key)
        for source in row["source_refs"]:
            sid = source.get("source_id")
            if sid not in sources:
                errors.append(cid + ":unknown_source")
            if not source.get("locator"):
                errors.append(cid + ":missing_locator")
            if sid == "SHIJI_WIKISOURCE" and not source.get("url", "").startswith("https://zh.wikisource.org/"):
                errors.append(cid + ":invalid_wikisource")
            if sid == "ZZTJ_DRIVE_BAIYANG" and not (source.get("epub_file") or source.get("local_evidence_ids")):
                errors.append(cid + ":missing_epub_anchor")
    for pid, count in collections.Counter(p["pair_id"] for p in pairs).items():
        if count != 1:
            errors.append("duplicate_pair:" + pid)
    for pair in pairs:
        pid = pair["pair_id"]
        for key in ("primary_case_id", "contrast_case_id", "surface_similarity", "critical_difference",
                    "why_outcomes_diverge", "retrieval_trigger", "misuse_warning"):
            if not pair.get(key):
                errors.append(pid + ":empty:" + key)
        if pair["primary_case_id"] not in active or pair["contrast_case_id"] not in active:
            errors.append(pid + ":inactive_endpoint")
        if pair["primary_case_id"] == pair["contrast_case_id"]:
            errors.append(pid + ":same_endpoint")
    return dict(records=len(rows), active=len(active), deprecated=len(rows)-len(active),
                contrast_pairs=len(pairs), errors=errors,
                scope="offline_structure_only; historical evidence flags are not current verification")


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result["errors"]))
