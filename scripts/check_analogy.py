#!/usr/bin/env python3
"""Validate explicit audit constraints, not the truth of supplied judgments."""
import json
import math
import sys


def check_record(record):
    if not isinstance(record, dict):
        return ["invalid_record"]
    errors = []
    decision = record.get("decision")
    if decision not in {"allow", "abstain", "request_information"}:
        errors.append("invalid_decision")
    score = record.get("structural_score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        errors.append("invalid_score")
    flags = ("role_match", "mechanism_match", "hindsight_used_as_t0",
             "modern_rights_overtransfer", "invented_user_values")
    for flag in flags:
        if type(record.get(flag)) is not bool:
            errors.append("invalid_" + flag)
    differences = record.get("critical_differences")
    if not isinstance(differences, list) or any(not isinstance(x, str) or not x.strip() for x in differences):
        errors.append("invalid_differences")
    if record.get("evidence_status") not in {"verified", "unavailable", "mismatch", "unverified"}:
        errors.append("invalid_evidence_status")
    if record.get("contrast_status") not in {"verified", "missing", "unverified"}:
        errors.append("invalid_contrast_status")
    if errors:
        return errors
    if decision == "allow":
        if score <= 0:
            errors.append("no_structural_support")
        if not record["role_match"]:
            errors.append("role_mismatch")
        if not record["mechanism_match"]:
            errors.append("mechanism_mismatch")
        if not differences:
            errors.append("missing_critical_differences")
        if record["evidence_status"] != "verified":
            errors.append("evidence_not_verified")
        if record["contrast_status"] != "verified":
            errors.append("contrast_not_verified")
    for flag in flags[2:]:
        if record[flag]:
            errors.append(flag)
    return errors


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("用法：python3 scripts/check_analogy.py record.json")
        with open(sys.argv[1], encoding="utf-8") as stream:
            errors = check_record(json.load(stream))
        print(json.dumps({"valid": not errors, "errors": errors}, ensure_ascii=False))
        sys.exit(1 if errors else 0)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
