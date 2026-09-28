#!/usr/bin/env python3
"""Reread designated source anchors. Raw texts remain outside the package.

An anchor match verifies text location, not every historical/modern claim.
"""
import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup
from validate_database import ROOT, load_lines


def normalize(text):
    return re.sub(r"[\s，。；：、！？‘’“”「」『』…,.!?;:\"'（）()—-]", "", text or "")


def clauses(text):
    return [normalize(s) for s in re.split(r"[…。；;，,]", text or "") if len(normalize(s)) >= 4]


def numeral(text):
    if text.isdigit():
        return int(text)
    digits = dict(zip("零一二三四五六七八九", range(10)))
    units = {"十": 10, "百": 100}
    total = value = 0
    for ch in text:
        if ch in digits:
            value = digits[ch]
        elif ch in units:
            total += (value or 1) * units[ch]
            value = 0
    return total + value


def fetch_page(url, cache, refresh=False):
    name = hashlib.sha256(url.encode()).hexdigest() + ".html"
    path = cache / name
    if refresh or not path.exists():
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "zh.wikisource.org":
            return url, None, "unexpected_source_host"
        request_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc,
                       urllib.parse.quote(urllib.parse.unquote(parsed.path)), parsed.query, ""))
        try:
            req = urllib.request.Request(request_url, headers={"User-Agent": "HistoricalDecisionMirror/1.0 source-verification"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            path.write_bytes(data)
        except Exception as exc:
            return url, None, type(exc).__name__
    return url, path, None


def validate(epub_ops, cache, refresh=False):
    cache = Path(cache).resolve()
    if cache == ROOT or ROOT in cache.parents:
        raise ValueError("原文缓存必须位于发布目录之外")
    cache.mkdir(parents=True, exist_ok=True)
    epub_ops = Path(epub_ops).resolve()
    rows = [r for r in load_lines(ROOT / "data/decision_nodes.jsonl") if r["lifecycle_status"] == "ACTIVE"]
    integrity = {r["volume"]: r for r in json.loads((ROOT / "data/source_integrity.json").read_text())}
    urls = sorted({s["url"] for r in rows for s in r["source_refs"] if s["source_id"] == "SHIJI_WIKISOURCE"})
    with ThreadPoolExecutor(max_workers=3) as pool:
        pages = {url: (path, error) for url, path, error in pool.map(lambda u: fetch_page(u, cache, refresh), urls)}
    text_cache = {}

    def read(path, wiki=False):
        key = str(path)
        if key not in text_cache:
            data = path.read_bytes()
            soup = BeautifulSoup(data, "html.parser")
            if wiki:
                bodies = soup.select(".mw-parser-output")
                soup = max(bodies, key=lambda b: len(b.get_text())) if bodies else soup
            text_cache[key] = (normalize(soup.get_text("", strip=True)), hashlib.sha256(data).hexdigest())
        return text_cache[key]

    results = []
    for row in rows:
        for index, source in enumerate(row["source_refs"]):
            entry = dict(case_id=row["case_id"], ref_index=index, source_id=source["source_id"],
                         locator=source.get("locator"), status="UNVERIFIED", matched_anchors=[], file_sha256=[])
            paths = []
            if source["source_id"] == "SHIJI_WIKISOURCE":
                path, error = pages[source["url"]]
                if error:
                    entry["reason"] = error
                else:
                    paths = [path]
            else:
                match = re.search(r"卷([零一二三四五六七八九十百\d]+)", source.get("locator", ""))
                volume = numeral(match.group(1)) if match else None
                record = integrity.get(volume, {})
                entry["volume"] = volume
                if record.get("status") == "MISSING_CONFIRMED":
                    entry["reason"] = "designated_source_gap"
                elif source.get("epub_file"):
                    path = (epub_ops / source["epub_file"]).resolve()
                    if epub_ops not in path.parents:
                        entry["reason"] = "unsafe_source_path"
                    else:
                        paths = [path]
                else:
                    paths = [epub_ops / ("chapter" + str(n) + ".htm") for n in record.get("chapters", [])]
            anchors = clauses(source.get("anchor", "")) or [a for x in row["evidence_anchors"] for a in clauses(x)]
            entry["anchor_clauses_checked"] = len(anchors)
            for path in paths:
                if not path.is_file():
                    entry["reason"] = "source_file_unavailable"
                    continue
                text, digest = read(path, source["source_id"] == "SHIJI_WIKISOURCE")
                entry["file_sha256"].append(digest)
                entry["matched_anchors"] += [a for a in anchors if a in text]
            if entry["matched_anchors"]:
                entry["status"] = "ANCHOR_MATCH"
            elif paths and entry["file_sha256"]:
                entry["status"] = "ANCHOR_MISMATCH"
            results.append(entry)
    return dict(timestamp=datetime.now(timezone.utc).isoformat(), scope="literal_anchor_location_only",
                current_network_refresh=refresh, references=len(results),
                matched=sum(x["status"] == "ANCHOR_MATCH" for x in results),
                not_matched=sum(x["status"] != "ANCHOR_MATCH" for x in results),
                claim_semantics="requires contextual reading; not automatically certified", results=results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epub-ops", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    report = validate(args.epub_ops, args.cache, args.refresh)
    (ROOT / "reports/source_anchors.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["not_matched"] else 0)
