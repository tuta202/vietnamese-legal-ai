"""
submit.py — validate and package a submission for the Vietnamese Legal AI competition.

Standalone module: only stdlib (json, re, zipfile, pathlib, argparse, sys).

CLI:
    python submit.py --input results.json --output submission/
    python submit.py --validate-only results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

# Patterns
_DOC_RE     = re.compile(r'^.+\|.+$')                        # id|type name
_ARTICLE_RE = re.compile(r'^.+\|.+\|Điều\s+\d+[a-zđ]?',     # id|type name|Điều X
                          re.UNICODE)
_DIEU_RE    = re.compile(r'Điều\s+\d+',                       # Điều X in answer
                          re.UNICODE)

REQUIRED_FIELDS = {"id", "question", "answer", "relevant_docs", "relevant_articles"}


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_submission(results: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate a list of submission entries.

    Returns (is_valid, error_messages).
    is_valid is True only when errors is empty.
    """
    errors: list[str] = []
    ids_seen: set = set()

    for i, entry in enumerate(results):
        label = f"Entry[{i}] id={entry.get('id', '?')}"

        # a) All 5 required fields present
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
            continue   # skip deeper checks for this entry

        entry_id = entry["id"]

        # f) id must be convertible to int
        try:
            int(entry_id)
        except (TypeError, ValueError):
            errors.append(f"{label}: 'id' is not an integer ({entry_id!r})")

        # e) no duplicate ids
        if entry_id in ids_seen:
            errors.append(f"{label}: duplicate id {entry_id!r}")
        ids_seen.add(entry_id)

        # b) relevant_docs format: "{id}|{type} {name}"
        for doc in entry["relevant_docs"]:
            if not _DOC_RE.match(doc):
                errors.append(
                    f"{label}: relevant_doc bad format: {doc!r}"
                    " (expected '{law_id}|{type} {name}')"
                )

        # c) relevant_articles format: "{id}|{type} {name}|Điều X"
        for art in entry["relevant_articles"]:
            if not _ARTICLE_RE.match(art):
                errors.append(
                    f"{label}: relevant_article bad format: {art!r}"
                    " (expected '{law_id}|{type} {name}|Điều X')"
                )

        # d) answer must contain ≥1 "Điều X" citation
        if not _DIEU_RE.search(entry["answer"]):
            errors.append(
                f"{label}: answer missing 'Điều X' citation — IR score will be 0"
            )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_submission(results: list[dict], output_dir: Path | str) -> Path:
    """
    Write results.json and package it into submission.zip.

    Returns the path to the created ZIP file.
    Raises AssertionError if the ZIP structure is wrong.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write results.json
    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2. Create submission.zip with results.json at root (flat)
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    # 3. Verify structure
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert names == ["results.json"], (
        f"Unexpected ZIP structure: {names} (expected ['results.json'])"
    )

    size_kb = zip_path.stat().st_size // 1024
    print(f"Saved {len(results)} entries → {zip_path}  ({size_kb or '<1'}KB)")
    print(f"  ZIP contents: {names}")
    return zip_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and package submission")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",         metavar="FILE",   help="results.json to validate + package")
    group.add_argument("--validate-only", metavar="FILE",   help="Validate only, no ZIP created")
    parser.add_argument("--output", default="submission/",  help="Output directory for ZIP")
    args = parser.parse_args()

    input_file = args.input or args.validate_only
    results    = json.loads(Path(input_file).read_text(encoding="utf-8"))

    valid, errors = validate_submission(results)
    if errors:
        print(f"INVALID: {len(errors)} error(s)")
        for err in errors:
            print(f"  ✗ {err}")
        sys.exit(1)

    print(f"VALID: {len(results)} entries, 0 errors")

    if args.input:
        save_submission(results, args.output)


if __name__ == "__main__":
    main()
