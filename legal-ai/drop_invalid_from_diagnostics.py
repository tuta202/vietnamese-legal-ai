from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from submit import save_submission


INVALID_CLASSES = {
    "expired_before_as_of",
    "not_yet_effective_as_of",
    "expired_full_unknown_date",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def law_id_from_article(article_ref: str) -> str:
    return str(article_ref).partition("|")[0].lstrip("'").strip()


def doc_ref_from_article(article_ref: str) -> str:
    parts = str(article_ref).split("|")
    return "|".join(parts[:2]) if len(parts) >= 2 else str(article_ref)


def load_effective_classes(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            law_id = (row.get("law_id") or "").lstrip("'").strip()
            class_name = (row.get("effective_class_as_of") or "").strip()
            if law_id:
                out[law_id] = class_name or "unknown"
    return out


def filter_invalid(article_ids: list[str], class_by_law: dict[str, str]) -> tuple[list[str], list[dict[str, str]]]:
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    for article_id in dedupe_keep_order(article_ids):
        class_name = class_by_law.get(law_id_from_article(article_id), "unknown")
        if class_name in INVALID_CLASSES:
            dropped.append({"article_id": article_id, "class": class_name})
        else:
            kept.append(article_id)
    return kept, dropped


def to_submission_row(row: dict[str, Any]) -> dict[str, Any]:
    articles = dedupe_keep_order(row.get("final_article_ids", []))
    return {
        "id": row["id"],
        "question": row.get("question", ""),
        "answer": "",
        "relevant_docs": dedupe_keep_order([doc_ref_from_article(article) for article in articles]),
        "relevant_articles": articles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop definitely invalid effective-status docs from diagnostics output.")
    parser.add_argument("--input-diagnostics", required=True)
    parser.add_argument("--status-csv", default="outputs/vbpl_full_corpus_status.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fallback-original-if-empty",
        action="store_true",
        help="Keep original final_article_ids if all articles would be dropped.",
    )
    args = parser.parse_args()

    rows = read_json(Path(args.input_diagnostics))
    class_by_law = load_effective_classes(Path(args.status_csv))
    out_rows: list[dict[str, Any]] = []
    before_sizes: list[int] = []
    after_sizes: list[int] = []
    class_counts: Counter[str] = Counter()
    changed_questions = 0
    fallback_empty = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        original = dedupe_keep_order(row.get("final_article_ids", []))
        before_sizes.append(len(original))
        final, dropped = filter_invalid(original, class_by_law)
        for item in dropped:
            class_counts[item["class"]] += 1

        fallback_used = False
        if args.fallback_original_if_empty and original and not final:
            final = original
            dropped = []
            fallback_used = True
            fallback_empty += 1

        if dropped:
            changed_questions += 1
            if len(examples) < 25:
                examples.append(
                    {
                        "id": row.get("id"),
                        "question": row.get("question", ""),
                        "before": original,
                        "dropped": dropped,
                        "after": final,
                    }
                )

        new_row = dict(row)
        new_row["final_article_ids_before_invalid_drop"] = original
        new_row["final_article_ids"] = final
        # Make downstream rescue/evidence order consistent with the cleaned set.
        new_row["candidate_article_ids_before_invalid_drop"] = row.get("candidate_article_ids", [])
        new_row["candidate_article_ids"] = final
        new_row["drop_definitely_invalid_asof_20260301"] = {
            "invalid_classes": sorted(INVALID_CLASSES),
            "dropped": dropped,
            "fallback_original_if_empty": args.fallback_original_if_empty,
            "fallback_used": fallback_used,
        }
        out_rows.append(new_row)
        after_sizes.append(len(final))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    save_submission([to_submission_row(row) for row in out_rows], output_dir)

    summary = {
        "input_diagnostics": args.input_diagnostics,
        "status_csv": args.status_csv,
        "questions": len(out_rows),
        "changed_questions": changed_questions,
        "removed_articles": sum(class_counts.values()),
        "class_counts": dict(class_counts.most_common()),
        "mean_before": statistics.mean(before_sizes) if before_sizes else 0,
        "mean_after": statistics.mean(after_sizes) if after_sizes else 0,
        "min_after": min(after_sizes) if after_sizes else 0,
        "max_after": max(after_sizes) if after_sizes else 0,
        "empty_after": sum(1 for size in after_sizes if size == 0),
        "fallback_empty": fallback_empty,
        "examples": examples,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
