"""Build deterministic article-only submissions for intermediate retrieval phases."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from legal_rag.output.submission import save_submission


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def article_doc_ref(article_ref: str) -> str:
    document, separator, _article = str(article_ref).rpartition("|")
    return document if separator else str(article_ref)


def round_robin_top_each(intent_rankings: list[list[str]], top_each: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for rank in range(max(0, top_each)):
        for ranking in intent_rankings:
            if rank >= len(ranking):
                continue
            article = ranking[rank]
            if article and article not in seen:
                seen.add(article)
                selected.append(article)
    return selected


def load_jsonl_latest(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            try:
                row = json.loads(raw)
                key = (str(row["question_id"]), int(row["intent_index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            rows[key] = row
    return rows


def raw_intent_rankings(row: dict[str, Any]) -> list[list[str]]:
    return [
        [str(item.get("article") or "") for item in intent.get("ranked_articles", [])]
        for intent in row.get("intent_ranked_hits", [])
    ]


def bge_intent_rankings(
    question_id: str,
    num_intents: int,
    cache: dict[tuple[str, int], dict[str, Any]],
) -> list[list[str]]:
    return [
        [
            str(item.get("article") or "")
            for item in cache.get((question_id, intent_index), {}).get("ranked_articles", [])
        ]
        for intent_index in range(num_intents)
    ]


def build_snapshot_rows(
    questions: list[dict[str, Any]],
    rankings_by_question: dict[str, list[list[str]]],
    top_each: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question["id"])
        articles = round_robin_top_each(rankings_by_question.get(question_id, []), top_each)
        if not articles:
            raise ValueError(f"Q{question_id} has no articles for phase snapshot")
        output.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answer": "",
                "relevant_docs": dedupe_keep_order([article_doc_ref(item) for item in articles]),
                "relevant_articles": articles,
            }
        )
    return output


def write_snapshot(rows: list[dict[str, Any]], output_dir: Path, metadata: dict[str, Any]) -> Path:
    zip_path = save_submission(rows, output_dir)
    sizes = [len(row["relevant_articles"]) for row in rows]
    summary = {
        **metadata,
        "questions": len(rows),
        "article_count_min": min(sizes) if sizes else 0,
        "article_count_max": max(sizes) if sizes else 0,
        "article_count_mean": statistics.mean(sizes) if sizes else 0,
        "submission_zip": str(zip_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return zip_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Package an intermediate phase submission.")
    parser.add_argument("--mode", choices=("raw-intent", "bge-intent"), required=True)
    parser.add_argument("--input", required=True, help="Original question JSON")
    parser.add_argument("--intent-results", required=True)
    parser.add_argument("--bge-cache", default="")
    parser.add_argument("--top-each", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    intent_rows = json.loads(Path(args.intent_results).read_text(encoding="utf-8"))
    intents_by_id = {str(row["id"]): row for row in intent_rows}

    if args.mode == "raw-intent":
        rankings = {
            question_id: raw_intent_rankings(row)
            for question_id, row in intents_by_id.items()
        }
    else:
        if not args.bge_cache:
            raise ValueError("--bge-cache is required for bge-intent mode")
        cache = load_jsonl_latest(Path(args.bge_cache))
        rankings = {
            question_id: bge_intent_rankings(
                question_id,
                len(row.get("legal_intents", [])),
                cache,
            )
            for question_id, row in intents_by_id.items()
        }

    rows = build_snapshot_rows(questions, rankings, args.top_each)
    write_snapshot(
        rows,
        Path(args.output_dir),
        {"mode": args.mode, "top_each_intent": args.top_each},
    )


if __name__ == "__main__":
    main()
