from __future__ import annotations

import argparse
import csv
import json
import statistics
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("outputs/runs/gpu_phase_benchmark")


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_item in items:
        item = " ".join(str(raw_item).split())
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def article_doc_ref(article_ref: str) -> str:
    parts = article_ref.split("|")
    return "|".join(parts[:2]) if len(parts) >= 2 else article_ref


def index_rows(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("id"))
        if question_id in indexed:
            raise ValueError(f"Duplicate question id {question_id} in {source}")
        indexed[question_id] = row
    return indexed


def rescue_question(
    final_articles: list[str],
    stage1_articles: list[str],
    intent_rows: list[dict[str, Any]],
    *,
    coverage_depth: int = 3,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Restore raw-intent top1 only when final lost all top-N coverage.

    The candidate must have survived Stage 1 cleanup. This keeps the rescue
    focused on over-pruning by the final verifier rather than undoing Stage 1.
    """
    final = dedupe_keep_order(final_articles)
    stage1_set = set(dedupe_keep_order(stage1_articles))
    final_set = set(final)
    rescued: list[dict[str, Any]] = []

    for intent_row in intent_rows:
        ranked = sorted(
            intent_row.get("ranked_articles", []),
            key=lambda item: int(item.get("rank", 10**9)),
        )
        if not ranked:
            continue

        coverage_articles = {
            " ".join(str(item.get("article") or "").split())
            for item in ranked[:coverage_depth]
        }
        coverage_articles.discard("")
        if final_set & coverage_articles:
            continue

        top1 = " ".join(str(ranked[0].get("article") or "").split())
        if not top1 or top1 not in stage1_set or top1 in final_set:
            continue

        final.append(top1)
        final_set.add(top1)
        rescued.append(
            {
                "intent_index": intent_row.get("intent_index"),
                "intent_id": intent_row.get("intent_id"),
                "intent": intent_row.get("intent", ""),
                "article": top1,
                "raw_intent_rank": ranked[0].get("rank", 1),
                "raw_intent_rrf_score": ranked[0].get("rrf_score"),
            }
        )

    return final, rescued


def build_rescued_submission(
    final_rows: list[dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
    intent_rows: list[dict[str, Any]],
    *,
    coverage_depth: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage1_by_id = index_rows(stage1_rows, "stage1 results")
    intents_by_id = index_rows(intent_rows, "intent rankings")
    final_by_id = index_rows(final_rows, "final results")
    expected_ids = set(final_by_id)
    if set(stage1_by_id) != expected_ids or set(intents_by_id) != expected_ids:
        raise ValueError("Final, Stage 1, and intent files do not contain the same question ids")

    output_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in final_rows:
        question_id = str(row.get("id"))
        before = dedupe_keep_order(row.get("relevant_articles", []))
        after, rescued = rescue_question(
            before,
            stage1_by_id[question_id].get("relevant_articles", []),
            intents_by_id[question_id].get("intent_ranked_hits", []),
            coverage_depth=coverage_depth,
        )
        new_row = dict(row)
        new_row["relevant_articles"] = after
        new_row["relevant_docs"] = dedupe_keep_order([article_doc_ref(item) for item in after])
        output_rows.append(new_row)
        diagnostics.append(
            {
                "question_id": row.get("id"),
                "num_intents": len(intents_by_id[question_id].get("intent_ranked_hits", [])),
                "before_size": len(before),
                "after_size": len(after),
                "rescued_count": len(after) - len(before),
                "rescued": rescued,
            }
        )
    return output_rows, diagnostics


def write_outputs(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(results_path, arcname="results.json")

    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("question_id", "num_intents", "before_size", "after_size", "rescued_count"),
        )
        writer.writeheader()
        for row in diagnostics:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    sizes = [len(row.get("relevant_articles", [])) for row in rows]
    changed = [row for row in diagnostics if row["rescued_count"]]
    summary = {
        "questions": len(rows),
        "changed_questions": len(changed),
        "rescued_articles": sum(row["rescued_count"] for row in changed),
        "min_articles": min(sizes) if sizes else 0,
        "max_articles": max(sizes) if sizes else 0,
        "mean_articles": statistics.mean(sizes) if sizes else 0,
        "submission_zip": str(zip_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescue uncovered raw-intent top1 articles after final verification")
    parser.add_argument(
        "--final-results",
        default=str(DEFAULT_RUN_DIR / "submissions/best_final_enforcement_gate/results.json"),
    )
    parser.add_argument(
        "--stage1-results",
        default=str(DEFAULT_RUN_DIR / "submissions/stage1_gemma_v4_penalty_cleanup/results.json"),
    )
    parser.add_argument(
        "--intent-results",
        default=str(DEFAULT_RUN_DIR / "artifacts/intent_ranked_hits.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RUN_DIR / "submissions/best_final_enforcement_gate_rawintent_top1_rescue"),
    )
    parser.add_argument("--coverage-depth", type=int, default=3)
    args = parser.parse_args()
    if args.coverage_depth < 1:
        parser.error("--coverage-depth must be at least 1")

    final_rows = json.loads(Path(args.final_results).read_text(encoding="utf-8"))
    stage1_rows = json.loads(Path(args.stage1_results).read_text(encoding="utf-8"))
    intent_rows = json.loads(Path(args.intent_results).read_text(encoding="utf-8"))
    rows, diagnostics = build_rescued_submission(
        final_rows,
        stage1_rows,
        intent_rows,
        coverage_depth=args.coverage_depth,
    )
    summary = write_outputs(rows, diagnostics, Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
