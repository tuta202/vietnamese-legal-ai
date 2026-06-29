from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from legal_rag.common.article_lookup import ArticleLookup
from legal_rag.output.submission import save_submission


INVALID_CLASSES = {
    "expired_before_as_of",
    "not_yet_effective_as_of",
    "expired_full_unknown_date",
}

PENALTY_LAW_OR_TITLE_RE = re.compile(
    r"(xử phạt|vi phạm hành chính|mức phạt|phạt tiền|biện pháp khắc phục hậu quả|"
    r"thẩm quyền xử phạt|hình thức xử phạt)",
    re.I,
)

PENALTY_QUESTION_RE = re.compile(
    r"(xử phạt|mức phạt|phạt|vi phạm|xử lý vi phạm|bị xử lý|xử lý như thế nào|"
    r"hình thức xử lý|rủi ro pháp lý|hậu quả pháp lý|khắc phục hậu quả|chế tài|"
    r"trách nhiệm pháp lý|sai phạm|cưỡng chế)",
    re.I,
)

NON_PENALTY_QUESTION_RE = re.compile(
    r"(điều kiện|thủ tục|hồ sơ|thời hạn|bao lâu|cơ quan nào|nộp ở đâu|nộp tại đâu|"
    r"được hưởng|ưu đãi|quyền|nghĩa vụ|cách tính|xác định|cần chuẩn bị|"
    r"đăng ký|kê khai|khai thuế|sử dụng loại|áp dụng|có được|phải thực hiện|"
    r"thành phần|bao gồm|cần có)",
    re.I,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


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
    return str(article_ref).split("|", 1)[0].lstrip("'").strip()


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


def is_penalty_article(article: dict[str, Any]) -> bool:
    # Be conservative before the final verifier: broad legal articles may mention
    # enforcement/penalties in the body while still being useful for non-penalty
    # questions. Only use law name and article title for this deterministic drop.
    text = " ".join([norm(article.get("law_name", "")), norm(article.get("dieu_title", ""))])
    return bool(PENALTY_LAW_OR_TITLE_RE.search(text))


def should_drop_penalty(question: str, article: dict[str, Any]) -> tuple[bool, str]:
    if not is_penalty_article(article):
        return False, ""
    q = norm(question)
    if PENALTY_QUESTION_RE.search(q):
        return False, "question_penalty_signal"
    if NON_PENALTY_QUESTION_RE.search(q):
        return True, "non_penalty_question_with_penalty_article"
    return False, "ambiguous_question"


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
    parser = argparse.ArgumentParser(description="Apply deterministic cleanup before final collective verifier.")
    parser.add_argument(
        "--input-diagnostics",
        default="outputs/diagnostics_stage1_alias_intents_b6_c1800_stage1only_full_w12.json",
    )
    parser.add_argument("--corpus", default="corpus/data/corpus_clean_asof_20260301.json")
    parser.add_argument("--status-csv", default="outputs/vbpl_full_corpus_status.csv")
    parser.add_argument("--output-dir", default="outputs/stage1_cleaned_before_final_collective")
    parser.add_argument("--fallback-penalty-if-empty", action="store_true", default=True)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip status-CSV filtering when the corpus/index already enforces the as-of cutoff.",
    )
    parser.add_argument("--strict-errors", action="store_true")
    args = parser.parse_args()

    rows = read_json(Path(args.input_diagnostics))
    lookup = ArticleLookup(args.corpus)
    class_by_law = {} if args.skip_invalid else load_effective_classes(Path(args.status_csv))

    out_rows: list[dict[str, Any]] = []
    before_sizes: list[int] = []
    after_invalid_sizes: list[int] = []
    after_sizes: list[int] = []
    invalid_drop_classes: Counter[str] = Counter()
    penalty_drop_reasons: Counter[str] = Counter()
    changed_invalid = 0
    changed_penalty = 0
    missing_lookup = 0
    penalty_empty_fallback = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        original = dedupe_keep_order(row.get("final_article_ids", []))
        before_sizes.append(len(original))

        after_invalid: list[str] = []
        invalid_dropped: list[dict[str, str]] = []
        for article_id in original:
            class_name = class_by_law.get(law_id_from_article(article_id), "unknown")
            if class_name in INVALID_CLASSES:
                invalid_dropped.append({"article_id": article_id, "class": class_name})
                invalid_drop_classes[class_name] += 1
            else:
                after_invalid.append(article_id)
        if invalid_dropped:
            changed_invalid += 1
        after_invalid_sizes.append(len(after_invalid))

        final: list[str] = []
        penalty_dropped: list[dict[str, str]] = []
        kept_penalty: list[dict[str, str]] = []
        for article_id in after_invalid:
            article = lookup.get(article_id)
            if article is None:
                if args.strict_errors:
                    raise KeyError(f"Article is missing from corpus lookup: {article_id}")
                missing_lookup += 1
                final.append(article_id)
                continue
            drop, reason = should_drop_penalty(row.get("question", ""), article)
            if drop:
                penalty_dropped.append(
                    {
                        "article_id": article_id,
                        "reason": reason,
                        "title": article.get("dieu_title", ""),
                    }
                )
                penalty_drop_reasons[reason] += 1
            else:
                final.append(article_id)
                if reason:
                    kept_penalty.append(
                        {
                            "article_id": article_id,
                            "reason": reason,
                            "title": article.get("dieu_title", ""),
                        }
                    )

        penalty_fallback_used = False
        if args.fallback_penalty_if_empty and after_invalid and not final:
            final = after_invalid
            penalty_dropped = []
            penalty_fallback_used = True
            penalty_empty_fallback += 1

        if penalty_dropped:
            changed_penalty += 1
            if len(examples) < 25:
                examples.append(
                    {
                        "id": row.get("id"),
                        "question": row.get("question", ""),
                        "before": original,
                        "after_invalid": after_invalid,
                        "penalty_dropped": penalty_dropped,
                        "kept_penalty": kept_penalty[:8],
                        "after": final,
                    }
                )

        final = dedupe_keep_order(final)
        after_sizes.append(len(final))
        out_rows.append(
            {
                **row,
                "final_article_ids_before_deterministic_cleanup": original,
                "final_article_ids_after_invalid_drop": after_invalid,
                "final_article_ids": final,
                "deterministic_cleanup": {
                    "invalid_classes": sorted(INVALID_CLASSES),
                    "invalid_dropped": invalid_dropped,
                    "penalty_dropped": penalty_dropped,
                    "penalty_empty_fallback_used": penalty_fallback_used,
                },
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    save_submission([to_submission_row(row) for row in out_rows], output_dir)

    summary = {
        "input_diagnostics": args.input_diagnostics,
        "corpus": args.corpus,
        "status_csv": args.status_csv,
        "questions": len(out_rows),
        "changed_invalid_questions": changed_invalid,
        "changed_penalty_questions": changed_penalty,
        "invalid_drop_counts": dict(invalid_drop_classes.most_common()),
        "penalty_drop_counts": dict(penalty_drop_reasons.most_common()),
        "missing_lookup": missing_lookup,
        "penalty_empty_fallback": penalty_empty_fallback,
        "mean_before": statistics.mean(before_sizes) if before_sizes else 0,
        "mean_after_invalid": statistics.mean(after_invalid_sizes) if after_invalid_sizes else 0,
        "mean_after": statistics.mean(after_sizes) if after_sizes else 0,
        "min_after": min(after_sizes) if after_sizes else 0,
        "max_after": max(after_sizes) if after_sizes else 0,
        "empty_after": sum(1 for size in after_sizes if size == 0),
        "examples": examples,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
