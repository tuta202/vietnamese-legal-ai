from __future__ import annotations

import argparse
import json
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from legal_rag.common.article_lookup import ArticleLookup


DEFAULT_INPUT = Path("outputs/submission_final_collective_no_preserve_top1/results.json")
DEFAULT_CORPUS = Path("corpus/data/corpus_clean_asof_20260301.json")
DEFAULT_OUTPUT_DIR = Path("outputs/submission_final_no_top1_enforcement_role_gate")

PENALTY_LAW_RE = re.compile(
    r"xử phạt vi phạm hành chính",
    re.I,
)
SPECIFIC_OFFENCE_TITLE_RE = re.compile(
    r"^(hành vi )?vi phạm\b|^xử phạt hành vi|xử phạt đối với|phạt tiền đối với",
    re.I,
)
COERCION_TITLE_RE = re.compile(
    r"^(?:các )?(?:trường hợp bị |đối tượng áp dụng )?cưỡng chế\b|"
    r"^biện pháp cưỡng chế\b|^trình tự.{0,50}cưỡng chế\b|"
    r"^thi hành quyết định.{0,80}cưỡng chế\b|"
    r"giao dịch điện tử trong công tác quản lý nợ và cưỡng chế nợ thuế",
    re.I,
)
OPERATIVE_PENALTY_RE = re.compile(
    r"phạt cảnh cáo|phạt tiền|bị xử phạt|mức phạt|hình thức xử phạt",
    re.I,
)

EXPLICIT_PENALTY_QUESTION_RE = re.compile(
    r"xử phạt|bị phạt|mức phạt|phạt tiền|tiền phạt|hình thức xử phạt|"
    r"chế tài|vi phạm hành chính|biện pháp khắc phục hậu quả",
    re.I,
)
IMPLICIT_ENFORCEMENT_QUESTION_RE = re.compile(
    r"bị xử lý(?: như thế nào| ra sao)?|xử lý (?:như thế nào|ra sao)|"
    r"phải khắc phục|khắc phục hậu quả|rủi ro (?:pháp lý|bị xử phạt)|"
    r"hậu quả pháp lý|trách nhiệm pháp lý|khắc phục sai phạm|xử lý vi phạm",
    re.I,
)
VIOLATION_DETERMINATION_RE = re.compile(
    r"có (?:bị coi là )?vi phạm|hành vi (?:này )?vi phạm|"
    r"vi phạm (?:pháp luật|quy định|luật)|trái (?:pháp luật|quy định)|"
    r"bị nghiêm cấm|(?:được )?xác định là vi phạm",
    re.I,
)
COERCION_QUESTION_RE = re.compile(
    r"cưỡng chế|nợ thuế|thu hồi nợ|kê biên|phong tỏa tài khoản|"
    r"trích tiền từ tài khoản|thu tiền từ (?:đối tác|bên thứ ba)|"
    r"ngừng sử dụng hóa đơn|dừng làm thủ tục hải quan|"
    r"thu hồi giấy chứng nhận đăng ký doanh nghiệp do nợ thuế|"
    r"chưa nộp (?:đủ )?(?:số )?thuế|không thể nộp thuế",
    re.I,
)

# Generic provisions do not answer a concrete offence unless the question asks
# for the same legal dimension explicitly.
GENERIC_ROLE_RULES: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    (
        "penalty_form",
        re.compile(r"^(?:các )?hình thức xử phạt|hình thức xử phạt bổ sung", re.I),
        re.compile(r"hình thức xử phạt|bị xử phạt bằng hình thức|chế tài (?:nào|gì)", re.I),
    ),
    (
        "penalty_level",
        re.compile(
            r"^(?:(?:quy định về )?mức phạt tiền|mức xử phạt|"
            r"mức phạt vi phạm hành chính|xác định mức tiền phạt)",
            re.I,
        ),
        re.compile(r"mức phạt|phạt bao nhiêu|số tiền phạt|mức tiền phạt", re.I),
    ),
    (
        "penalty_authority",
        re.compile(r"^thẩm quyền (?:lập biên bản|xử phạt)|phân định thẩm quyền xử phạt", re.I),
        re.compile(
            r"ai|cơ quan nào|người nào|thẩm quyền.{0,30}(?:xử phạt|lập biên bản)|"
            r"phạt tiền tối đa|được phạt tối đa|(?:có )?quyền xử phạt",
            re.I,
        ),
    ),
    (
        "penalty_limitation",
        re.compile(r"^thời hiệu xử phạt|thời hạn được coi là chưa bị xử lý", re.I),
        re.compile(r"thời hiệu xử phạt|thời hạn xử phạt|bao lâu.{0,30}xử phạt", re.I),
    ),
    (
        "penalty_principle",
        re.compile(r"^nguyên tắc xử phạt|nguyên tắc áp dụng hình thức xử phạt", re.I),
        re.compile(r"nguyên tắc.{0,30}xử phạt", re.I),
    ),
    (
        "penalty_subject",
        re.compile(
            r"^đối tượng (?:bị xử phạt|xử phạt|áp dụng xử phạt|áp dụng cưỡng chế)",
            re.I,
        ),
        re.compile(r"ai|đối tượng nào|chủ thể nào|người nào.{0,30}(?:bị )?xử phạt", re.I),
    ),
    (
        "penalty_circumstance",
        re.compile(r"^tình tiết (?:giảm nhẹ|tăng nặng)|tình tiết giảm nhẹ, tình tiết tăng nặng", re.I),
        re.compile(r"tình tiết (?:giảm nhẹ|tăng nặng)|quy mô lớn", re.I),
    ),
    (
        "remedy",
        re.compile(r"^biện pháp khắc phục hậu quả", re.I),
        re.compile(
            r"biện pháp khắc phục hậu quả|phải khắc phục|"
            r"khắc phục (?:hậu quả )?(?:ra sao|thế nào|như thế nào)",
            re.I,
        ),
    ),
)


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def article_doc_ref(article_ref: str) -> str:
    parts = str(article_ref).split("|")
    return "|".join(parts[:2]) if len(parts) >= 2 else str(article_ref)


def generic_role(article: dict[str, Any]) -> tuple[str, re.Pattern[str]] | None:
    title = norm(article.get("dieu_title", ""))
    if re.search(r"đối với (?:các )?hành vi|vi phạm (?:quy định )?về", title, re.I):
        return None
    for role, title_re, question_re in GENERIC_ROLE_RULES:
        if title_re.search(title):
            return role, question_re
    return None


def is_coercion_article(article: dict[str, Any]) -> bool:
    title = norm(article.get("dieu_title", ""))
    return bool(COERCION_TITLE_RE.search(title))


def is_penalty_article(article: dict[str, Any]) -> bool:
    law_name = norm(article.get("law_name", ""))
    title = norm(article.get("dieu_title", ""))
    opening = norm(article.get("content", ""))[:700]
    return bool(
        (PENALTY_LAW_RE.search(law_name) and SPECIFIC_OFFENCE_TITLE_RE.search(title))
        or (SPECIFIC_OFFENCE_TITLE_RE.search(title) and OPERATIVE_PENALTY_RE.search(opening))
        or generic_role(article)
    )


def question_requests_enforcement(question: str) -> bool:
    return bool(
        EXPLICIT_PENALTY_QUESTION_RE.search(question)
        or IMPLICIT_ENFORCEMENT_QUESTION_RE.search(question)
    )


def question_requests_violation_determination(question: str) -> bool:
    return bool(VIOLATION_DETERMINATION_RE.search(question))


def drop_reason(
    question: str,
    article: dict[str, Any],
    *,
    has_non_enforcement_alternative: bool,
) -> str:
    if is_coercion_article(article):
        if not COERCION_QUESTION_RE.search(question) and has_non_enforcement_alternative:
            return "coercion_not_requested_with_alternative"
        return ""

    role = generic_role(article)
    if role is not None:
        role_name, required_question_re = role
        if not required_question_re.search(question):
            return f"generic_{role_name}_not_requested"
        return ""

    if not is_penalty_article(article):
        return ""

    if question_requests_enforcement(question) or question_requests_violation_determination(question):
        return ""

    if has_non_enforcement_alternative:
        return "specific_penalty_not_requested_with_alternative"
    return ""


def write_submission(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservatively remove penalty/coercion articles whose legal role is not requested."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--strict-errors", action="store_true")
    args = parser.parse_args()

    input_rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    lookup = ArticleLookup(args.corpus)
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    before_sizes: list[int] = []
    after_sizes: list[int] = []
    removed_by_reason: Counter[str] = Counter()
    guarded_by_reason: Counter[str] = Counter()
    missing_lookup = 0
    changed_questions = 0

    for row in input_rows:
        original = dedupe_keep_order(row.get("relevant_articles", []))
        hydrated = {article_id: lookup.get(article_id) for article_id in original}
        if args.strict_errors:
            missing = [article_id for article_id, article in hydrated.items() if article is None]
            if missing:
                raise KeyError(f"Q{row.get('id')} articles missing from corpus lookup: {missing[:5]}")
        missing_lookup += sum(article is None for article in hydrated.values())
        non_enforcement_ids = {
            article_id
            for article_id, article in hydrated.items()
            if article is not None and not is_penalty_article(article) and not is_coercion_article(article)
        }

        dropped: list[dict[str, str]] = []
        guarded: list[dict[str, str]] = []
        final: list[str] = []
        for article_id in original:
            article = hydrated.get(article_id)
            if article is None:
                final.append(article_id)
                continue
            reason = drop_reason(
                row.get("question", ""),
                article,
                has_non_enforcement_alternative=bool(non_enforcement_ids),
            )
            # Never turn a non-empty answer into an empty one.
            if reason and len(original) - len(dropped) > 1:
                dropped.append(
                    {
                        "article_id": article_id,
                        "reason": reason,
                        "title": norm(article.get("dieu_title", "")),
                    }
                )
                removed_by_reason[reason] += 1
            else:
                final.append(article_id)
                if reason:
                    guarded.append(
                        {
                            "article_id": article_id,
                            "reason": reason,
                            "title": norm(article.get("dieu_title", "")),
                        }
                    )
                    guarded_by_reason[reason] += 1

        before_sizes.append(len(original))
        after_sizes.append(len(final))
        if dropped:
            changed_questions += 1
        new_row = dict(row)
        new_row["relevant_articles"] = final
        new_row["relevant_docs"] = dedupe_keep_order([article_doc_ref(item) for item in final])
        output_rows.append(new_row)
        if dropped or guarded:
            audit_rows.append(
                {
                    "id": row.get("id"),
                    "question": row.get("question", ""),
                    "before": original,
                    "dropped": dropped,
                    "guarded_to_avoid_empty": guarded,
                    "after": final,
                }
            )

    output_dir = Path(args.output_dir)
    zip_path = write_submission(output_rows, output_dir)
    (output_dir / "audit.json").write_text(
        json.dumps(audit_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "input": args.input,
        "questions": len(input_rows),
        "changed_questions": changed_questions,
        "audited_questions": len(audit_rows),
        "removed_articles": sum(removed_by_reason.values()),
        "removed_by_reason": dict(removed_by_reason.most_common()),
        "guarded_articles": sum(guarded_by_reason.values()),
        "guarded_by_reason": dict(guarded_by_reason.most_common()),
        "missing_lookup": missing_lookup,
        "mean_before": statistics.mean(before_sizes) if before_sizes else 0,
        "mean_after": statistics.mean(after_sizes) if after_sizes else 0,
        "min_after": min(after_sizes) if after_sizes else 0,
        "max_after": max(after_sizes) if after_sizes else 0,
        "empty_after": sum(size == 0 for size in after_sizes),
        "submission_zip": str(zip_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
