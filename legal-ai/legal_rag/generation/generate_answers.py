"""Resumable final-answer generation from the verified article submission."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from legal_rag.common.article_lookup import ArticleLookup
from legal_rag.generation.prompt_builder import PROMPT_VERSION
from legal_rag.output.submission import save_submission, validate_submission
from legal_rag.retrieval.config import RetrievalConfig, load_config, validate_config


log = logging.getLogger("legal_rag.generation.generate_answers")
_CACHE_LOCK = threading.Lock()
_WORKER_LOCAL = threading.local()
_CITATION_RE = re.compile(r"Điều\s+\d+", re.UNICODE)
_ARTICLE_NUMBER_RE = re.compile(r"Điều\s+(\d+[a-zđ]?)", re.IGNORECASE | re.UNICODE)
_LEGAL_ID_RE = re.compile(
    r"\b(?:\d{1,4}(?:/\d{4})?/[0-9A-ZÀ-ỸĐ-]+(?:-[0-9A-ZÀ-ỸĐ-]+)*"
    r"|\d{1,4}-[0-9A-ZÀ-ỸĐ-]+/[0-9A-ZÀ-ỸĐ-]+"
    r"|\d{1,4}-[A-ZÀ-ỸĐ][0-9A-ZÀ-ỸĐ-]*)\b",
    re.IGNORECASE | re.UNICODE,
)
_CITATION_CLAUSE_MAX_CHARS = 300
_DIRECT_CITATION_MAX_DISTANCE = 100
_CITATION_BOUNDARY_RE = re.compile(r"[\n.;]")
_STANDARD_LEGAL_ID_RE = re.compile(
    r"^(?:\d{1,4}/\d{4}/(?:QH\d+|NĐ-CP|ND-CP|TT[A-ZÀ-ỸĐ0-9-]*|"
    r"TTLT[A-ZÀ-ỸĐ0-9-]*|NQ[A-ZÀ-ỸĐ0-9-]*|UBTVQH[A-ZÀ-ỸĐ0-9-]*)"
    r"|\d{1,4}-(?:CP|HĐBT|TC[A-ZÀ-ỸĐ0-9/-]*))$",
    re.IGNORECASE | re.UNICODE,
)
_INVALID_PAIR_DETAIL_RE = re.compile(r"([0-9]+[a-z\u0111]?)\s+\(([^)]+)\)", re.IGNORECASE)
_EVIDENCE_REFERENCE_MAX_DISTANCE = 300
_ARTICLE_HEADER_SCAN_CHARS = 240


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def article_doc_ref(article_ref: str) -> str:
    doc, separator, _article = str(article_ref).rpartition("|")
    return doc if separator else str(article_ref)


def make_generator(config: RetrievalConfig):
    if config.backend == "vertex_ai":
        from legal_rag.backends.vertex import VertexGenerator

        return VertexGenerator(config, mock=False)
    if config.backend == "gpu":
        from legal_rag.backends.gpu import GpuGenerator

        return GpuGenerator(config, mock=False)
    raise ValueError(f"Unsupported backend: {config.backend}")


def get_worker_generator(config: RetrievalConfig):
    generator = getattr(_WORKER_LOCAL, "generator", None)
    if generator is None:
        generator = make_generator(config)
        _WORKER_LOCAL.generator = generator
    return generator


def generation_signature(
    config: RetrievalConfig,
    question: str,
    article_refs: list[str],
    articles: list[dict[str, Any]],
) -> str:
    generator = config.generator
    model = config.gpu.llm_model if config.backend == "gpu" else config.vllm.model
    material = {
        "prompt_version": PROMPT_VERSION,
        "backend": config.backend,
        "model": model,
        "generator": {
            "temperature": generator.temperature,
            "max_tokens": generator.max_tokens,
            "top_p": generator.top_p,
            "max_articles": generator.max_articles,
            "content_max_chars": generator.content_max_chars,
            "total_content_max_chars": generator.total_content_max_chars,
        },
        "question": question,
        "article_refs": article_refs,
        "articles": [
            {
                "law_id": article.get("law_id", ""),
                "law_type": article.get("law_type", ""),
                "law_name": article.get("law_name", ""),
                "dieu_number": article.get("dieu_number", ""),
                "dieu_title": article.get("dieu_title", ""),
                "content": article.get("content", ""),
            }
            for article in articles
        ],
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            try:
                row = json.loads(raw)
                question_id = str(row["id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            rows[question_id] = row
    return rows


def append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()


def acquire_cache_lock(cache_path: Path) -> None:
    lock_path = cache_path.with_suffix(cache_path.suffix + ".running")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            text = lock_path.read_text(encoding="utf-8")
            pid = int(next(line for line in text.splitlines() if line.startswith("pid=")).split("=", 1)[1])
            os.kill(pid, 0)
        except (OSError, ValueError, StopIteration):
            lock_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Generation cache is already active with PID {pid}: {lock_path}")
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(f"pid={os.getpid()}\n")
    atexit.register(lambda: lock_path.unlink(missing_ok=True))


def prepare_cache(path: Path, resume: bool) -> None:
    if resume or not path.exists() or path.stat().st_size == 0:
        return
    backup = path.with_suffix(path.suffix + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.move(path, backup)
    log.warning("Fresh generation run requested; moved existing cache to %s", backup)


def normalize_citation_part(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).upper()
    return re.sub(r"\s+", "", text)


def article_citation_pair(article: dict[str, Any]) -> tuple[str, str] | None:
    number_match = _ARTICLE_NUMBER_RE.search(str(article.get("dieu_number") or ""))
    law_id = normalize_citation_part(article.get("law_id"))
    if not number_match or not law_id:
        return None
    return number_match.group(1).casefold(), law_id


def invalid_pairs_from_detail(detail: str) -> set[tuple[str, str]]:
    return {
        (match.group(1).casefold(), normalize_citation_part(match.group(2)))
        for match in _INVALID_PAIR_DETAIL_RE.finditer(detail)
    }


def evidence_explanation_for_invalid_pair(
    pair: tuple[str, str],
    articles: list[dict[str, Any]],
) -> str | None:
    """Return why an outside-input citation may still be evidence-derived.

    This deliberately requires textual evidence. Merely sharing a law or an
    article number is not enough to downgrade a hard failure.
    """
    article_number, law_id = pair
    for article in articles:
        article_law_id = normalize_citation_part(article.get("law_id"))
        if article_law_id == law_id:
            metadata_pair = article_citation_pair(article)
            for header in (
                str(article.get("dieu_title") or ""),
                str(article.get("content") or "")[:_ARTICLE_HEADER_SCAN_CHARS],
            ):
                header_match = _ARTICLE_NUMBER_RE.match(header.lstrip())
                if (
                    header_match
                    and header_match.group(1).casefold() == article_number
                    and metadata_pair != pair
                ):
                    return "supplied article metadata conflicts with its title/content header"

        evidence_text = "\n".join(
            [
                str(article.get("dieu_title") or ""),
                str(article.get("content") or ""),
            ]
        )
        for mention in _ARTICLE_NUMBER_RE.finditer(evidence_text):
            if mention.group(1).casefold() != article_number:
                continue
            start = max(0, mention.start() - _EVIDENCE_REFERENCE_MAX_DISTANCE)
            end = min(len(evidence_text), mention.end() + _EVIDENCE_REFERENCE_MAX_DISTANCE)
            if article_law_id == law_id or law_id in normalize_citation_part(
                evidence_text[start:end]
            ):
                return "citation is an explicit cross-reference in supplied evidence"
    return None


def validate_answer_citations(
    answer: str,
    articles: list[dict[str, Any]],
    known_law_ids: set[str] | None = None,
) -> tuple[bool, str]:
    """Validate explicit ``Điều X ... law_id`` citations against supplied evidence.

    Wording and punctuation may vary, but an explicit legal ID appearing near an
    article mention must form an allowed ``(article_number, law_id)`` pair. Later
    shorthand mentions such as ``khoản 2 Điều 5`` are tolerated after at least
    one fully qualified citation has established a valid legal basis.
    """
    text = str(answer or "").strip()
    if not text:
        return False, "answer is empty"

    allowed_pairs = {
        pair for article in articles if (pair := article_citation_pair(article)) is not None
    }
    allowed_law_ids = {law_id for _number, law_id in allowed_pairs}
    allowed_article_numbers = {number for number, _law_id in allowed_pairs}
    normalized_known_law_ids = {
        normalize_citation_part(law_id) for law_id in (known_law_ids or set())
    }
    if not allowed_pairs:
        return False, "supplied articles do not contain valid citation metadata"

    article_mentions = list(_ARTICLE_NUMBER_RE.finditer(text))
    if not article_mentions:
        return False, "answer does not contain an 'Điều X' citation"

    valid_pairs: set[tuple[str, str]] = set()
    invalid_pairs: set[tuple[str, str]] = set()
    invalid_unqualified_numbers: set[str] = set()
    for mention_index, mention in enumerate(article_mentions):
        article_number = mention.group(1).casefold()
        left_limit = max(0, mention.start() - _CITATION_CLAUSE_MAX_CHARS)
        right_limit = min(len(text), mention.end() + _CITATION_CLAUSE_MAX_CHARS)
        left_text = text[left_limit : mention.start()]
        right_text = text[mention.end() : right_limit]
        left_boundaries = list(_CITATION_BOUNDARY_RE.finditer(left_text))
        if left_boundaries:
            left_limit += left_boundaries[-1].end()
        right_boundary = _CITATION_BOUNDARY_RE.search(right_text)
        if right_boundary:
            right_limit = mention.end() + right_boundary.start()

        previous_mention_end = (
            article_mentions[mention_index - 1].end() if mention_index > 0 else left_limit
        )
        next_mention_start = (
            article_mentions[mention_index + 1].start()
            if mention_index + 1 < len(article_mentions)
            else right_limit
        )
        before_start = max(left_limit, previous_mention_end)
        after_end = min(right_limit, next_mention_start)

        def recognized_ids(segment: str) -> list[re.Match[str]]:
            matches: list[re.Match[str]] = []
            for id_match in _LEGAL_ID_RE.finditer(segment):
                normalized_id = normalize_citation_part(id_match.group(0))
                if (
                    normalized_id in allowed_law_ids
                    or normalized_id in normalized_known_law_ids
                    or _STANDARD_LEGAL_ID_RE.fullmatch(normalized_id)
                ):
                    matches.append(id_match)
            return matches

        before_segment = text[before_start : mention.start()]
        after_segment = text[mention.end() : after_end]
        before_ids = [
            id_match
            for id_match in recognized_ids(before_segment)
            if len(before_segment) - id_match.end() <= _DIRECT_CITATION_MAX_DISTANCE
        ]
        after_ids = [
            id_match
            for id_match in recognized_ids(after_segment)
            if id_match.start() <= _DIRECT_CITATION_MAX_DISTANCE
        ]
        parenthesized_article = text[before_start : mention.start()].rstrip().endswith("(")
        if parenthesized_article and before_ids:
            selected_id = before_ids[-1]
        elif after_ids:
            selected_id = after_ids[0]
        elif before_ids:
            selected_id = before_ids[-1]
        else:
            if article_number not in allowed_article_numbers:
                invalid_unqualified_numbers.add(article_number)
            continue
        pair = article_number, normalize_citation_part(selected_id.group(0))
        if pair in allowed_pairs:
            valid_pairs.add(pair)
        else:
            invalid_pairs.add(pair)

    if invalid_pairs:
        rendered = ", ".join(f"Điều {number} ({law_id})" for number, law_id in sorted(invalid_pairs))
        return False, f"citation is not present in supplied articles: {rendered}"
    if invalid_unqualified_numbers:
        rendered = ", ".join(f"Điều {number}" for number in sorted(invalid_unqualified_numbers))
        return False, f"unqualified article is not present in supplied articles: {rendered}"
    if not valid_pairs:
        return False, "answer has no fully qualified citation from supplied articles"
    return True, "ok"


def answer_requires_regeneration(
    answer: str,
    articles: list[dict[str, Any]],
    known_law_ids: set[str] | None = None,
) -> tuple[bool, str]:
    """Request another generation only for a clearly unsupported citation.

    Ambiguous citations are accepted. A citation is considered clearly
    hallucinated only when it is an explicit article/law pair outside the
    supplied input and is not explained by a cross-reference or metadata
    conflict inside that evidence.
    """
    if not str(answer or "").strip():
        return True, "answer is empty"

    valid, detail = validate_answer_citations(answer, articles, known_law_ids)
    if valid:
        return False, detail
    if detail.startswith("citation is not present in supplied articles"):
        invalid_pairs = invalid_pairs_from_detail(detail)
        clearly_hallucinated = {
            pair
            for pair in invalid_pairs
            if evidence_explanation_for_invalid_pair(pair, articles) is None
        }
        if clearly_hallucinated:
            return True, detail
    return False, detail


def answer_is_valid(
    answer: str,
    articles: list[dict[str, Any]] | None = None,
    known_law_ids: set[str] | None = None,
) -> bool:
    if articles is None:
        return bool(str(answer or "").strip() and _CITATION_RE.search(str(answer)))
    requires_regeneration, _detail = answer_requires_regeneration(
        answer,
        articles,
        known_law_ids,
    )
    return not requires_regeneration


def allowed_citation_labels(articles: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for article in articles:
        pair = article_citation_pair(article)
        if pair is None:
            continue
        article_number = str(article.get("dieu_number") or "").strip()
        law_name = str(article.get("law_name") or "").strip()
        law_id = str(article.get("law_id") or "").strip()
        labels.append(" ".join(part for part in (article_number, law_name, f"({law_id})") if part))
    return labels


def cached_answer_is_acceptable(
    cached_row: dict[str, Any],
    articles: list[dict[str, Any]],
    known_law_ids: set[str] | None = None,
) -> bool:
    answer = str(cached_row.get("answer") or "").strip()
    if not answer:
        return False
    if cached_row.get("citation_repair_applied") is True:
        return True
    return answer_is_valid(answer, articles, known_law_ids)


def generate_one(config: RetrievalConfig, job: dict[str, Any]) -> dict[str, Any]:
    generator = get_worker_generator(config)
    answer = generator.generate(job["question"], job["articles"]).strip()
    citation_repair_applied = False
    requires_regeneration, detail = answer_requires_regeneration(
        answer,
        job["articles"],
        job.get("known_law_ids"),
    )
    if requires_regeneration:
        citation_repair_applied = True
        answer = generator.repair_citations(
            job["question"],
            job["articles"],
            answer,
            detail,
            allowed_citation_labels(job["articles"]),
        ).strip()
        if not answer:
            raise ValueError("Citation repair returned an empty answer")
    return {
        "id": job["id"],
        "signature": job["signature"],
        "answer": answer,
        "article_count": len(job["article_refs"]),
        "prompt_version": PROMPT_VERSION,
        "citation_repair_applied": citation_repair_applied,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Generate final grounded answers from verified articles.")
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input", required=True, help="Stage 10 results.json")
    parser.add_argument("--corpus", default="corpus/data/corpus_clean_asof_20260301.json")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--errors", default="")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-errors", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        raise ValueError("Invalid config: " + "; ".join(errors))

    input_path = Path(args.input)
    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir)
    errors_path = Path(args.errors) if args.errors else output_dir / "errors.json"
    input_rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(input_rows, list):
        raise ValueError("Generation input must be a JSON list")
    ids = [str(row.get("id")) for row in input_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Generation input contains duplicate question IDs")

    prepare_cache(cache_path, args.resume)
    acquire_cache_lock(cache_path)
    lookup = ArticleLookup(args.corpus)
    cached = load_cache(cache_path)
    valid_cached: dict[str, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []

    for row in input_rows:
        question_id = str(row["id"])
        question = str(row.get("question") or "").strip()
        article_refs = dedupe_keep_order(row.get("relevant_articles", []))
        if not question or not article_refs:
            raise ValueError(f"Q{question_id} has an empty question or article list")
        articles = [lookup.require(article_ref) for article_ref in article_refs]
        signature = generation_signature(config, question, article_refs, articles)
        job = {
            "id": row["id"],
            "question": question,
            "article_refs": article_refs,
            "articles": articles,
            "known_law_ids": lookup.law_ids,
            "signature": signature,
        }
        cached_row = cached.get(question_id)
        if (
            cached_row
            and cached_row.get("signature") == signature
            and cached_answer_is_acceptable(
                cached_row,
                articles,
                lookup.law_ids,
            )
        ):
            valid_cached[question_id] = cached_row
        else:
            jobs.append(job)

    log.info(
        "Final generation rows=%d cached_valid=%d todo=%d workers=%d prompt=%s",
        len(input_rows),
        len(input_rows) - len(jobs),
        len(jobs),
        args.workers,
        PROMPT_VERSION,
    )

    run_errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(generate_one, config, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                generated = future.result()
            except Exception as exc:
                run_errors.append(
                    {"id": str(job["id"]), "error": f"{type(exc).__name__}: {exc}"}
                )
                log.error("Q%s generation failed: %s", job["id"], exc)
                continue
            append_cache(cache_path, generated)
            cached[str(generated["id"])] = generated
            valid_cached[str(generated["id"])] = generated

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(json.dumps(run_errors, ensure_ascii=False, indent=2), encoding="utf-8")

    output_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in input_rows:
        question_id = str(row["id"])
        generated = valid_cached.get(question_id)
        articles = [lookup.require(article_ref) for article_ref in row.get("relevant_articles", [])]
        if generated is None or not cached_answer_is_acceptable(
            generated, articles, lookup.law_ids
        ):
            missing.append(question_id)
            continue
        article_refs = dedupe_keep_order(row.get("relevant_articles", []))
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": generated["answer"],
                "relevant_docs": dedupe_keep_order(
                    row.get("relevant_docs", [])
                    or [article_doc_ref(article) for article in article_refs]
                ),
                "relevant_articles": article_refs,
            }
        )

    if missing or run_errors:
        log.error("Generation incomplete: missing=%d errors_this_pass=%d", len(missing), len(run_errors))
        raise SystemExit(1)

    valid, validation_errors = validate_submission(output_rows)
    if not valid:
        raise ValueError("Generated submission is invalid: " + "; ".join(validation_errors[:10]))
    zip_path = save_submission(output_rows, output_dir)
    (output_dir / "generation_metadata.json").write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "row_count": len(output_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Final answer submission complete: %s", zip_path)


if __name__ == "__main__":
    main()
