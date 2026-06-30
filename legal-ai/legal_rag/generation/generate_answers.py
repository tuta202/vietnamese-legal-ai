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


def answer_is_valid(answer: str) -> bool:
    return bool(str(answer or "").strip() and _CITATION_RE.search(str(answer)))


def generate_one(config: RetrievalConfig, job: dict[str, Any]) -> dict[str, Any]:
    answer = get_worker_generator(config).generate(job["question"], job["articles"]).strip()
    if not answer_is_valid(answer):
        raise ValueError("Generated answer is empty or does not contain an 'Điều X' citation")
    return {
        "id": job["id"],
        "signature": job["signature"],
        "answer": answer,
        "article_count": len(job["article_refs"]),
        "prompt_version": PROMPT_VERSION,
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
            "signature": signature,
        }
        cached_row = cached.get(question_id)
        if (
            cached_row
            and cached_row.get("signature") == signature
            and answer_is_valid(cached_row.get("answer", ""))
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
        if generated is None or not answer_is_valid(generated.get("answer", "")):
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
    log.info("Final answer submission complete: %s", zip_path)


if __name__ == "__main__":
    main()
