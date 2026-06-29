from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from legal_rag.retrieval.config import load_config, validate_config
from legal_rag.retrieval.intent_decomposer import LegalIntentDecomposer


log = logging.getLogger("legal_rag.retrieval.query_analysis")
_THREAD_LOCAL = threading.local()


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(line)
            file.flush()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_jsonl(path: Path, required_fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw in enumerate(file, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Ignoring malformed cache line %s:%d", path, line_number)
                continue
            qid = row.get("id")
            if qid is None or not all(row.get(field) for field in required_fields):
                continue
            records[str(qid)] = row
    return records


def _make_rewriter(config):
    if config.backend == "vertex_ai":
        from legal_rag.backends.vertex import VertexQueryRewriter

        return VertexQueryRewriter(config, mock=False)
    if config.backend == "gpu":
        from legal_rag.backends.gpu import make_gpu_components

        _, rewriter, _, _ = make_gpu_components(config, mock=False)
        return rewriter
    raise ValueError(f"Unsupported backend: {config.backend}")


class WorkerContext:
    def __init__(self, config) -> None:
        self.rewriter = _make_rewriter(config)
        self.decomposer = LegalIntentDecomposer(
            config,
            mock=False,
            chat_complete=getattr(self.rewriter, "_chat_complete", None),
        )


def _get_context(config) -> WorkerContext:
    context = getattr(_THREAD_LOCAL, "context", None)
    if context is None:
        context = WorkerContext(config)
        _THREAD_LOCAL.context = context
    return context


def _rewrite_one(config, question: dict[str, Any]) -> dict[str, Any]:
    result = _get_context(config).rewriter.rewrite_strict(question["question"])
    rewritten_query = str(result.get("rewritten_query") or "").strip()
    topic_description = str(result.get("topic_description") or "").strip()
    if not rewritten_query or not topic_description:
        raise ValueError("rewrite output is missing rewritten_query or topic_description")
    return {
        "id": question["id"],
        "question": question["question"],
        "rewritten_query": rewritten_query,
        "topic_description": topic_description,
    }


def _decompose_one(config, question: dict[str, Any]) -> dict[str, Any]:
    analysis = _get_context(config).decomposer.decompose_strict(question["question"])
    intents = [str(value).strip() for value in analysis.intents if str(value).strip()]
    if not 1 <= len(intents) <= 6:
        raise ValueError(f"intent decomposition returned {len(intents)} intents; expected 1..6")
    return {
        "id": question["id"],
        "question": question["question"],
        "legal_intents": intents,
        "num_intents": len(intents),
        "is_multihop": bool(getattr(analysis, "is_multihop", len(intents) > 1)),
    }


def _run_jobs(
    *,
    config,
    questions: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    cache_path: Path,
    workers: int,
    function,
    label: str,
) -> None:
    todo = [question for question in questions if str(question["id"]) not in records]
    if not todo:
        log.info("%s already complete: %d/%d", label, len(records), len(questions))
        return
    lock = threading.Lock()
    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, config, question): question for question in todo}
        for future in as_completed(futures):
            question = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                log.error("%s Q%s failed and remains uncached: %s", label, question["id"], exc)
                completed += 1
                continue
            records[str(row["id"])] = row
            _append_jsonl(cache_path, row, lock)
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                elapsed = max(time.time() - started, 1e-6)
                log.info("%s %d/%d cached (%.2f q/s)", label, len(records), len(questions), completed / elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict two-call legal query analyzer with resume caches.")
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input", default="../R2AIStage1DATA.json")
    parser.add_argument("--rewrite-cache", required=True)
    parser.add_argument("--intent-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    config = load_config(args.config)
    problems = validate_config(config)
    if problems:
        raise ValueError(f"Invalid config: {'; '.join(problems)}")

    rewrite_cache = Path(args.rewrite_cache)
    intent_cache = Path(args.intent_cache)
    if not args.resume:
        rewrite_cache.unlink(missing_ok=True)
        intent_cache.unlink(missing_ok=True)
    rewrites = _load_jsonl(rewrite_cache, ("rewritten_query", "topic_description")) if args.resume else {}
    intents = _load_jsonl(intent_cache, ("legal_intents",)) if args.resume else {}

    _run_jobs(
        config=config,
        questions=questions,
        records=rewrites,
        cache_path=rewrite_cache,
        workers=max(1, args.workers),
        function=_rewrite_one,
        label="rewrite",
    )
    _run_jobs(
        config=config,
        questions=questions,
        records=intents,
        cache_path=intent_cache,
        workers=max(1, args.workers),
        function=_decompose_one,
        label="decompose",
    )

    missing_rewrites = [q["id"] for q in questions if str(q["id"]) not in rewrites]
    missing_intents = [q["id"] for q in questions if str(q["id"]) not in intents]
    if missing_rewrites or missing_intents:
        raise RuntimeError(
            f"analysis incomplete: missing_rewrites={missing_rewrites[:20]} "
            f"missing_intents={missing_intents[:20]}"
        )

    output = []
    for question in questions:
        key = str(question["id"])
        output.append({**rewrites[key], **{k: v for k, v in intents[key].items() if k not in {"id", "question"}}})
    _atomic_write_json(Path(args.output), output)
    log.info("Analysis complete: %d rows", len(output))


if __name__ == "__main__":
    main()
