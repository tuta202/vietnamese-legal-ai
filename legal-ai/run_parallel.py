"""
run_parallel.py — concurrent batch runner for the pipeline.

Each question is I/O-bound (Gemini + Qdrant API latency), so running many
questions at once collapses wall-time from ~N*latency to ~N*latency/workers
(until the Gemini rate limit becomes the bottleneck; transient 429s are already
retried with backoff inside the Vertex components).

Does NOT modify pipeline.py — it reuses LegalAIPipeline.process_question
unchanged (which already isolates per-question failures into a fallback entry).
A single shared pipeline instance is safe: BM25 is read-only and the Gemini /
Qdrant clients are used concurrently; the lazy clients are warmed single-thread
on the first question before the pool starts.

CLI:
    python run_parallel.py --config config_vertex.yaml \
        --input R2AIStage1DATA.json --output results.json --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from pipeline import LegalAIPipeline  # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_parallel")
log.setLevel(logging.INFO)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Concurrent pipeline batch runner")
    ap.add_argument("--config", default="config_vertex.yaml")
    ap.add_argument("--input", required=True, help="JSON [{id, question}, ...]")
    ap.add_argument("--output", default="results.json")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent questions (bounded by Gemini rate limits)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N questions (dry-run/benchmark)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed entries from an existing --output file "
                         "and only process the missing/fallback ones")
    ap.add_argument("--scores-detail", default=None,
                    help="path for the per-question scored-pool JSONL "
                         "(default: scores_detail.jsonl next to --output). Only "
                         "written when the backend reranker supports it (garden).")
    args = ap.parse_args()

    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    total = len(questions)
    out = Path(args.output)
    lock = threading.Lock()

    def _persist() -> None:
        """Atomically write whatever results we have so far (input order)."""
        ordered = [results[q["id"]] for q in questions if q["id"] in results]
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)

    # Resume: load prior output; an entry counts as DONE only if it has
    # relevant_articles (fallback/empty entries are reprocessed).
    results: dict = {}
    if args.resume and out.exists():
        try:
            for r in json.loads(out.read_text(encoding="utf-8")):
                if r.get("id") is not None and r.get("relevant_articles"):
                    results[r["id"]] = r
            log.info("Resume: %d/%d already complete in %s", len(results), total, out)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Resume failed to read %s (%s) — starting fresh", out, e)

    todo = [q for q in questions if q["id"] not in results]
    log.info("Loaded %d questions; %d to process; workers=%d", total, len(todo), args.workers)

    pipeline = LegalAIPipeline(config_path=args.config, mock=False)

    # Scored-pool detail JSONL (for offline threshold sweeping). Only emitted when
    # the reranker exposes detail (garden 2-tier). Append-safe; a separate lock so
    # it never nests inside the results lock.
    detail_path = Path(args.scores_detail) if args.scores_detail \
        else out.with_name("scores_detail.jsonl")
    write_detail = hasattr(pipeline._reranker, "rerank_detailed")
    detail_lock = threading.Lock()
    if write_detail and not (args.resume and detail_path.exists()):
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text("", encoding="utf-8")   # fresh run → truncate

    def _emit_detail(detail: dict | None) -> None:
        if not (write_detail and detail):
            return
        with detail_lock:
            with detail_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    def _work(q: dict) -> tuple[dict, dict | None]:
        if write_detail:
            return pipeline.process_question_detailed(q["id"], q["question"])
        return pipeline.process_question(q["id"], q["question"]), None

    t0 = time.time()
    if not todo:
        log.info("Nothing to process — all complete.")
        _persist()
        return

    # Warm up lazy clients single-threaded on the first pending question, so
    # concurrent threads never race the lazy-init.
    first = todo[0]
    entry, detail = _work(first)
    results[first["id"]] = entry
    _emit_detail(detail)
    log.info("Warmup question done in %.1fs (clients initialised)", time.time() - t0)
    _persist()

    rest = todo[1:]
    processed = 1
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_work, q): q["id"] for q in rest}
        for fut in as_completed(futures):
            entry, detail = fut.result()
            _emit_detail(detail)           # own lock, outside the results lock
            with lock:
                results[entry["id"]] = entry
                processed += 1
                if processed % 50 == 0 or processed == len(todo):
                    rate = processed / (time.time() - t0)
                    eta = (len(todo) - processed) / rate if rate else 0
                    log.info("  %d/%d done  (%.2f q/s, ETA %.0f min)",
                             len(results), total, rate, eta / 60)
                    _persist()   # checkpoint to disk

    # Retry stragglers. A fallback entry (empty relevant_articles) usually means
    # a transient Qdrant/LLM timeout under concurrency. Retry in rounds at a
    # gentler concurrency so a large backlog (from high --workers) clears fast.
    retry_workers = max(1, min(4, args.workers))
    for rnd in range(1, 4):
        failed = [q for q in questions if not results.get(q["id"], {}).get("relevant_articles")]
        if not failed:
            break
        log.info("Retry round %d: %d straggler(s) at %d workers…",
                 rnd, len(failed), retry_workers)
        with ThreadPoolExecutor(max_workers=retry_workers) as ex:
            for entry, detail in ex.map(_work, failed):
                _emit_detail(detail)
                results[entry["id"]] = entry
        _persist()
    still = sum(1 for q in questions if not results.get(q["id"], {}).get("relevant_articles"))
    if still:
        log.info("After retries: %d still empty (kept as fallback entries)", still)

    # Final write — guarantee every id is present (fallback entry if missing).
    for q in questions:
        results.setdefault(q["id"], {
            "id": q["id"], "question": q["question"], "answer": "",
            "relevant_docs": [], "relevant_articles": [],
        })
    _persist()
    elapsed = time.time() - t0
    log.info("Wrote %d results → %s in %.1f min (%.2f q/s this run)",
             len(results), out, elapsed / 60, processed / elapsed)


if __name__ == "__main__":
    main()
