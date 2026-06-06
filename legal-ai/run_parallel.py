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
    args = ap.parse_args()

    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    total = len(questions)
    log.info("Loaded %d questions; workers=%d", total, args.workers)

    pipeline = LegalAIPipeline(config_path=args.config, mock=False)

    # Warm up lazy Gemini/Qdrant clients single-threaded on the first question,
    # so concurrent threads never race the lazy-init.
    t0 = time.time()
    first = questions[0]
    results: dict = {first["id"]: pipeline.process_question(first["id"], first["question"])}
    log.info("Warmup question done in %.1fs (clients initialised)", time.time() - t0)

    rest = questions[1:]
    done = 1
    lock = threading.Lock()

    def _work(q: dict) -> dict:
        return pipeline.process_question(q["id"], q["question"])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_work, q): q["id"] for q in rest}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results[r["id"]] = r
                done += 1
                if done % 50 == 0 or done == total:
                    rate = done / (time.time() - t0)
                    eta = (total - done) / rate if rate else 0
                    log.info("  %d/%d done  (%.2f q/s, ETA %.0f min)",
                             done, total, rate, eta / 60)

    # Retry stragglers. A fallback entry (empty relevant_articles) usually means
    # a transient Qdrant/Gemini timeout under concurrency. Retry in rounds at a
    # gentler concurrency so a large backlog (from high --workers) clears fast
    # without re-thrashing the free-tier Qdrant.
    retry_workers = max(1, min(4, args.workers))
    for rnd in range(1, 4):
        failed = [q for q in questions if not results[q["id"]].get("relevant_articles")]
        if not failed:
            break
        log.info("Retry round %d: %d straggler(s) at %d workers…",
                 rnd, len(failed), retry_workers)
        with ThreadPoolExecutor(max_workers=retry_workers) as ex:
            for r in ex.map(_work, failed):
                results[r["id"]] = r
    still = sum(1 for q in questions if not results[q["id"]].get("relevant_articles"))
    if still:
        log.info("After retries: %d still empty (kept as fallback entries)", still)

    # Preserve input order; guarantee every id is present.
    ordered = [results[q["id"]] for q in questions]
    out = Path(args.output)
    out.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.time() - t0
    log.info("Wrote %d results → %s in %.1f min (%.2f q/s)",
             len(ordered), out, elapsed / 60, total / elapsed)


if __name__ == "__main__":
    main()
