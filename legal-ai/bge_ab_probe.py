"""
bge_ab_probe.py — BGE vs RRF ranking A/B probe (leaderboard-based validation).

WHY THIS SHAPE: there are no gold labels offline (R2AIStage1DATA.json is
id+question only); recall is measured only by submitting to the contest system.
So we cannot compute recall@K offline. Instead we isolate BGE's *ranking* value
with a same-pool A/B:

  1. For every question: rewrite → retrieve the top-`pool_size` RRF pool.
  2. Score that SAME pool with BGE-reranker-v2-m3 (bge_scorer.BgeScorer).
  3. Emit the scored pool (RRF order + bge score per candidate) to one JSON.
  4. From that pool, package TWO retrieve-only submissions at the same K:
        • RRF-top-K   (pool ordered by RRF)
        • BGE-top-K   (pool re-ordered by BGE)
     You submit both; the leaderboard gives recall@K for each. Same pool, same K,
     only the ranker differs → recall@K(BGE) − recall@K(RRF) is exactly BGE's
     ranking contribution at the head.

Modes:
  run      — full GPU run over all questions → pool JSON (resumable).
  dryrun   — N questions, print BGE score spread + timing (Part C gate). No write.
  package  — pool JSON → RRF-top-K and BGE-top-K submissions (cheap, re-sliceable
             at any K ≤ pool_size without re-running).

This file does NOT touch pipeline.py; it reuses LegalAIPipeline's public steps
plus the standalone BgeScorer. See TIP-020.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from bge_scorer import BgeScorer              # noqa: E402
from pipeline import LegalAIPipeline, PipelineState  # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("bge_ab_probe")
log.setLevel(logging.INFO)

_DEFAULT_POOL_SIZE = 200


# ---------------------------------------------------------------------------
# Per-question work: rewrite → retrieve pool → BGE score → compact pool entry
# ---------------------------------------------------------------------------

def _pool_entry(pipeline: LegalAIPipeline, bge: BgeScorer,
                qid, question: str) -> dict:
    """Return {id, question, rewritten_query, pool:[{article,doc,rrf,bge}...]}.

    `pool` stays in RRF order; each candidate carries its bge score so packaging
    can re-sort by either signal. Fail-soft: any error → empty pool entry.
    """
    try:
        state = PipelineState(question_id=qid, question=question)
        state = pipeline.step_rewrite(state)
        state = pipeline.step_decompose(state)
        state = pipeline.step_retrieve(state)       # RRF-ordered pool
        pool = state.fused_results
        # BGE judges the ORIGINAL question (mirrors the LLM reranker, which scores
        # against state.question). Adds 'bge_score' in place; RRF order preserved.
        bge.score_candidates(question, pool)
        return {
            "id": qid,
            "question": question,
            "rewritten_query": state.rewritten_query,
            "topic_description": state.topic_description,
            "intents": state.intent_queries,
            "num_intents": len(state.intent_queries),
            "retrieval_metrics": state.retrieval_metrics,
            "intent_hits": [
                {
                    "article": c.get("relevant_article_str", ""),
                    "doc": c.get("relevant_doc_str", ""),
                    "intent_ids": c.get("intent_ids", []),
                    "intent_queries": c.get("intent_queries", []),
                }
                for c in state.intent_hits
            ],
            "pool": [
                {
                    "article": c.get("relevant_article_str", ""),
                    "doc": c.get("relevant_doc_str", ""),
                    "rrf": round(float(c.get("rrf_score", 0.0)), 8),
                    "bge": round(float(c.get("bge_score", 0.0)), 6),
                }
                for c in pool
            ],
        }
    except Exception:
        log.exception("Q%s failed (bge pool) — emitting empty entry", qid)
        return {"id": qid, "question": question, "rewritten_query": "", "pool": []}


def _build(config_path: str, pool_size: int, dry_bge: bool):
    """Construct the pipeline (pinned to a pool_size pool) + a BgeScorer."""
    pipeline = LegalAIPipeline(config_path=config_path, mock=False)
    r = pipeline.config.retrieval
    r.top_k_fusion = pool_size
    r.top_k_dense = max(r.top_k_dense, pool_size + 40)
    r.top_k_bm25 = max(r.top_k_bm25, pool_size + 40)
    bge = BgeScorer(pipeline.config, dry_run=dry_bge)
    return pipeline, bge


def _preflight_or_exit(bge: BgeScorer) -> None:
    """Fail fast (<~12s) if the BGE endpoint is unreachable, instead of letting a
    wrong DNS hang for minutes via per-batch retries across every question."""
    ok, detail = bge.ping()
    if not ok:
        print(f"✗ BGE preflight FAILED: {detail}")
        print(f"  endpoint_id : {bge._b.endpoint_id or '(empty)'}")
        print(f"  dns         : {bge._b.dns or '(shared regional domain)'}")
        print("  → fix GPU_BGE_DNS / GPU_BGE_ENDPOINT_ID in .env and retry. "
              "Get the dedicated DNS via:\n"
              "    gcloud ai endpoints describe <ENDPOINT_ID> --region=asia-northeast1 "
              "--format='value(dedicatedEndpointDns)'")
        sys.exit(2)
    print(f"✓ BGE preflight: {detail}")


# ---------------------------------------------------------------------------
# run mode (resumable parallel run, mirrors run_parallel.py)
# ---------------------------------------------------------------------------

def mode_run(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    total = len(questions)
    out = Path(args.output)
    lock = threading.Lock()
    results: dict = {}

    def _persist() -> None:
        ordered = [results[q["id"]] for q in questions if q["id"] in results]
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(ordered, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)

    if args.resume and out.exists():
        try:
            for r in json.loads(out.read_text(encoding="utf-8")):
                if r.get("id") is not None and r.get("pool"):
                    results[r["id"]] = r
            log.info("Resume: %d/%d already complete in %s", len(results), total, out)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Resume failed to read %s (%s) — starting fresh", out, e)

    todo = [q for q in questions if q["id"] not in results]
    log.info("Loaded %d questions; %d to process; workers=%d", total, len(todo), args.workers)

    pipeline, bge = _build(args.config, args.pool_size, args.dry_bge)
    _preflight_or_exit(bge)

    def _work(q: dict) -> dict:
        return _pool_entry(pipeline, bge, q["id"], q["question"])

    t0 = time.time()
    if not todo:
        log.info("Nothing to process — all complete.")
        _persist()
        return

    first = todo[0]
    results[first["id"]] = _work(first)
    log.info("Warmup question done in %.1fs (pool=%d)",
             time.time() - t0, len(results[first["id"]].get("pool", [])))
    _persist()

    processed = 1
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_work, q): q["id"] for q in todo[1:]}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results[r["id"]] = r
                processed += 1
                if processed % 50 == 0 or processed == len(todo):
                    rate = processed / (time.time() - t0)
                    eta = (len(todo) - processed) / rate if rate else 0
                    log.info("  %d/%d done  (%.2f q/s, ETA %.0f min)",
                             len(results), total, rate, eta / 60)
                    _persist()

    retry_workers = max(1, min(4, args.workers))
    for rnd in range(1, 4):
        failed = [q for q in questions if not results.get(q["id"], {}).get("pool")]
        if not failed:
            break
        log.info("Retry round %d: %d straggler(s) at %d workers…",
                 rnd, len(failed), retry_workers)
        with ThreadPoolExecutor(max_workers=retry_workers) as ex:
            for r in ex.map(_work, failed):
                results[r["id"]] = r
        _persist()

    for q in questions:
        results.setdefault(q["id"], {
            "id": q["id"], "question": q["question"], "rewritten_query": "", "pool": [],
        })
    _persist()
    still = sum(1 for q in questions if not results.get(q["id"], {}).get("pool"))
    log.info("Wrote %d pool entries → %s in %.1f min (%d still empty)",
             len(results), out, (time.time() - t0) / 60, still)


# ---------------------------------------------------------------------------
# dryrun mode (Part C gate — verify endpoint health + score spread)
# ---------------------------------------------------------------------------

def mode_dryrun(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))[: args.limit or 5]
    pipeline, bge = _build(args.config, args.pool_size, args.dry_bge)
    _preflight_or_exit(bge)

    print(f"=== BGE DRY-RUN ({len(questions)} questions, pool={args.pool_size}) ===")
    flat_warn = 0
    for q in questions:
        t0 = time.time()
        e = _pool_entry(pipeline, bge, q["id"], q["question"])
        dt = time.time() - t0
        pool = e["pool"]
        if not pool:
            print(f"Q{q['id']}: EMPTY POOL (retrieval failed) — {dt:.1f}s")
            continue
        bge_scores = [c["bge"] for c in pool]
        lo, hi = min(bge_scores), max(bge_scores)
        mean = statistics.mean(bge_scores)
        spread = hi - lo
        if spread < 1e-6:
            flat_warn += 1
        top = sorted(pool, key=lambda c: c["bge"], reverse=True)[:5]
        print(f"\nQ{q['id']} ({dt:.1f}s, {len(pool)} cands) "
              f"bge[min={lo:.3f} mean={mean:.3f} max={hi:.3f} spread={spread:.3f}]")
        print(f"  Q: {q['question'][:90]}")
        for c in top:
            print(f"   bge={c['bge']:.3f} rrf={c['rrf']:.4f}  {c['article'][:80]}")

    print("\n=== GATE ===")
    if flat_warn:
        print(f"⚠ {flat_warn} question(s) had FLAT bge scores (spread≈0). "
              "Endpoint may be mis-serving — DO NOT run full. Investigate.")
    else:
        print("✓ BGE scores show spread; endpoint healthy. OK to run full.")


# ---------------------------------------------------------------------------
# package mode (pool JSON → RRF-top-K + BGE-top-K submissions)
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return " ".join(s.split())


def _submission_entry(e: dict, items: list[dict]) -> dict:
    """Build one retrieve-only submission entry (empty answer) with deduped refs."""
    seen_a, arts = set(), []
    seen_d, docs = set(), []
    for c in items:
        a = _norm(c.get("article", ""))
        if a and a not in seen_a:
            seen_a.add(a); arts.append(a)
        d = _norm(c.get("doc", ""))
        if d and d not in seen_d:
            seen_d.add(d); docs.append(d)
    return {"id": e["id"], "question": e["question"], "answer": "",
            "relevant_docs": docs, "relevant_articles": arts}


def mode_package(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from submit import save_submission

    pool_path = Path(args.pool_json or args.output)
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    k = args.top_k

    rrf_sub, bge_sub, empty = [], [], 0
    for e in data:
        pool = e.get("pool", [])
        if not pool:
            empty += 1
        rrf_order = pool[:k]                                   # stored in RRF order
        bge_order = sorted(pool, key=lambda c: c["bge"], reverse=True)[:k]
        rrf_sub.append(_submission_entry(e, rrf_order))
        bge_sub.append(_submission_entry(e, bge_order))

    out_rrf = args.out_rrf or f"outputs/submission_rrf_top{k}"
    out_bge = args.out_bge or f"outputs/submission_bge_top{k}"
    save_submission(rrf_sub, out_rrf)
    save_submission(bge_sub, out_bge)

    def _stats(sub):
        arts = [len(s["relevant_articles"]) for s in sub]
        return min(arts), max(arts), round(statistics.mean(arts), 1)

    print(f"\nPackaged K={k} from {pool_path} ({len(data)} questions, {empty} empty pools)")
    print(f"  RRF-top{k} → {out_rrf}   articles/q min/max/mean = {_stats(rrf_sub)}")
    print(f"  BGE-top{k} → {out_bge}   articles/q min/max/mean = {_stats(bge_sub)}")
    print("Submit BOTH; compare leaderboard recall@K. recall(BGE) − recall(RRF) "
          "= BGE ranking value at the head.")


# ---------------------------------------------------------------------------
# analyze mode — overlap + union recall/size + unique contributions
# ---------------------------------------------------------------------------

def mode_analyze(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from submit import save_submission

    pool_path = Path(args.pool_json or args.output)
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    KB, KR1, KR2 = args.top_k, args.top_k, args.k_rrf2   # 80, 80, 60

    overlap, u8080_sz, u8060_sz, only_bge, only_rrf = [], [], [], [], []
    sub_u8080, sub_u8060, empty = [], [], 0
    for e in data:
        pool = e.get("pool", [])
        intent_order = e.get("intent_hits", [])
        if not pool:
            empty += 1
            sub_u8080.append(_submission_entry(e, intent_order))
            sub_u8060.append(_submission_entry(e, intent_order))
            continue
        rrf_order = pool                                  # stored in RRF order
        bge_order = sorted(pool, key=lambda c: c["bge"], reverse=True)
        # article-string sets (chunk == article 1:1); drop empties
        s_bge80 = {c["article"] for c in bge_order[:KB] if c["article"]}
        s_rrf80 = {c["article"] for c in rrf_order[:KR1] if c["article"]}
        s_rrf60 = {c["article"] for c in rrf_order[:KR2] if c["article"]}
        s_intent = {c["article"] for c in intent_order if c.get("article")}

        overlap.append(len(s_bge80 & s_rrf80))
        u8080_sz.append(len(s_bge80 | s_rrf80 | s_intent))
        u8060_sz.append(len(s_bge80 | s_rrf60 | s_intent))
        only_bge.append(len(s_bge80 - s_rrf80))
        only_rrf.append(len(s_rrf80 - s_bge80))

        # union submissions (empty answer) — dedup handled by _submission_entry
        sub_u8080.append(_submission_entry(e, bge_order[:KB] + rrf_order[:KR1] + intent_order))
        sub_u8060.append(_submission_entry(e, bge_order[:KB] + rrf_order[:KR2] + intent_order))

    out_u8080 = "outputs/submission_union_b80_r80"
    out_u8060 = "outputs/submission_union_b80_r60"
    save_submission(sub_u8080, out_u8080)
    save_submission(sub_u8060, out_u8060)

    n = len(data)
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    print("\n=== UNION & OVERLAP ANALYSIS ===")
    print(f"Questions: {n}  (empty pools: {empty})")
    print("\n--- Overlap ---")
    print(f"Avg |top{KB}_bge ∩ top{KR1}_rrf|: {avg(overlap):.1f} / {KB}")
    print(f"Avg overlap %: {100*avg(overlap)/KB:.1f}%")
    print("\n--- Recall (RRF/BGE from leaderboard; union = submit the zips below) ---")
    rr = args.recall_rrf; rb = args.recall_bge
    print(f"recall@{KR1} (RRF only):           {rr if rr is not None else 'n/a':}")
    print(f"recall@{KB} (BGE only):           {rb if rb is not None else 'n/a':}")
    print(f"recall (top{KB}_bge ∪ top{KR1}_rrf): SUBMIT {out_u8080}/  (avg union size: {avg(u8080_sz):.1f})")
    print(f"recall (top{KB}_bge ∪ top{KR2}_rrf): SUBMIT {out_u8060}/  (avg union size: {avg(u8060_sz):.1f})")
    print("\n--- Unique contributions (counts; 'of which gold' needs leaderboard) ---")
    print(f"Articles ONLY in BGE top{KB} (not in RRF top{KR1}): avg {avg(only_bge):.1f}/câu")
    print(f"  → of which gold: needs gold; INFER from recall(∪{KB}/{KR1}) − recall@{KR1}(RRF)")
    print(f"Articles ONLY in RRF top{KR1} (not in BGE top{KB}): avg {avg(only_rrf):.1f}/câu")
    print(f"  → of which gold: needs gold; INFER from recall(∪{KB}/{KR1}) − recall@{KB}(BGE)")
    if rr is not None and rb is not None:
        print("\n--- How to read once union recalls are in ---")
        print(f"  recall(∪{KB}/{KR1}) > {rr} (RRF)  → BGE's unique finds DO contain gold RRF missed → BGE complements.")
        print(f"  recall(∪{KB}/{KR1}) ≈ {rr}        → BGE adds no new gold → drop BGE.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="BGE vs RRF ranking A/B probe")
    ap.add_argument("--mode", choices=["run", "dryrun", "package", "analyze"], required=True)
    ap.add_argument("--config", default="config_gpu.yaml")
    ap.add_argument("--input", default="../R2AIStage1DATA.json",
                    help="JSON [{id, question}, ...] (run/dryrun)")
    ap.add_argument("--output", default="outputs/bge_ab_pool.json",
                    help="pool JSON path (written by run; read by package)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N questions (dryrun defaults to 5)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pool-size", type=int, default=_DEFAULT_POOL_SIZE,
                    help="RRF pool size to score with BGE (default 200)")
    ap.add_argument("--dry-bge", action="store_true",
                    help="use deterministic mock BGE scores (offline plumbing test)")
    # package-only:
    ap.add_argument("--top-k", type=int, default=80, help="K for the two submissions")
    ap.add_argument("--pool-json", default=None, help="pool JSON to package (default: --output)")
    ap.add_argument("--out-rrf", default=None)
    ap.add_argument("--out-bge", default=None)
    # analyze-only:
    ap.add_argument("--k-rrf2", type=int, default=60,
                    help="second union's RRF depth (top80_bge ∪ top60_rrf)")
    ap.add_argument("--recall-rrf", type=float, default=None,
                    help="known leaderboard recall@K for RRF (to print in report)")
    ap.add_argument("--recall-bge", type=float, default=None,
                    help="known leaderboard recall@K for BGE (to print in report)")
    args = ap.parse_args()

    {"run": mode_run, "dryrun": mode_dryrun,
     "package": mode_package, "analyze": mode_analyze}[args.mode](args)


if __name__ == "__main__":
    main()
