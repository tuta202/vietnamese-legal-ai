from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.backends.bge import BgeScorer  # noqa: E402
from legal_rag.retrieval.config import load_config, validate_config  # noqa: E402
from legal_rag.output.submission import save_submission  # noqa: E402


log = logging.getLogger("legal_rag.ranking.intent_bge")

DEFAULT_CACHE = "cache/bge_intent_compression_clean.jsonl"
DEFAULT_CORPUS = "corpus/data/corpus_clean_asof_20260301.json"
DEFAULT_KEEP_SUBMISSION = "outputs/submission_rrf_top60_union_intent_hits_clean/results.json"
DEFAULT_INTENT_RESULTS = "outputs/intent_hits_clean_results.json"
DEFAULT_RRF60_SUBMISSION = "outputs/submission_rrf_top60_clean/results.json"

DOC_MAXCHARS = 1800
CACHE_TOP_N = 50
DEFAULT_TOP_EACH_INTENT = (3, 5)
DEFAULT_CAPS = (10, 12)
ADAPTIVE_CAP_MAX = 20
ADAPTIVE_CAP_MIN = 10
ADAPTIVE_CAP_PER_INTENT = 5

# Normalized aggregation weights. BGE rank remains the primary signal; the other
# terms add bounded evidence instead of letting intent count dominate the score.
W_BGE_RANK = 1.0
W_RRF_RANK_BUCKET = 0.25
W_INTENT_MATCH = 0.25
W_IN_INTENT_HITS = 0.15


def norm_text(s: str) -> str:
    return " ".join((s or "").split())


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def article_to_doc(article: str) -> str:
    parts = article.split("|")
    return "|".join(parts[:2]) if len(parts) >= 2 else article


def load_questions(path: Path) -> list[dict]:
    return list(read_json(path))


def load_corpus(path: Path) -> dict[str, dict]:
    raw = read_json(path)
    articles = raw.get("articles", raw) if isinstance(raw, dict) else raw
    by_article: dict[str, dict] = {}
    for item in articles:
        key = norm_text(item.get("relevant_article_str", ""))
        if key:
            by_article[key] = item
    return by_article


def load_submission_articles(path: Path) -> dict[object, list[str]]:
    rows = read_json(path)
    out: dict[object, list[str]] = {}
    for row in rows:
        out[row["id"]] = [norm_text(a) for a in row.get("relevant_articles", []) if a]
    return out


def load_intent_results(path: Path) -> tuple[dict[object, list[str]], dict[object, list[str]]]:
    rows = read_json(path)
    intents_by_qid: dict[object, list[str]] = {}
    intent_hits_by_qid: dict[object, list[str]] = {}
    for row in rows:
        qid = row["id"]
        intents_by_qid[qid] = [str(x).strip() for x in row.get("legal_intents", []) if str(x).strip()]
        hits: list[str] = []
        source_hits = row.get("intent_hits") or row.get("intent_hits_union") or []
        for item in source_hits:
            article = norm_text(item.get("article", ""))
            if article:
                hits.append(article)
        if not hits:
            hits = [norm_text(a) for a in row.get("relevant_articles", []) if a]
        intent_hits_by_qid[qid] = dedupe_keep_order(hits)
    return intents_by_qid, intent_hits_by_qid


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_keep(
    qid: object,
    rrf60_by_qid: dict[object, list[str]],
    intent_hits_by_qid: dict[object, list[str]],
) -> list[str]:
    return dedupe_keep_order(rrf60_by_qid.get(qid, []) + intent_hits_by_qid.get(qid, []))


def doc_text(article: dict) -> str:
    law_type = norm_text(article.get("law_type", ""))
    law_name = norm_text(article.get("law_name", ""))
    number = norm_text(article.get("dieu_number", ""))
    title = norm_text(article.get("dieu_title", ""))
    content = norm_text(article.get("content", ""))[:DOC_MAXCHARS]
    return f"{law_type} {law_name}\n{number}. {title}\n{content}".strip()


def load_cache(path: Path) -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id")
            idx = row.get("intent_index")
            if qid is None or idx is None:
                continue
            if row.get("ranked_articles"):
                records[(str(qid), int(idx))] = row
    return records


def append_cache(path: Path, record: dict, lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def rank_one_intent(
    bge: BgeScorer,
    qid: object,
    intent_index: int,
    intent: str,
    keep_articles: list[str],
    corpus_by_article: dict[str, dict],
) -> dict:
    docs: list[str] = []
    valid_articles: list[str] = []
    missing: list[str] = []
    for article in keep_articles:
        meta = corpus_by_article.get(article)
        if meta is None:
            missing.append(article)
            continue
        valid_articles.append(article)
        docs.append(doc_text(meta))

    if missing:
        raise ValueError(
            f"Q{qid} intent {intent_index}: {len(missing)} keep articles missing from corpus"
        )

    scores = bge.score_pairs(intent, docs)
    if len(scores) != len(valid_articles):
        raise ValueError(
            f"Q{qid} intent {intent_index}: score count {len(scores)} != documents {len(valid_articles)}"
        )
    ranked = sorted(
        [
            {
                "chunk_id": corpus_by_article[article].get("chunk_id", ""),
                "article": article,
                "doc": norm_text(corpus_by_article[article].get("relevant_doc_str", "")) or article_to_doc(article),
                "score": float(score),
            }
            for article, score in zip(valid_articles, scores)
        ],
        key=lambda x: (-x["score"], x["article"]),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return {
        "question_id": str(qid),
        "intent_index": intent_index,
        "intent": intent,
        "keep_size": len(keep_articles),
        "scored_size": len(valid_articles),
        "missing_articles": missing[:20],
        "ranked_articles": ranked[:CACHE_TOP_N],
    }


def run_mode(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    problems = validate_config(config)
    if problems:
        raise ValueError(f"Invalid config: {'; '.join(problems)}")
    bge = BgeScorer(config, dry_run=args.dry_bge)
    ok, detail = bge.ping()
    if not ok:
        raise RuntimeError(f"BGE preflight failed: {detail}")
    log.info("BGE preflight: %s", detail)

    questions = load_questions(Path(args.input))
    if args.limit:
        questions = questions[: args.limit]
    corpus_by_article = load_corpus(Path(args.corpus))
    rrf60_by_qid = load_submission_articles(Path(args.rrf60_submission))
    intents_by_qid, intent_hits_by_qid = load_intent_results(Path(args.intent_results))
    cache_path = Path(args.cache)
    cache = load_cache(cache_path) if args.resume else {}

    jobs: list[tuple[object, int, str, list[str]]] = []
    skipped_no_intent = 0
    skipped_no_keep = 0
    for q in questions:
        qid = q["id"]
        intents = intents_by_qid.get(qid, [])
        keep = build_keep(qid, rrf60_by_qid, intent_hits_by_qid)
        if not intents:
            skipped_no_intent += 1
            continue
        if not keep:
            skipped_no_keep += 1
            continue
        for idx, intent in enumerate(intents):
            if (str(qid), idx) not in cache:
                jobs.append((qid, idx, intent, keep))

    log.info(
        "Loaded questions=%d corpus=%d cache=%d jobs=%d skipped_no_intent=%d skipped_no_keep=%d",
        len(questions),
        len(corpus_by_article),
        len(cache),
        len(jobs),
        skipped_no_intent,
        skipped_no_keep,
    )

    lock = threading.Lock()
    started = time.time()
    done = 0

    def work(job: tuple[object, int, str, list[str]]) -> dict:
        qid, idx, intent, keep = job
        return rank_one_intent(bge, qid, idx, intent, keep, corpus_by_article)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                qid, idx, intent, keep = job
                log.warning(
                    "Intent job failed qid=%s intent_index=%s keep=%d: %s",
                    qid,
                    idx,
                    len(keep),
                    exc,
                )
                done += 1
                continue
            append_cache(cache_path, row, lock)
            done += 1
            if done % args.checkpoint_every == 0 or done == len(jobs):
                rate = done / max(time.time() - started, 1e-6)
                eta = (len(jobs) - done) / rate if rate else 0
                log.info("%d/%d intent jobs done (%.2f jobs/s, ETA %.1f min)", done, len(jobs), rate, eta / 60)

    completed_cache = load_cache(cache_path)
    expected_keys = {
        (str(q["id"]), idx)
        for q in questions
        for idx, _ in enumerate(intents_by_qid.get(q["id"], []))
        if build_keep(q["id"], rrf60_by_qid, intent_hits_by_qid)
    }
    missing_keys = sorted(expected_keys - set(completed_cache))
    if missing_keys and args.strict_errors:
        raise RuntimeError(
            f"{len(missing_keys)} BGE intent jobs missing; resume required: {missing_keys[:20]}"
        )
    log.info("Done. Cache written to %s", cache_path)


def rank_component(rank: int | None) -> float:
    if not rank:
        return 0.0
    return 1.0 / float(rank)


def rrf_rank_bucket_score(rank: int | None) -> float:
    if not rank:
        return 0.0
    if rank <= 10:
        return 1.0
    if rank <= 30:
        return 0.7
    if rank <= 60:
        return 0.4
    return 0.0


def aggregate_question(
    qid: object,
    question: str,
    cache_rows: list[dict],
    rrf60: list[str],
    intent_hits: list[str],
    top_each_intent: int,
    cap: int | None,
    pin_rrf_top: int = 0,
    pin_intent_top: int = 0,
    adaptive_cap_min: int = ADAPTIVE_CAP_MIN,
    adaptive_cap_max: int = ADAPTIVE_CAP_MAX,
    adaptive_cap_per_intent: int = ADAPTIVE_CAP_PER_INTENT,
) -> tuple[dict, list[dict]]:
    rrf_set = set(rrf60)
    rrf_rank_by_article = {article: rank for rank, article in enumerate(rrf60, start=1)}
    intent_set = set(intent_hits)
    num_intents = max(len(cache_rows), 1)
    effective_cap = cap if cap is not None else min(
        adaptive_cap_max,
        max(adaptive_cap_min, num_intents * adaptive_cap_per_intent),
    )
    features: dict[str, dict] = {}
    must_keep_articles = dedupe_keep_order(rrf60[:pin_rrf_top] + intent_hits[:pin_intent_top])
    pin_order = {article: idx for idx, article in enumerate(must_keep_articles, start=1)}

    def ensure_feature(article: str, item: dict | None = None) -> dict:
        item = item or {}
        return features.setdefault(
            article,
            {
                "question_id": qid,
                "article": article,
                "doc": norm_text(item.get("doc", "")) or article_to_doc(article),
                "best_bge_rank": None,
                "best_bge_score": 0.0,
                "num_intents_matched": 0,
                "matched_intent_indexes": [],
                "rrf_rank": rrf_rank_by_article.get(article),
                "rrf_rank_bucket_score": rrf_rank_bucket_score(rrf_rank_by_article.get(article)),
                "is_in_top60_rrf": article in rrf_set,
                "is_in_intent_hits": article in intent_set,
                "is_must_keep": article in pin_order,
                "pin_order": pin_order.get(article),
            },
        )

    for row in cache_rows:
        intent_index = int(row.get("intent_index", 0))
        for item in sorted(row.get("ranked_articles", []), key=lambda x: int(x.get("rank", 10**9)))[:top_each_intent]:
            article = norm_text(item.get("article", ""))
            if not article:
                continue
            rank = int(item.get("rank", 10**9))
            score = float(item.get("score", 0.0))
            f = ensure_feature(article, item)
            current_rank = f["best_bge_rank"]
            f["best_bge_rank"] = rank if current_rank is None else min(current_rank, rank)
            f["best_bge_score"] = max(f["best_bge_score"], score)
            if intent_index not in f["matched_intent_indexes"]:
                f["matched_intent_indexes"].append(intent_index)
                f["num_intents_matched"] += 1

    for article in must_keep_articles:
        ensure_feature(article)

    for article, f in features.items():
        intent_match_ratio = min(1.0, f["num_intents_matched"] / num_intents)
        f["bge_rank_score"] = rank_component(f["best_bge_rank"])
        f["intent_match_ratio"] = intent_match_ratio
        f["aggregate_score"] = (
            W_BGE_RANK * f["bge_rank_score"]
            + W_RRF_RANK_BUCKET * f["rrf_rank_bucket_score"]
            + W_INTENT_MATCH * f["intent_match_ratio"]
            + W_IN_INTENT_HITS * (1.0 if f["is_in_intent_hits"] else 0.0)
        )

    ranked = sorted(
        features.values(),
        key=lambda f: (
            -float(f["aggregate_score"]),
            int(f["best_bge_rank"] or 10**9),
            -float(f["best_bge_score"]),
            f["article"],
        ),
    )
    pinned = sorted(
        [f for f in ranked if f.get("is_must_keep")],
        key=lambda f: (int(f.get("pin_order") or 10**9), -float(f["aggregate_score"]), f["article"]),
    )
    selected: list[dict] = pinned[:effective_cap]
    selected_articles = {f["article"] for f in selected}
    for item in ranked:
        if len(selected) >= effective_cap:
            break
        if item["article"] in selected_articles:
            continue
        selected.append(item)
        selected_articles.add(item["article"])
    for item in ranked:
        item["selected"] = item["article"] in selected_articles

    docs: list[str] = []
    seen_docs: set[str] = set()
    articles: list[str] = []
    seen_articles: set[str] = set()
    for item in selected:
        article = item["article"]
        if article not in seen_articles:
            seen_articles.add(article)
            articles.append(article)
        doc = item.get("doc") or article_to_doc(article)
        if doc and doc not in seen_docs:
            seen_docs.add(doc)
            docs.append(doc)
    for rank, item in enumerate(selected, start=1):
        item["output_rank"] = rank

    return (
        {
            "id": qid,
            "question": question,
            "answer": "",
            "relevant_docs": docs,
            "relevant_articles": articles,
        },
        ranked,
    )


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_caps(s: str) -> list[int | None]:
    caps: list[int | None] = []
    for raw in s.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"adaptive", "auto"}:
            caps.append(None)
        else:
            caps.append(int(item))
    return caps


def cap_label(
    cap: int | None,
    adaptive_cap_min: int = ADAPTIVE_CAP_MIN,
    adaptive_cap_max: int = ADAPTIVE_CAP_MAX,
    adaptive_cap_per_intent: int = ADAPTIVE_CAP_PER_INTENT,
) -> str:
    if cap is None:
        return f"adaptive_cap{adaptive_cap_max}_min{adaptive_cap_min}_x{adaptive_cap_per_intent}"
    return f"cap{cap}"


def pin_label(pin_rrf_top: int, pin_intent_top: int) -> str:
    if pin_rrf_top <= 0 and pin_intent_top <= 0:
        return ""
    return f"_pinrrf{pin_rrf_top}_pinintent{pin_intent_top}"


def package_mode(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.input))
    if args.limit:
        questions = questions[: args.limit]
    rrf60_by_qid = load_submission_articles(Path(args.rrf60_submission))
    intents_by_qid, intent_hits_by_qid = load_intent_results(Path(args.intent_results))
    cache = load_cache(Path(args.cache))

    cache_by_qid: dict[str, list[dict]] = {}
    for (qid, _idx), row in cache.items():
        cache_by_qid.setdefault(qid, []).append(row)
    for rows in cache_by_qid.values():
        rows.sort(key=lambda r: int(r.get("intent_index", 0)))

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    diagnostics_rows: list[dict] = []
    summary_rows: list[dict] = []

    for top_each in parse_int_list(args.top_each_intent):
        for cap in parse_caps(args.caps):
            label = cap_label(cap, args.adaptive_cap_min, args.adaptive_cap_max, args.adaptive_cap_per_intent)
            label = f"{label}{pin_label(args.pin_rrf_top, args.pin_intent_top)}"
            sub: list[dict] = []
            feature_rows: list[dict] = []
            empty = 0
            for q in questions:
                qid = q["id"]
                qid_key = str(qid)
                rows = cache_by_qid.get(qid_key, [])
                entry, ranked_features = aggregate_question(
                    qid=qid,
                    question=q["question"],
                    cache_rows=rows,
                    rrf60=rrf60_by_qid.get(qid, []),
                    intent_hits=intent_hits_by_qid.get(qid, []),
                    top_each_intent=top_each,
                    cap=cap,
                    pin_rrf_top=args.pin_rrf_top,
                    pin_intent_top=args.pin_intent_top,
                    adaptive_cap_min=args.adaptive_cap_min,
                    adaptive_cap_max=args.adaptive_cap_max,
                    adaptive_cap_per_intent=args.adaptive_cap_per_intent,
                )
                if not entry["relevant_articles"]:
                    empty += 1
                sub.append(entry)
                diagnostics_rows.append(
                    {
                        "question_id": qid,
                        "variant": f"top{top_each}_{label}",
                        "num_intents": len(intents_by_qid.get(qid, [])),
                        "rrf60_size": len(rrf60_by_qid.get(qid, [])),
                        "intent_hits_size": len(intent_hits_by_qid.get(qid, [])),
                        "candidate_feature_count": len(ranked_features),
                        "output_size": len(entry["relevant_articles"]),
                    }
                )
                for f in ranked_features:
                    if f.get("selected"):
                        feature_rows.append(dict(f))

            out_dir = out_root / f"submission_bge_intent_compress_top{top_each}_{label}_clean"
            save_submission(sub, out_dir)
            feature_path = out_root / f"features_bge_intent_compress_top{top_each}_{label}_clean.jsonl"
            with feature_path.open("w", encoding="utf-8") as f:
                for row in feature_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            sizes = [len(x["relevant_articles"]) for x in sub]
            summary_rows.append(
                {
                    "variant": f"top{top_each}_{label}",
                    "questions": len(sub),
                    "empty": empty,
                    "min": min(sizes) if sizes else 0,
                    "max": max(sizes) if sizes else 0,
                    "mean": statistics.mean(sizes) if sizes else 0.0,
                    "median": statistics.median(sizes) if sizes else 0.0,
                    "submission_dir": str(out_dir),
                    "feature_path": str(feature_path),
                }
            )

    diag_path = out_root / "diagnostics_bge_intent_compression_clean.csv"
    with diag_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "variant",
                "num_intents",
                "rrf60_size",
                "intent_hits_size",
                "candidate_feature_count",
                "output_size",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostics_rows)

    summary_path = out_root / "summary_bge_intent_compression_clean.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Diagnostics: {diag_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="BGE intent-wise compression from clean keep = top60_rrf | intent_hits."
    )
    parser.add_argument("--mode", choices=["run", "package"], required=True)
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input", default="../R2AIStage1DATA.json")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--keep-submission", default=DEFAULT_KEEP_SUBMISSION)
    parser.add_argument("--rrf60-submission", default=DEFAULT_RRF60_SUBMISSION)
    parser.add_argument("--intent-results", default=DEFAULT_INTENT_RESULTS)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-errors", action="store_true")
    parser.add_argument("--dry-bge", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--top-each-intent", default="3,5")
    parser.add_argument("--caps", default="10,12")
    parser.add_argument("--pin-rrf-top", type=int, default=0)
    parser.add_argument("--pin-intent-top", type=int, default=0)
    parser.add_argument("--adaptive-cap-min", type=int, default=ADAPTIVE_CAP_MIN)
    parser.add_argument("--adaptive-cap-max", type=int, default=ADAPTIVE_CAP_MAX)
    parser.add_argument("--adaptive-cap-per-intent", type=int, default=ADAPTIVE_CAP_PER_INTENT)
    args = parser.parse_args()

    if args.mode == "run":
        run_mode(args)
    else:
        package_mode(args)


if __name__ == "__main__":
    main()
