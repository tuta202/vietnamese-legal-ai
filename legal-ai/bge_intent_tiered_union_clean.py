from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submit import save_submission  # noqa: E402


DEFAULT_BGE_CACHE = "cache/bge_intent_compression_clean.jsonl"
DEFAULT_RRF60_SUBMISSION = "outputs/submission_rrf_top60_clean/results.json"
DEFAULT_INTENT_RESULTS = "outputs/intent_hits_clean_results.json"
DEFAULT_INTENT_RANKED_RESULTS = "outputs/intent_ranked_hits_clean_results.json"


def norm_text(s: str) -> str:
    return " ".join((s or "").split())


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = norm_text(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def article_to_doc(article: str) -> str:
    parts = article.split("|")
    return "|".join(parts[:2]) if len(parts) >= 2 else article


def load_submission_articles(path: Path) -> dict[object, list[str]]:
    rows = read_json(path)
    return {
        row["id"]: [norm_text(a) for a in row.get("relevant_articles", []) if a]
        for row in rows
    }


def load_intent_results(path: Path) -> tuple[dict[object, list[str]], dict[object, list[dict]]]:
    rows = read_json(path)
    intents_by_qid: dict[object, list[str]] = {}
    hits_by_qid: dict[object, list[dict]] = {}
    for row in rows:
        qid = row["id"]
        intents_by_qid[qid] = [str(x).strip() for x in row.get("legal_intents", []) if str(x).strip()]
        hits_by_qid[qid] = [
            {
                "article": norm_text(item.get("article", "")),
                "doc": norm_text(item.get("doc", "")) or article_to_doc(norm_text(item.get("article", ""))),
                "intent_ids": item.get("intent_ids", []),
                "intent_rank": item.get("intent_rank") or 10**9,
                "intent_rrf_score": float(item.get("intent_rrf_score") or item.get("rrf_score") or 0.0),
            }
            for item in row.get("intent_hits", [])
            if item.get("article")
        ]
    return intents_by_qid, hits_by_qid


def load_intent_ranked_results(path: Path) -> tuple[dict[object, list[str]], dict[object, dict[int, list[dict]]]]:
    if not path.exists():
        return {}, {}
    rows = read_json(path)
    intents_by_qid: dict[object, list[str]] = {}
    ranked_by_qid: dict[object, dict[int, list[dict]]] = {}
    for row in rows:
        qid = row["id"]
        intents_by_qid[qid] = [str(x).strip() for x in row.get("legal_intents", []) if str(x).strip()]
        per_intent: dict[int, list[dict]] = {}
        for intent_row in row.get("intent_ranked_hits", []):
            idx = int(intent_row.get("intent_index", int(intent_row.get("intent_id", 1)) - 1))
            ranked = [
                {
                    "article": norm_text(item.get("article", "")),
                    "doc": norm_text(item.get("doc", "")) or article_to_doc(norm_text(item.get("article", ""))),
                    "rank": int(item.get("rank", 10**9)),
                    "score": float(item.get("rrf_score", 0.0)),
                }
                for item in intent_row.get("ranked_articles", [])
                if item.get("article")
            ]
            per_intent[idx] = sorted(ranked, key=lambda x: (x["rank"], -x["score"], x["article"]))
        ranked_by_qid[qid] = per_intent
    return intents_by_qid, ranked_by_qid


def load_bge_cache(path: Path) -> dict[str, dict[int, list[dict]]]:
    by_qid: dict[str, dict[int, list[dict]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("question_id"))
            idx = int(row.get("intent_index", 0))
            ranked = [
                {
                    "article": norm_text(item.get("article", "")),
                    "doc": norm_text(item.get("doc", "")) or article_to_doc(norm_text(item.get("article", ""))),
                    "rank": int(item.get("rank", 10**9)),
                    "score": float(item.get("score", 0.0)),
                }
                for item in row.get("ranked_articles", [])
                if item.get("article")
            ]
            if ranked:
                # Keep the latest completed row for duplicate resume lines.
                by_qid.setdefault(qid, {})[idx] = sorted(ranked, key=lambda x: (x["rank"], -x["score"], x["article"]))
    return by_qid


def top_m_intent_hits_each_intent(
    hits: list[dict],
    num_intents: int,
    top_m: int,
) -> list[dict]:
    selected: list[dict] = []
    for intent_id in range(1, num_intents + 1):
        candidates = [
            item
            for item in hits
            if intent_id in {int(x) for x in item.get("intent_ids", []) if str(x).isdigit()}
        ]
        candidates.sort(
            key=lambda x: (
                int(x.get("intent_rank") or 10**9),
                -float(x.get("intent_rrf_score") or 0.0),
                x["article"],
            )
        )
        selected.extend(candidates[:top_m])
    return selected


def build_entry(
    qid: object,
    question: str,
    rrf_articles: list[str],
    bge_by_intent: dict[int, list[dict]],
    intent_hits: list[dict],
    intent_ranked_by_intent: dict[int, list[dict]],
    num_intents: int,
    top_b_rrf: int,
    top_n_bge: int,
    top_m_intent: int,
) -> tuple[dict, dict]:
    final_items: list[dict] = []

    for article in rrf_articles[:top_b_rrf]:
        final_items.append({"article": article, "doc": article_to_doc(article), "source": "rrf"})

    for intent_index in range(num_intents):
        for item in bge_by_intent.get(intent_index, [])[:top_n_bge]:
            final_items.append({**item, "source": f"bge_intent_{intent_index}"})

    if intent_ranked_by_intent:
        for intent_index in range(num_intents):
            for item in intent_ranked_by_intent.get(intent_index, [])[:top_m_intent]:
                final_items.append({**item, "source": f"intent_raw_{intent_index}"})
        intent_raw_count = sum(min(top_m_intent, len(intent_ranked_by_intent.get(i, []))) for i in range(num_intents))
    else:
        inferred = top_m_intent_hits_each_intent(intent_hits, num_intents, top_m_intent)
        for item in inferred:
            final_items.append({**item, "source": "intent_hits_inferred"})
        intent_raw_count = len(inferred)

    seen_articles: set[str] = set()
    articles: list[str] = []
    seen_docs: set[str] = set()
    docs: list[str] = []
    for item in final_items:
        article = norm_text(item.get("article", ""))
        if not article or article in seen_articles:
            continue
        seen_articles.add(article)
        articles.append(article)
        doc = norm_text(item.get("doc", "")) or article_to_doc(article)
        if doc and doc not in seen_docs:
            seen_docs.add(doc)
            docs.append(doc)

    diagnostics = {
        "question_id": qid,
        "num_intents": num_intents,
        "rrf_added_raw": min(top_b_rrf, len(rrf_articles)),
        "bge_added_raw": sum(min(top_n_bge, len(bge_by_intent.get(i, []))) for i in range(num_intents)),
        "intent_hits_added_raw": intent_raw_count,
        "intent_hits_source": "raw_per_intent" if intent_ranked_by_intent else "inferred_union",
        "output_size": len(articles),
    }
    return (
        {
            "id": qid,
            "question": question,
            "answer": "",
            "relevant_docs": docs,
            "relevant_articles": articles,
        },
        diagnostics,
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tiered recall-preserving union from clean RRF, BGE intent cache, and intent hits.")
    parser.add_argument("--input", default="../R2AIStage1DATA.json")
    parser.add_argument("--rrf60-submission", default=DEFAULT_RRF60_SUBMISSION)
    parser.add_argument("--intent-results", default=DEFAULT_INTENT_RESULTS)
    parser.add_argument("--intent-ranked-results", default=DEFAULT_INTENT_RANKED_RESULTS)
    parser.add_argument("--bge-cache", default=DEFAULT_BGE_CACHE)
    parser.add_argument("--output-dir", default="outputs/submission_bge_intent_tiered_rrf8_bge3_intent3_clean")
    parser.add_argument("--diagnostics", default="outputs/diagnostics_bge_intent_tiered_rrf8_bge3_intent3_clean.csv")
    parser.add_argument("--top-b-rrf", type=int, default=8)
    parser.add_argument("--top-n-bge", type=int, default=3)
    parser.add_argument("--top-m-intent", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = list(read_json(Path(args.input)))
    if args.limit:
        questions = questions[: args.limit]
    rrf_by_qid = load_submission_articles(Path(args.rrf60_submission))
    intents_by_qid, hits_by_qid = load_intent_results(Path(args.intent_results))
    raw_intents_by_qid, ranked_hits_by_qid = load_intent_ranked_results(Path(args.intent_ranked_results))
    if raw_intents_by_qid:
        intents_by_qid.update(raw_intents_by_qid)
    bge_by_qid = load_bge_cache(Path(args.bge_cache))

    submission: list[dict] = []
    diagnostics: list[dict] = []
    for q in questions:
        qid = q["id"]
        qid_key = str(qid)
        num_intents = len(intents_by_qid.get(qid, []))
        entry, diag = build_entry(
            qid=qid,
            question=q["question"],
            rrf_articles=rrf_by_qid.get(qid, []),
            bge_by_intent=bge_by_qid.get(qid_key, {}),
            intent_hits=hits_by_qid.get(qid, []),
            intent_ranked_by_intent=ranked_hits_by_qid.get(qid, {}),
            num_intents=num_intents,
            top_b_rrf=args.top_b_rrf,
            top_n_bge=args.top_n_bge,
            top_m_intent=args.top_m_intent,
        )
        submission.append(entry)
        diagnostics.append(diag)

    save_submission(submission, args.output_dir)
    diag_path = Path(args.diagnostics)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with diag_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "num_intents",
                "rrf_added_raw",
                "bge_added_raw",
                "intent_hits_added_raw",
                "intent_hits_source",
                "output_size",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostics)

    sizes = [len(row["relevant_articles"]) for row in submission]
    print(
        "tiered union stats: "
        f"questions={len(sizes)} min={min(sizes)} max={max(sizes)} "
        f"mean={statistics.mean(sizes):.3f} median={statistics.median(sizes):.1f}"
    )
    print(f"submission: {Path(args.output_dir) / 'submission.zip'}")
    print(f"diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
