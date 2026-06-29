"""
coverage_probe.py — proxy "retrievability" probe (TIP-CORPUS-001 AC-8).

For a given corpus + the 2000-question test set, builds a BM25 index (reusing
retrieval/bm25_index.py — not modified) and, for each question, records the
top-1 and mean-top-5 BM25 scores. Questions whose top-1 score falls below a
relative threshold (a low percentile of the top-1 distribution) are flagged as
"coverage gaps". Reports gap-rate and dumps the 30 lowest-scoring questions.

CAVEAT: This is a PROXY for retrievability, NOT recall. There is no gold here;
a high BM25 score only means *something* lexically matched, not that it is the
correct article. Use it to compare corpora relatively (old vs new), not as an
absolute recall figure.

CLI:
    # single corpus
    python corpus/coverage_probe.py --corpus corpus/data/corpus.json \
        --questions R2AIStage1DATA.json

    # compare old vs new (prints before/after gap-rate table)
    python corpus/coverage_probe.py --compare \
        --old corpus/data/corpus_v1_1044.json \
        --new corpus/data/corpus.json \
        --questions R2AIStage1DATA.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from legal_rag.retrieval.bm25_index import BM25Index  # noqa: E402

_PCTL = 10          # flag questions below this percentile of top-1 scores
_DUMP_N = 30        # dump N lowest-scoring questions


def _load_corpus_articles(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["articles"] if isinstance(data, dict) else data


def _load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # accept [{id, question}] or [{id, question, ...}]
    return data


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def probe(corpus_path: Path, questions: list[dict], label: str,
          shared_threshold: float | None = None) -> dict:
    """Probe a corpus. If shared_threshold is given (compare mode), the gap-rate
    is computed against that ABSOLUTE score so the two corpora are comparable;
    otherwise the threshold is this corpus's own p10 (single-corpus mode).

    NOTE on gap-rate: a per-corpus percentile threshold is ~p/100 by
    construction, so it cannot show improvement. In compare mode we derive the
    threshold once from the OLD corpus and apply the SAME absolute value to
    both — only then does a lower NEW gap-rate mean better retrievability.
    """
    articles = _load_corpus_articles(corpus_path)
    print(f"[{label}] building BM25 over {len(articles)} articles…", flush=True)
    idx = BM25Index().build_from_corpus(articles)

    rows = []
    for q in questions:
        question = q.get("question", "")
        hits = idx.search(question, top_k=5)
        top1 = hits[0][1] if hits else 0.0
        mean5 = statistics.fmean([s for _, s in hits]) if hits else 0.0
        rows.append({"id": q.get("id"), "question": question,
                     "top1": top1, "mean5": mean5})

    top1s = sorted(r["top1"] for r in rows)
    own_thresh = _percentile(top1s, _PCTL)
    thresh = shared_threshold if shared_threshold is not None else own_thresh
    flagged = [r for r in rows if r["top1"] <= thresh]
    gap_rate = len(flagged) / len(rows) if rows else 0.0

    return {
        "label": label,
        "corpus": str(corpus_path),
        "n_articles": len(articles),
        "n_questions": len(rows),
        "threshold_pctl": _PCTL,
        "threshold_value": round(thresh, 4),
        "own_p10": round(own_thresh, 4),
        "gap_rate": round(gap_rate, 4),
        "median_top1": round(statistics.median(top1s), 4) if top1s else 0.0,
        "mean_top1": round(statistics.fmean(top1s), 4) if top1s else 0.0,
        "rows": rows,
    }


def _dump_lowest(report: dict, out_path: Path) -> None:
    rows = sorted(report["rows"], key=lambda r: r["top1"])[:_DUMP_N]
    lines = [
        f"# Lowest-{_DUMP_N} top-1 BM25 questions — {report['label']}",
        f"# corpus={report['corpus']}  articles={report['n_articles']}",
        f"# gap_rate={report['gap_rate']} (<= p{report['threshold_pctl']} "
        f"= {report['threshold_value']})",
        "",
    ]
    for r in rows:
        lines.append(f"[{r['top1']:.3f}] id={r['id']}: {r['question']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{report['label']}] dumped {len(rows)} lowest → {out_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="BM25 proxy-coverage probe")
    ap.add_argument("--questions", required=True, help="2000-question JSON [{id,question}]")
    ap.add_argument("--corpus", help="single corpus.json to probe")
    ap.add_argument("--compare", action="store_true", help="compare --old vs --new")
    ap.add_argument("--old", help="old corpus.json (with --compare)")
    ap.add_argument("--new", help="new corpus.json (with --compare)")
    ap.add_argument("--out-dir", default=str(_HERE / "data"))
    args = ap.parse_args()

    questions = _load_questions(Path(args.questions))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("  PROXY-COVERAGE PROBE  (BM25 retrievability — NOT recall)")
    print("=" * 64)

    if args.compare:
        if not (args.old and args.new):
            ap.error("--compare requires --old and --new")
        rep_old = probe(Path(args.old), questions, "OLD")
        # Apply OLD's p10 as a shared ABSOLUTE threshold to NEW so gap-rate is comparable
        rep_new = probe(Path(args.new), questions, "NEW",
                        shared_threshold=rep_old["own_p10"])
        _dump_lowest(rep_old, out_dir / "coverage_gaps_old.txt")
        _dump_lowest(rep_new, out_dir / "coverage_gaps_new.txt")
        (out_dir / "coverage_report.json").write_text(
            json.dumps({"old": {k: v for k, v in rep_old.items() if k != "rows"},
                        "new": {k: v for k, v in rep_new.items() if k != "rows"}},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n" + "-" * 64)
        print(f"  shared abs. threshold (OLD p10) = {rep_old['own_p10']}")
        print(f"  {'metric':<24}{'OLD':>13}{'NEW':>13}")
        print("-" * 64)
        print(f"  {'articles':<24}{rep_old['n_articles']:>13}{rep_new['n_articles']:>13}")
        print(f"  {'gap-rate (shared thr)':<24}{rep_old['gap_rate']:>13}{rep_new['gap_rate']:>13}")
        print(f"  {'median top-1':<24}{rep_old['median_top1']:>13}{rep_new['median_top1']:>13}")
        print(f"  {'mean top-1':<24}{rep_old['mean_top1']:>13}{rep_new['mean_top1']:>13}")
        print("-" * 64)
        better = rep_new["gap_rate"] <= rep_old["gap_rate"]
        richer = rep_new["mean_top1"] >= rep_old["mean_top1"]
        print(f"  AC-8 gap-rate NEW <= OLD (shared thr) → {'PASS ✓' if better else 'FAIL ✗'}")
        print(f"  retrievability (mean top-1) NEW >= OLD → {'PASS ✓' if richer else 'FAIL ✗'}")
        print("=" * 64)
    else:
        if not args.corpus:
            ap.error("provide --corpus or --compare")
        rep = probe(Path(args.corpus), questions, "CORPUS")
        _dump_lowest(rep, out_dir / "coverage_gaps.txt")
        print(f"\n  articles={rep['n_articles']}  gap-rate(p10)={rep['gap_rate']}  "
              f"median_top1={rep['median_top1']}")
        print("=" * 64)


if __name__ == "__main__":
    main()
