"""
Evaluator — computes article-level Precision / Recall / F2 against an eval set.

F2 weights recall 4x over precision (matching the competition metric).
Article matching is normalised to (law_id, dieu_number), ignoring the law-name
component so cosmetic name differences do not cause false negatives.

CLI:
    python eval/evaluator.py --eval-set eval/data/eval_set.json --predictions results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_DIEU_RE = re.compile(r"Điều\s+\d+[a-zđ]?", re.UNICODE)


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------

def normalize_article(s: str) -> tuple[str, str]:
    """
    Reduce 'law_id|type name|Điều X' to (law_id, 'Điều X').

    Ignores the middle law-name segment so differing names still match.
    Returns ("", "") for unparseable strings.
    """
    parts = [p.strip() for p in s.split("|")]
    if len(parts) < 2:
        return ("", "")
    law_id = parts[0]
    # Last segment holds the article; extract the canonical "Điều X"
    m = _DIEU_RE.search(parts[-1])
    dieu = m.group(0) if m else parts[-1]
    # Normalise internal whitespace in "Điều  4" → "Điều 4"
    dieu = re.sub(r"\s+", " ", dieu).strip()
    return (law_id, dieu)


def _normset(items: list[str]) -> set[tuple[str, str]]:
    out = set()
    for it in items:
        key = normalize_article(it)
        if key != ("", ""):
            out.add(key)
    return out


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def compute_f2(precision: float, recall: float) -> float:
    """F-beta with beta=2: (5 * P * R) / (4 * P + R). Returns 0 if P=R=0."""
    denom = 4 * precision + recall
    if denom == 0:
        return 0.0
    return (5 * precision * recall) / denom


def score_query(
    predicted: list[str], expected: list[str]
) -> dict:
    """Compute precision/recall/F2 for one query, plus correct/missed/extra sets."""
    pred = _normset(predicted)
    exp  = _normset(expected)

    correct = pred & exp
    missed  = exp - pred
    extra   = pred - exp

    precision = len(correct) / len(pred) if pred else 0.0
    recall    = len(correct) / len(exp) if exp else 0.0
    f2        = compute_f2(precision, recall)

    fmt = lambda s: [f"{lid}|{d}" for lid, d in sorted(s)]
    return {
        "precision": precision,
        "recall":    recall,
        "f2":        f2,
        "correct":   fmt(correct),
        "missed":    fmt(missed),
        "extra":     fmt(extra),
    }


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------

class Evaluator:
    """Evaluates pipeline predictions against a synthetic eval set."""

    def __init__(self, eval_set_path: str | Path) -> None:
        self.eval_set: list[dict] = json.loads(
            Path(eval_set_path).read_text(encoding="utf-8")
        )
        self._by_id = {e["id"]: e for e in self.eval_set}
        self._report: dict = {}

    # ------------------------------------------------------------------

    def evaluate(self, predictions: list[dict]) -> dict:
        """
        Compare predictions to the eval set.

        predictions: [{id, question, answer, relevant_docs, relevant_articles}]
        Returns the full report dict.
        """
        per_query = []
        f2s, ps, rs = [], [], []
        covered = 0
        evaluated = 0

        for pred in predictions:
            qid = pred["id"]
            gold = self._by_id.get(qid)
            if gold is None:
                continue   # prediction for a question not in eval set
            evaluated += 1

            expected_articles = gold["expected_articles"]
            predicted_articles = pred.get("relevant_articles", [])

            qscore = score_query(predicted_articles, expected_articles)
            f2s.append(qscore["f2"])
            ps.append(qscore["precision"])
            rs.append(qscore["recall"])

            # Answer coverage: does answer mention >=1 CORRECT article number?
            if self._answer_covers(pred.get("answer", ""), expected_articles):
                covered += 1

            per_query.append({
                "id":        qid,
                "f2":        round(qscore["f2"], 4),
                "precision": round(qscore["precision"], 4),
                "recall":    round(qscore["recall"], 4),
                "predicted": predicted_articles,
                "expected":  expected_articles,
                "correct":   qscore["correct"],
                "missed":    qscore["missed"],
                "extra":     qscore["extra"],
            })

        n = evaluated or 1
        macro_f2 = sum(f2s) / n
        macro_p  = sum(ps) / n
        macro_r  = sum(rs) / n
        coverage = covered / n

        self._report = {
            "macro_f2":        round(macro_f2, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall":    round(macro_r, 4),
            "answer_coverage": round(coverage, 4),
            "n_evaluated":     evaluated,
            "per_query":       per_query,
            "summary": (
                f"F2={macro_f2:.4f} P={macro_p:.4f} "
                f"R={macro_r:.4f} Coverage={coverage*100:.0f}%"
            ),
        }
        return self._report

    # ------------------------------------------------------------------

    @staticmethod
    def _answer_covers(answer: str, expected_articles: list[str]) -> bool:
        """True if the answer mentions >=1 expected article's 'Điều X'."""
        answer_dieus = {
            re.sub(r"\s+", " ", m.group(0)).strip()
            for m in _DIEU_RE.finditer(answer)
        }
        if not answer_dieus:
            return False
        expected_dieus = {normalize_article(a)[1] for a in expected_articles}
        return bool(answer_dieus & expected_dieus)

    # ------------------------------------------------------------------

    def print_report(self) -> None:
        if not self._report:
            print("No report — call evaluate() first.")
            return
        r = self._report
        print("=" * 60)
        print("  EVAL REPORT")
        print("=" * 60)
        print(f"  Questions evaluated : {r['n_evaluated']}")
        print(f"  Macro F2            : {r['macro_f2']:.4f}")
        print(f"  Macro Precision     : {r['macro_precision']:.4f}")
        print(f"  Macro Recall        : {r['macro_recall']:.4f}")
        print(f"  Answer Coverage     : {r['answer_coverage']*100:.0f}%")
        print("-" * 60)
        print(f"  {r['summary']}")
        print("=" * 60)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate predictions vs eval set")
    parser.add_argument("--eval-set", required=True, help="eval_set.json path")
    parser.add_argument("--predictions", required=True, help="results.json path")
    parser.add_argument("--output", default=None, help="Optional report output path")
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    evaluator = Evaluator(args.eval_set)
    evaluator.evaluate(predictions)
    evaluator.print_report()

    if args.output:
        out = evaluator.save(args.output)
        print(f"\nReport saved → {out}")


if __name__ == "__main__":
    main()
