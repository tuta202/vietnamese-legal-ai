"""
SyntheticQAGenerator — rule-based synthetic eval-set builder.

The competition ships no train/dev set, so we synthesise legal questions
directly from the parsed corpus and keep the ground-truth article references.
No LLM is used in v1: deterministic templates + a fixed random seed make the
output fully reproducible.

CLI:
    python eval/synthetic_qa.py --n 50 --output eval/data/eval_set.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CORPUS = _ROOT / "corpus" / "data" / "corpus.json"
_DEFAULT_OUTPUT = _ROOT / "eval" / "data" / "eval_set.json"

_SEED = 42

# Strategy mix (must sum to 1.0)
_RATIO_SINGLE = 0.60
_RATIO_CROSS  = 0.30
_RATIO_MULTI  = 0.10

# Vietnamese stop-words for keyword overlap in cross-doc pairing
_STOPWORDS = {
    "và", "của", "các", "có", "được", "theo", "về", "tại", "cho", "trong",
    "đối", "với", "là", "một", "những", "này", "đó", "khi", "để", "từ",
    "hoặc", "như", "thì", "mà", "do", "bị", "ra", "vào", "ở", "nếu",
}

# Single-article templates: {topic} is filled from title/content
_SINGLE_TEMPLATES = [
    "Theo quy định pháp luật, {topic} như thế nào?",
    "{topic} được quy định ra sao?",
    "Quy định về {topic} là gì?",
    "Pháp luật quy định thế nào về {topic}?",
]

# Cross-document templates: {topic}, {law1}, {law2}
_CROSS_TEMPLATES = [
    "Quy định về {topic} theo {law1} và {law2} như thế nào?",
    "{topic} được quy định và hướng dẫn chi tiết tại những văn bản nào?",
]

# Multi-article (same law) templates
_MULTI_TEMPLATES = [
    "Các quy định liên quan đến {topic} được quy định như thế nào?",
    "Tiêu chí và điều kiện về {topic} được quy định ra sao?",
]


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) > 1]


def _keywords(title: str) -> set[str]:
    return {t for t in _tokenize(title) if t not in _STOPWORDS}


class SyntheticQAGenerator:
    """Generates a reproducible synthetic eval set from the corpus."""

    def __init__(self, corpus_path: str | Path = _DEFAULT_CORPUS) -> None:
        data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        self.articles: list[dict] = data["articles"]
        self._by_law: dict[str, list[dict]] = defaultdict(list)
        for art in self.articles:
            self._by_law[art["law_id"]].append(art)
        self._eval_set: list[dict] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_eval_set(self, n_questions: int = 50) -> list[dict]:
        """Build n_questions synthetic QA entries (deterministic with seed=42)."""
        rng = random.Random(_SEED)

        n_single = round(n_questions * _RATIO_SINGLE)
        n_cross  = round(n_questions * _RATIO_CROSS)
        n_multi  = n_questions - n_single - n_cross   # remainder → multi

        entries: list[dict] = []
        entries.extend(self._gen_single(rng, n_single))
        entries.extend(self._gen_cross(rng, n_cross))
        entries.extend(self._gen_multi(rng, n_multi))

        # Assign sequential integer ids (stable order: single, cross, multi)
        for i, e in enumerate(entries, start=1):
            e["id"] = i

        self._eval_set = entries
        return entries

    def save(self, path: str | Path = _DEFAULT_OUTPUT) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._eval_set, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    # ------------------------------------------------------------------
    # Topic extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_topic(article: dict) -> str:
        """Return a topic phrase: prefer dieu_title, else first sentence of content."""
        title = (article.get("dieu_title") or "").strip()
        if title:
            # Lower the first char for natural insertion into templates
            return title[0].lower() + title[1:] if len(title) > 1 else title.lower()
        return SyntheticQAGenerator.extract_noun_phrase(article)

    @staticmethod
    def extract_noun_phrase(article: dict) -> str:
        """Take the first ~20 words of content, cut at the first period."""
        content = (article.get("content") or "").strip()
        if not content:
            return "quy định pháp luật"
        # Cut at first sentence boundary
        first = re.split(r"[.\n]", content, maxsplit=1)[0]
        words = first.split()[:20]
        phrase = " ".join(words).strip()
        return phrase.lower() if phrase else "quy định pháp luật"

    # ------------------------------------------------------------------
    # Strategy A — single article
    # ------------------------------------------------------------------

    def _gen_single(self, rng: random.Random, n: int) -> list[dict]:
        # Pick articles that have a usable title for higher-quality questions
        candidates = [a for a in self.articles if (a.get("dieu_title") or "").strip()]
        rng.shuffle(candidates)
        chosen = candidates[:n]

        out = []
        for art in chosen:
            topic = self.extract_topic(art)
            template = rng.choice(_SINGLE_TEMPLATES)
            question = template.format(topic=topic)
            out.append({
                "question": question,
                "expected_articles": [art["relevant_article_str"]],
                "expected_docs": [art["relevant_doc_str"]],
                "strategy": "single",
                "source_articles": [art["chunk_id"]],
            })
        return out

    # ------------------------------------------------------------------
    # Strategy B — cross-document (different law_id, shared keywords)
    # ------------------------------------------------------------------

    def _gen_cross(self, rng: random.Random, n: int) -> list[dict]:
        pairs = self._find_cross_pairs(rng, n)
        out = []
        for art1, art2 in pairs:
            topic = self.extract_topic(art1)
            template = rng.choice(_CROSS_TEMPLATES)
            law1 = f"{art1['law_type']} {art1['law_name']}"
            law2 = f"{art2['law_type']} {art2['law_name']}"
            question = template.format(topic=topic, law1=law1, law2=law2)
            out.append({
                "question": question,
                "expected_articles": [
                    art1["relevant_article_str"],
                    art2["relevant_article_str"],
                ],
                "expected_docs": list(dict.fromkeys([
                    art1["relevant_doc_str"],
                    art2["relevant_doc_str"],
                ])),
                "strategy": "cross",
                "source_articles": [art1["chunk_id"], art2["chunk_id"]],
            })
        return out

    def _find_cross_pairs(
        self, rng: random.Random, n: int
    ) -> list[tuple[dict, dict]]:
        """Find pairs of articles from different laws sharing >=2 title keywords."""
        # Pre-compute keyword sets per article (only titled ones)
        titled = [a for a in self.articles if (a.get("dieu_title") or "").strip()]
        rng.shuffle(titled)

        kw_cache = {a["chunk_id"]: _keywords(a["dieu_title"]) for a in titled}

        pairs: list[tuple[dict, dict]] = []
        used_chunks: set[str] = set()

        for i, art1 in enumerate(titled):
            if len(pairs) >= n:
                break
            if art1["chunk_id"] in used_chunks:
                continue
            kw1 = kw_cache[art1["chunk_id"]]
            if len(kw1) < 2:
                continue
            for art2 in titled[i + 1:]:
                if art2["chunk_id"] in used_chunks:
                    continue
                if art2["law_id"] == art1["law_id"]:
                    continue
                shared = kw1 & kw_cache[art2["chunk_id"]]
                if len(shared) >= 2:
                    pairs.append((art1, art2))
                    used_chunks.add(art1["chunk_id"])
                    used_chunks.add(art2["chunk_id"])
                    break

        return pairs[:n]

    # ------------------------------------------------------------------
    # Strategy C — multiple consecutive articles in the same law
    # ------------------------------------------------------------------

    def _gen_multi(self, rng: random.Random, n: int) -> list[dict]:
        out = []
        # Laws with enough articles for consecutive sampling
        eligible_laws = [lid for lid, arts in self._by_law.items() if len(arts) >= 3]
        rng.shuffle(eligible_laws)

        attempts = 0
        li = 0
        while len(out) < n and attempts < n * 10:
            attempts += 1
            law_id = eligible_laws[li % len(eligible_laws)]
            li += 1
            arts = self._by_law[law_id]
            # Pick a random start with room for a 2-3 article window
            window = rng.choice([2, 3])
            if len(arts) < window:
                continue
            start = rng.randint(0, len(arts) - window)
            group = arts[start:start + window]

            topic = self.extract_topic(group[0])
            template = rng.choice(_MULTI_TEMPLATES)
            question = template.format(topic=topic)
            out.append({
                "question": question,
                "expected_articles": [a["relevant_article_str"] for a in group],
                "expected_docs": list(dict.fromkeys(
                    a["relevant_doc_str"] for a in group
                )),
                "strategy": "multi",
                "source_articles": [a["chunk_id"] for a in group],
            })
        return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate synthetic eval set")
    parser.add_argument("--n", type=int, default=50, help="Number of questions")
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS), help="Corpus path")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT), help="Output path")
    args = parser.parse_args()

    gen = SyntheticQAGenerator(args.corpus)
    eval_set = gen.generate_eval_set(args.n)
    out_path = gen.save(args.output)

    # Distribution report
    from collections import Counter
    dist = Counter(e["strategy"] for e in eval_set)
    print(f"Generated {len(eval_set)} questions → {out_path}")
    print("Strategy distribution:")
    for strat in ("single", "cross", "multi"):
        count = dist.get(strat, 0)
        pct = count / len(eval_set) * 100 if eval_set else 0
        print(f"  {strat:<8} {count:>3}  ({pct:.0f}%)")


if __name__ == "__main__":
    main()
