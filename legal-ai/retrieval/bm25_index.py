"""
BM25Index — Okapi BM25 over Vietnamese legal articles.

CLI:
    python retrieval/bm25_index.py
    → loads corpus/data/corpus.json, builds index, saves retrieval/data/bm25_index.pkl
    → runs AC-3 self-test: search "doanh nghiệp nhỏ và vừa", verify 04/2017/QH14 in top-5
"""
from __future__ import annotations

import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).resolve().parent.parent
_CORPUS_PATH = _ROOT / "corpus" / "data" / "corpus.json"
_INDEX_PATH = Path(__file__).resolve().parent / "data" / "bm25_index.pkl"


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def vietnamese_simple_tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace. Keeps Vietnamese chars."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 (Okapi BM25)
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Inverted-index-backed Okapi BM25 over Article dicts.
    Each article is indexed on: dieu_title + content.
    """

    K1: float = 1.5
    B: float = 0.75

    def __init__(self) -> None:
        self._doc_ids:       list[str]            = []
        self._payloads:      list[dict]            = []
        self._doc_tf:        list[dict[str, int]]  = []  # term freq per doc
        self._doc_lengths:   list[int]             = []
        self._idf:           dict[str, float]      = {}
        self._avgdl:         float                 = 0.0
        # Inverted index: term → list of (doc_idx, tf)
        self._inverted:      dict[str, list[tuple[int, int]]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_corpus(self, articles: list[dict]) -> "BM25Index":
        """Tokenise and index all articles. Returns self for chaining."""
        for art in articles:
            text = f"{art.get('dieu_title', '')} {art.get('content', '')}"
            tokens = vietnamese_simple_tokenize(text)
            tf = Counter(tokens)
            idx = len(self._doc_ids)
            self._doc_ids.append(art["chunk_id"])
            self._payloads.append(art)
            self._doc_tf.append(dict(tf))
            self._doc_lengths.append(len(tokens))
            for term, freq in tf.items():
                self._inverted.setdefault(term, []).append((idx, freq))

        N = len(self._doc_ids)
        if N == 0:
            return self

        self._avgdl = sum(self._doc_lengths) / N

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, postings in self._inverted.items():
            df = len(postings)
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return self

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return list of (chunk_id, bm25_score) sorted by score desc."""
        q_tokens = vietnamese_simple_tokenize(query)
        if not q_tokens or not self._doc_ids:
            return []

        scores: dict[int, float] = defaultdict(float)

        for token in q_tokens:
            if token not in self._inverted:
                continue
            idf = self._idf[token]
            for doc_idx, freq in self._inverted[token]:
                dl = self._doc_lengths[doc_idx]
                tf_norm = (freq * (self.K1 + 1)) / (
                    freq + self.K1 * (1.0 - self.B + self.B * dl / self._avgdl)
                )
                scores[doc_idx] += idf * tf_norm

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._doc_ids[i], s) for i, s in top]

    # ------------------------------------------------------------------
    # Payload lookup
    # ------------------------------------------------------------------

    def get_payload(self, chunk_id: str) -> dict | None:
        """Return full article dict for a chunk_id, or None."""
        try:
            idx = self._doc_ids.index(chunk_id)
            return self._payloads[idx]
        except ValueError:
            return None

    def __len__(self) -> int:
        return len(self._doc_ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str = _INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str = _INDEX_PATH) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_corpus(path: Path = _CORPUS_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["articles"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Loading corpus from {_CORPUS_PATH} …")
    articles = _load_corpus()
    print(f"  {len(articles)} articles loaded")

    print("Building BM25 index …")
    idx = BM25Index().build_from_corpus(articles)
    print(f"  Vocabulary: {len(idx._idf):,} terms  |  avgdl={idx._avgdl:.1f}")

    idx.save(_INDEX_PATH)
    print(f"  Saved → {_INDEX_PATH}")

    # AC-3 self-test
    print()
    print("AC-3 self-test: search('doanh nghiệp nhỏ và vừa', top_k=5)")
    results = idx.search("doanh nghiệp nhỏ và vừa", top_k=5)
    target_law = "04/2017/QH14"
    found_target = False
    for rank, (chunk_id, score) in enumerate(results, 1):
        payload = idx.get_payload(chunk_id)
        if payload is None:
            continue
        law_id  = payload.get("law_id", "")
        dieu    = payload.get("dieu_number", "")
        title   = payload.get("dieu_title", "")[:50]
        marker  = " ← TARGET" if law_id == target_law else ""
        print(f"  {rank}. [{law_id}] {dieu} — {title}  (score={score:.4f}){marker}")
        if law_id == target_law:
            found_target = True

    status = "PASS ✓" if found_target else "FAIL ✗"
    print(f"\nAC-3: {target_law} in top-5 → {status}")


if __name__ == "__main__":
    main()
