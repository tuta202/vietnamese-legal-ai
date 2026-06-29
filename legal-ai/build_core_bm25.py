from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.bm25_index import BM25Index  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build BM25 over the cleaned legal corpus.")
    parser.add_argument("--corpus", default="corpus/data/corpus_clean_asof_20260301.json")
    parser.add_argument("--output", default="retrieval/data/bm25_index_asof_20260301.pkl")
    parser.add_argument("--expected-count", type=int, default=82570)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    articles = json.loads(corpus_path.read_text(encoding="utf-8"))["articles"]

    print(f"corpus={corpus_path}")
    print(f"corpus_articles={len(articles)}")
    if args.expected_count and len(articles) != args.expected_count:
        raise SystemExit(
            f"Unexpected clean article count: {len(articles)} != {args.expected_count}"
        )

    index = BM25Index().build_from_corpus(articles)
    output = Path(args.output)
    index.save(output)
    print(f"saved={output}")
    print(f"vocabulary={len(index._idf)}")
    print(f"avgdl={index._avgdl:.2f}")


if __name__ == "__main__":
    main()
