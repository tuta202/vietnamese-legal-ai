from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def norm_text(value: str) -> str:
    return " ".join((value or "").split())


def parse_article_ref(article_ref: str) -> tuple[str, str]:
    """Return the stable lookup key `(law_id, dieu_number)` from a submission article ref."""
    law_id, sep, rest = (article_ref or "").partition("|")
    if not sep:
        raise ValueError(f"Invalid article ref: {article_ref!r}")
    _middle, sep, dieu_number = rest.rpartition("|")
    if not sep:
        raise ValueError(f"Invalid article ref: {article_ref!r}")
    return law_id.strip(), dieu_number.strip()


class ArticleLookup:
    """
    Hydrate article metadata/content from corpus_clean using stable keys.

    `relevant_article_str` is unique, but exact string joins can fail across
    artifacts after whitespace normalization in law names. `(law_id, dieu_number)`
    is unique in the current clean corpus and is safer for internal hydration.
    """

    def __init__(self, corpus_path: Path | str = "corpus/data/corpus_clean.json") -> None:
        self.corpus_path = Path(corpus_path)
        raw = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        articles = raw.get("articles", raw) if isinstance(raw, dict) else raw
        self.by_key: dict[tuple[str, str], dict[str, Any]] = {}
        duplicates: list[tuple[str, str]] = []
        for article in articles:
            key = (article.get("law_id", "").strip(), article.get("dieu_number", "").strip())
            if not key[0] or not key[1]:
                continue
            if key in self.by_key:
                duplicates.append(key)
                continue
            self.by_key[key] = article
        if duplicates:
            sample = ", ".join(f"{law_id}|{dieu}" for law_id, dieu in duplicates[:5])
            raise ValueError(
                f"Duplicate (law_id, dieu_number) keys in {self.corpus_path}: {sample}"
            )
        self.law_ids = {law_id for law_id, _dieu_number in self.by_key}

    def get(self, article_ref: str) -> dict[str, Any] | None:
        try:
            return self.by_key.get(parse_article_ref(article_ref))
        except ValueError:
            return None

    def require(self, article_ref: str) -> dict[str, Any]:
        article = self.get(article_ref)
        if article is None:
            raise KeyError(f"Article not found by (law_id, dieu_number): {article_ref}")
        return article

    def verifier_candidate(
        self,
        article_ref: str,
        content_max_chars: int = 1800,
    ) -> dict[str, str] | None:
        article = self.get(article_ref)
        if article is None:
            return None
        return {
            "article_id": article_ref,
            "law_type": norm_text(article.get("law_type", "")),
            "law_name": norm_text(article.get("law_name", "")),
            "article_number": norm_text(article.get("dieu_number", "")),
            "article_title": norm_text(article.get("dieu_title", "")),
            "article_content": norm_text(article.get("content", ""))[:content_max_chars],
        }
