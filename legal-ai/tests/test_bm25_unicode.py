from __future__ import annotations

import unicodedata

import numpy as np
import pytest

from retrieval.bm25_index import BM25Index, vietnamese_simple_tokenize
from setup_qdrant_cloud import _validate_vectors


def test_tokenizer_normalizes_canonically_equivalent_vietnamese() -> None:
    nfc = "đăng ký sử dụng hóa đơn điện tử"
    nfd = unicodedata.normalize("NFD", nfc)

    assert nfc != nfd
    assert vietnamese_simple_tokenize(nfc) == vietnamese_simple_tokenize(nfd)


def test_bm25_returns_same_ranking_for_nfc_and_nfd_queries() -> None:
    articles = [
        {
            "chunk_id": "a",
            "dieu_title": "Đăng ký sử dụng hóa đơn điện tử",
            "content": "Người nộp thuế thực hiện đăng ký theo quy định.",
        },
        {
            "chunk_id": "b",
            "dieu_title": "Thời giờ làm việc",
            "content": "Thời giờ làm việc bình thường của người lao động.",
        },
    ]
    index = BM25Index().build_from_corpus(articles)
    query = "đăng ký hóa đơn điện tử"

    nfc_results = index.search(unicodedata.normalize("NFC", query), top_k=2)
    nfd_results = index.search(unicodedata.normalize("NFD", query), top_k=2)

    assert nfc_results == nfd_results
    assert nfc_results[0][0] == "a"


def test_embedding_batch_validation_rejects_wrong_shape_and_non_finite_values() -> None:
    valid = _validate_vectors([[0.1, 0.2], [0.3, 0.4]], expected_count=2, expected_dim=2)
    assert valid.shape == (2, 2)

    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_vectors([[0.1, 0.2]], expected_count=2, expected_dim=2)

    with pytest.raises(ValueError, match="NaN or Inf"):
        _validate_vectors(np.array([[0.1, np.nan]], dtype=np.float32), expected_count=1, expected_dim=2)
