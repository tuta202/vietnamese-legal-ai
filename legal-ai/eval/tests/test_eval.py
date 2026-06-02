"""
Unit tests for the synthetic eval generator and the evaluator.
All tests run OFFLINE; only the local corpus.json is required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.evaluator import (
    Evaluator,
    compute_f2,
    normalize_article,
    score_query,
)
from eval.synthetic_qa import SyntheticQAGenerator

_ROOT = Path(__file__).resolve().parent.parent.parent
_CORPUS = _ROOT / "corpus" / "data" / "corpus.json"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def generator() -> SyntheticQAGenerator:
    if not _CORPUS.exists():
        pytest.skip("corpus.json not found")
    return SyntheticQAGenerator(_CORPUS)


# ----------------------------------------------------------------------
# SyntheticQAGenerator tests
# ----------------------------------------------------------------------

class TestSyntheticQA:
    def test_count(self, generator):
        eval_set = generator.generate_eval_set(10)
        assert len(eval_set) == 10

    def test_format(self, generator):
        eval_set = generator.generate_eval_set(10)
        for entry in eval_set:
            assert "id" in entry
            assert "question" in entry
            assert "expected_articles" in entry
            assert "expected_docs" in entry
            assert "strategy" in entry
            assert "source_articles" in entry
            # expected_articles must be non-empty
            assert len(entry["expected_articles"]) >= 1
            # question must be substantial
            assert len(entry["question"]) > 10
            # strategy in allowed set
            assert entry["strategy"] in {"single", "cross", "multi"}

    def test_strategies_present(self, generator):
        eval_set = generator.generate_eval_set(50)
        strategies = {e["strategy"] for e in eval_set}
        assert "single" in strategies
        assert "cross" in strategies
        assert "multi" in strategies

    def test_reproducible(self, generator):
        gen2 = SyntheticQAGenerator(_CORPUS)
        set1 = generator.generate_eval_set(20)
        set2 = gen2.generate_eval_set(20)
        # Identical questions and ground truth across runs
        assert [e["question"] for e in set1] == [e["question"] for e in set2]
        assert [e["expected_articles"] for e in set1] == \
               [e["expected_articles"] for e in set2]

    def test_ids_sequential(self, generator):
        eval_set = generator.generate_eval_set(15)
        ids = [e["id"] for e in eval_set]
        assert ids == list(range(1, 16))

    def test_distribution_approx(self, generator):
        """AC-4: ~60/30/10 split within ±5% tolerance."""
        eval_set = generator.generate_eval_set(50)
        from collections import Counter
        dist = Counter(e["strategy"] for e in eval_set)
        n = len(eval_set)
        assert abs(dist["single"] / n - 0.60) <= 0.05
        assert abs(dist["cross"] / n - 0.30) <= 0.05
        assert abs(dist["multi"] / n - 0.10) <= 0.05

    def test_single_has_one_article(self, generator):
        eval_set = generator.generate_eval_set(50)
        singles = [e for e in eval_set if e["strategy"] == "single"]
        assert all(len(e["expected_articles"]) == 1 for e in singles)

    def test_cross_has_different_laws(self, generator):
        eval_set = generator.generate_eval_set(50)
        crosses = [e for e in eval_set if e["strategy"] == "cross"]
        for e in crosses:
            law_ids = {a.split("|")[0] for a in e["expected_articles"]}
            assert len(law_ids) >= 2, f"cross question lacks 2 laws: {e['id']}"


# ----------------------------------------------------------------------
# F2 metric tests
# ----------------------------------------------------------------------

class TestF2Metrics:
    def test_f2_perfect(self):
        s = score_query(
            ["04/2017/QH14|Luật A|Điều 4"],
            ["04/2017/QH14|Luật A|Điều 4"],
        )
        assert s["f2"] == 1.0
        assert s["precision"] == 1.0
        assert s["recall"] == 1.0

    def test_f2_zero(self):
        s = score_query(
            ["04/2017/QH14|Luật A|Điều 4"],
            ["59/2020/QH14|Luật B|Điều 7"],
        )
        assert s["f2"] == 0.0

    def test_f2_partial(self):
        # predicted = {A,B,C}, expected = {A,B,D} → P=2/3, R=2/3
        pred = [
            "59/2020/QH14|L|Điều 1",
            "59/2020/QH14|L|Điều 2",
            "59/2020/QH14|L|Điều 3",
        ]
        exp = [
            "59/2020/QH14|L|Điều 1",
            "59/2020/QH14|L|Điều 2",
            "59/2020/QH14|L|Điều 4",
        ]
        s = score_query(pred, exp)
        assert abs(s["precision"] - 2/3) < 1e-9
        assert abs(s["recall"] - 2/3) < 1e-9
        assert abs(s["f2"] - 0.6667) < 1e-3

    def test_f2_recall_weighted(self):
        # P=1/3 (1 correct, 2 extra), R=1/1 (1 correct, 1 expected)
        pred = [
            "59/2020/QH14|L|Điều 1",
            "59/2020/QH14|L|Điều 2",
            "59/2020/QH14|L|Điều 3",
        ]
        exp = ["59/2020/QH14|L|Điều 1"]
        s = score_query(pred, exp)
        assert abs(s["precision"] - 1/3) < 1e-9
        assert abs(s["recall"] - 1.0) < 1e-9
        # F2 = 5*(1/3)*1 / (4*(1/3)+1) = (5/3)/(7/3) = 5/7 ≈ 0.714
        assert abs(s["f2"] - 5/7) < 1e-3

        # Verify F2 > F1 (recall-weighting favours high recall)
        p, r = 1/3, 1.0
        f1 = 2 * p * r / (p + r)
        assert s["f2"] > f1

    def test_compute_f2_zero_division(self):
        assert compute_f2(0.0, 0.0) == 0.0

    def test_f2_only_recall_matters_when_p_high(self):
        # Symmetric check: low recall hurts F2 more than low precision
        s_low_recall    = score_query(
            ["59/2020/QH14|L|Điều 1"],
            ["59/2020/QH14|L|Điều 1", "59/2020/QH14|L|Điều 2",
             "59/2020/QH14|L|Điều 3"],
        )  # P=1, R=1/3
        s_low_precision = score_query(
            ["59/2020/QH14|L|Điều 1", "59/2020/QH14|L|Điều 2",
             "59/2020/QH14|L|Điều 3"],
            ["59/2020/QH14|L|Điều 1"],
        )  # P=1/3, R=1
        # F2 favours recall → low-precision case scores higher
        assert s_low_precision["f2"] > s_low_recall["f2"]


# ----------------------------------------------------------------------
# Evaluator aggregate tests
# ----------------------------------------------------------------------

class TestEvaluator:
    def _make_eval_set(self, tmp_path) -> Path:
        import json
        data = [
            {
                "id": 1,
                "question": "Q1?",
                "expected_articles": ["04/2017/QH14|Luật A|Điều 4"],
                "expected_docs": ["04/2017/QH14|Luật A"],
                "strategy": "single",
                "source_articles": ["c1"],
            },
            {
                "id": 2,
                "question": "Q2?",
                "expected_articles": ["59/2020/QH14|Luật B|Điều 7"],
                "expected_docs": ["59/2020/QH14|Luật B"],
                "strategy": "single",
                "source_articles": ["c2"],
            },
        ]
        path = tmp_path / "eval_set.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_macro_average(self, tmp_path):
        eval_path = self._make_eval_set(tmp_path)
        ev = Evaluator(eval_path)
        predictions = [
            {  # Q1 perfect → F2=1.0
                "id": 1, "question": "Q1?",
                "answer": "Theo Điều 4 ...",
                "relevant_docs": ["04/2017/QH14|Luật A"],
                "relevant_articles": ["04/2017/QH14|Luật A|Điều 4"],
            },
            {  # Q2 wrong → F2=0.0
                "id": 2, "question": "Q2?",
                "answer": "Theo Điều 99 ...",
                "relevant_docs": ["59/2020/QH14|Luật B"],
                "relevant_articles": ["59/2020/QH14|Luật B|Điều 99"],
            },
        ]
        report = ev.evaluate(predictions)
        # mean(1.0, 0.0) = 0.5
        assert abs(report["macro_f2"] - 0.5) < 1e-9

    def test_answer_coverage(self, tmp_path):
        eval_path = self._make_eval_set(tmp_path)
        ev = Evaluator(eval_path)
        predictions = [
            {  # answer mentions correct Điều 4 → covered
                "id": 1, "question": "Q1?",
                "answer": "Căn cứ Điều 4 của Luật A.",
                "relevant_docs": [],
                "relevant_articles": ["04/2017/QH14|Luật A|Điều 4"],
            },
            {  # answer has no Điều mention → not covered
                "id": 2, "question": "Q2?",
                "answer": "Không có trích dẫn.",
                "relevant_docs": [],
                "relevant_articles": [],
            },
        ]
        report = ev.evaluate(predictions)
        assert abs(report["answer_coverage"] - 0.5) < 1e-9

    def test_summary_string_present(self, tmp_path):
        eval_path = self._make_eval_set(tmp_path)
        ev = Evaluator(eval_path)
        report = ev.evaluate([
            {"id": 1, "question": "Q1?", "answer": "Điều 4",
             "relevant_docs": [], "relevant_articles": ["04/2017/QH14|Luật A|Điều 4"]},
        ])
        assert "F2=" in report["summary"]
        assert "Coverage=" in report["summary"]


# ----------------------------------------------------------------------
# Matching normalisation tests
# ----------------------------------------------------------------------

class TestMatching:
    def test_matching_ignores_law_name(self):
        # Same law_id + same Điều number, different name → must match
        s = score_query(
            ["04/2017/QH14|Luật HTDNNVV|Điều 4"],
            ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4"],
        )
        assert s["f2"] == 1.0
        assert s["recall"] == 1.0

    def test_normalize_article_basic(self):
        assert normalize_article("04/2017/QH14|Luật A|Điều 4") == \
               ("04/2017/QH14", "Điều 4")

    def test_normalize_article_whitespace(self):
        assert normalize_article("04/2017/QH14|Luật A|Điều  4") == \
               ("04/2017/QH14", "Điều 4")

    def test_normalize_article_letter_suffix(self):
        assert normalize_article("01/2021/NĐ-CP|Nghị định X|Điều 24a") == \
               ("01/2021/NĐ-CP", "Điều 24a")

    def test_normalize_different_law_id_no_match(self):
        s = score_query(
            ["04/2017/QH14|Luật A|Điều 4"],
            ["59/2020/QH14|Luật A|Điều 4"],   # same Điều, different law
        )
        assert s["f2"] == 0.0
