from __future__ import annotations

from legal_rag.output.phase_snapshot import build_snapshot_rows, round_robin_top_each


def test_round_robin_top_each_preserves_intent_coverage_and_deduplicates():
    rankings = [
        ["A", "B", "C"],
        ["D", "A", "E"],
    ]

    assert round_robin_top_each(rankings, 3) == ["A", "D", "B", "C", "E"]


def test_build_snapshot_rows_uses_question_order_and_document_refs():
    questions = [
        {"id": 2, "question": "Q2"},
        {"id": 1, "question": "Q1"},
    ]
    rankings = {
        "1": [["L1|Luật Một|Điều 1"]],
        "2": [["L2|Luật Hai|Điều 2", "L3|Luật Ba|Điều 3"]],
    }

    rows = build_snapshot_rows(questions, rankings, top_each=2)

    assert [row["id"] for row in rows] == [2, 1]
    assert rows[0]["relevant_docs"] == ["L2|Luật Hai", "L3|Luật Ba"]
    assert rows[0]["answer"] == ""
