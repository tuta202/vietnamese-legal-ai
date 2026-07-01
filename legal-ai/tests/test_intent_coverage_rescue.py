from __future__ import annotations

from legal_rag.orchestration.best_pipeline import STAGE_NAMES, make_paths
from legal_rag.verification.intent_coverage_rescue import rescue_question


def intent(*articles: str, intent_id: int = 1) -> dict:
    return {
        "intent_index": intent_id - 1,
        "intent_id": intent_id,
        "intent": f"intent {intent_id}",
        "ranked_articles": [
            {"rank": rank, "article": article, "rrf_score": 1 / rank}
            for rank, article in enumerate(articles, start=1)
        ],
    }


def test_does_not_rescue_when_final_has_top3_intent_coverage():
    final, rescued = rescue_question(
        ["A3"],
        ["A1", "A2", "A3"],
        [intent("A1", "A2", "A3")],
    )

    assert final == ["A3"]
    assert rescued == []


def test_does_not_restore_article_removed_before_final_verifier():
    final, rescued = rescue_question(
        ["Z"],
        ["Z"],
        [intent("A1", "A2", "A3")],
    )

    assert final == ["Z"]
    assert rescued == []


def test_rescues_each_missing_intent_top1_and_deduplicates_shared_top1():
    final, rescued = rescue_question(
        ["Z"],
        ["Z", "A1", "B1"],
        [
            intent("A1", "A2", "A3", intent_id=1),
            intent("A1", "C2", "C3", intent_id=2),
            intent("B1", "B2", "B3", intent_id=3),
        ],
    )

    assert final == ["Z", "A1", "B1"]
    assert [row["article"] for row in rescued] == ["A1", "B1"]


def test_pipeline_runs_rescue_before_generation_and_isolates_depth_outputs(tmp_path):
    paths = make_paths(tmp_path, rescue_coverage_depth=4)

    assert STAGE_NAMES[-3:] == (
        "09_enforcement_role_gate",
        "10_intent_coverage_rescue",
        "11_answer_generation",
    )
    assert paths.rescue_results.parent.name.endswith("depth4")
    assert paths.generated_results.parent.name == "final_answers_rescue_depth4"
