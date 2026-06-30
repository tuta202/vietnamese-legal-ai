from __future__ import annotations

import importlib

from legal_rag.common.paths import PROJECT_ROOT


STAGE_MODULES = (
    "legal_rag.retrieval.query_analysis",
    "legal_rag.retrieval.global_rrf",
    "legal_rag.retrieval.intent_rrf",
    "legal_rag.ranking.intent_bge",
    "legal_rag.ranking.candidate_union",
    "legal_rag.verification.candidate_verifier",
    "legal_rag.verification.deterministic_cleanup",
    "legal_rag.verification.final_collective",
    "legal_rag.verification.enforcement_gate",
    "legal_rag.generation.generate_answers",
    "legal_rag.output.phase_snapshot",
)


def test_public_entry_points_remain_available():
    for filename in (
        "run_best_pipeline.py",
        "setup_qdrant_cloud.py",
        "build_core_bm25.py",
        "pipeline.py",
    ):
        assert (PROJECT_ROOT / filename).is_file()


def test_every_orchestrated_stage_is_importable():
    for module_name in STAGE_MODULES:
        module = importlib.import_module(module_name)
        assert callable(module.main)


def test_runtime_data_is_outside_source_package():
    source_root = PROJECT_ROOT / "legal_rag"
    for runtime_directory in ("corpus", "cache", "outputs", "data"):
        assert not (source_root / runtime_directory).exists()


def test_orchestrator_configures_utf8_console_before_parsing(monkeypatch):
    from legal_rag.orchestration import best_pipeline

    calls = []

    class Stream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(best_pipeline.sys, "stdout", Stream())
    monkeypatch.setattr(best_pipeline.sys, "stderr", Stream())

    class StopAfterParser:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("stop")

    monkeypatch.setattr(best_pipeline.argparse, "ArgumentParser", StopAfterParser)
    try:
        best_pipeline.main()
    except RuntimeError as exc:
        assert str(exc) == "stop"

    assert calls == [
        {"encoding": "utf-8", "errors": "replace"},
        {"encoding": "utf-8", "errors": "replace"},
    ]
