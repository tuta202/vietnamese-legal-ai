from __future__ import annotations

import pytest

from legal_rag.orchestration.best_pipeline import prepare_manifest


def fingerprint(*, input_hash="input", code_hash="code", runtime_hash="runtime"):
    return {
        "input_sha256": input_hash,
        "corpus_sha256": "corpus",
        "bm25_sha256": "bm25",
        "config_sha256": "config",
        "runtime_target_sha256": runtime_hash,
        "code_sha256": {"module.py": code_hash},
    }


def test_manifest_rejects_code_change_by_default(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"setting": 1})

    with pytest.raises(ValueError, match="accept-code-change"):
        prepare_manifest(path, fingerprint(code_hash="fixed"), {"setting": 1})


def test_manifest_accepts_explicit_code_only_change(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"setting": 1})

    manifest = prepare_manifest(
        path,
        fingerprint(code_hash="fixed"),
        {"setting": 1},
        accept_code_change=True,
    )

    assert manifest["fingerprint"]["code_sha256"] == {"module.py": "fixed"}
    assert len(manifest["code_change_history"]) == 1


def test_manifest_never_accepts_non_code_change(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"setting": 1})

    with pytest.raises(ValueError):
        prepare_manifest(
            path,
            fingerprint(input_hash="different", code_hash="fixed"),
            {"setting": 1},
            accept_code_change=True,
        )


def test_manifest_does_not_mutate_when_settings_changed(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"setting": 1})
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="workflow settings differ"):
        prepare_manifest(
            path,
            fingerprint(code_hash="fixed"),
            {"setting": 2},
            accept_code_change=True,
        )

    assert path.read_text(encoding="utf-8") == before


def test_manifest_accepts_explicit_runtime_only_change(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"setting": 1})

    manifest = prepare_manifest(
        path,
        fingerprint(runtime_hash="redeployed"),
        {"setting": 1},
        accept_runtime_change=True,
    )

    assert manifest["fingerprint"]["runtime_target_sha256"] == "redeployed"
    assert len(manifest["runtime_change_history"]) == 1


def test_manifest_runtime_flag_does_not_accept_input_change(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"setting": 1})

    with pytest.raises(ValueError, match="input_sha256"):
        prepare_manifest(
            path,
            fingerprint(input_hash="different", runtime_hash="redeployed"),
            {"setting": 1},
            accept_runtime_change=True,
        )


def test_manifest_requires_both_flags_for_code_and_runtime_change(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"setting": 1})

    with pytest.raises(ValueError, match="code_sha256"):
        prepare_manifest(
            path,
            fingerprint(code_hash="fixed", runtime_hash="redeployed"),
            {"setting": 1},
            accept_runtime_change=True,
        )

    manifest = prepare_manifest(
        path,
        fingerprint(code_hash="fixed", runtime_hash="redeployed"),
        {"setting": 1},
        accept_code_change=True,
        accept_runtime_change=True,
    )
    assert len(manifest["code_change_history"]) == 1
    assert len(manifest["runtime_change_history"]) == 1


def test_manifest_accepts_explicit_workflow_change(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"workflow": "three-verifiers"})

    manifest = prepare_manifest(
        path,
        fingerprint(),
        {"workflow": "two-verifiers"},
        accept_workflow_change=True,
    )

    assert manifest["settings"] == {"workflow": "two-verifiers"}
    assert len(manifest["workflow_change_history"]) == 1


def test_manifest_rejects_unapproved_workflow_change_without_mutation(tmp_path):
    path = tmp_path / "manifest.json"
    prepare_manifest(path, fingerprint(), {"workflow": "three-verifiers"})
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="accept-workflow-change"):
        prepare_manifest(path, fingerprint(), {"workflow": "two-verifiers"})

    assert path.read_text(encoding="utf-8") == before
