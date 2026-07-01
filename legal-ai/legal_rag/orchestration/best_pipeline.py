from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from legal_rag.backends.bge import BgeScorer
from legal_rag.common.paths import PROJECT_ROOT
from legal_rag.common.qdrant import create_qdrant_client
from legal_rag.generation.prompt_builder import PROMPT_VERSION
from legal_rag.output.submission import validate_submission
from legal_rag.retrieval.bm25_index import BM25Index
from legal_rag.retrieval.config import load_config, validate_config
from legal_rag.retrieval.embedder import embedding_text_sha256
from legal_rag.retrieval.intent_decomposer import LegalIntentDecomposer


ROOT = PROJECT_ROOT
log = logging.getLogger("run_best_pipeline")

EXPECTED_QUESTIONS = 2000
EXPECTED_ARTICLES = 82570
GLOBAL_SOURCE_TOP_K = 350
GLOBAL_TOP_K = 60
INTENT_SOURCE_TOP_K = 50
INTENT_RRF_TOP_K = 10
TIER_RRF_TOP_K = 12
TIER_BGE_PER_INTENT = 5
TIER_RAW_INTENT_PER_INTENT = 5
STAGE_NAMES = (
    "01_query_analysis",
    "02_global_rrf",
    "03_raw_intent_retrieval",
    "04_bge_intent_rerank",
    "05_tiered_union",
    "06_stage1_compact",
    "07_penalty_cleanup",
    "08_final_collective",
    "09_enforcement_role_gate",
    "10_intent_coverage_rescue",
    "11_answer_generation",
)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    cache: Path
    artifacts: Path
    submissions: Path
    logs: Path
    errors: Path
    manifest: Path
    analysis: Path
    global_rrf: Path
    rrf60_results: Path
    intent_ranked: Path
    bge_cache: Path
    tiered_results: Path
    stage1_diagnostics: Path
    penalty_diagnostics: Path
    final_diagnostics: Path
    final_results: Path
    rescue_results: Path
    generated_results: Path


def make_paths(root: Path, *, rescue_coverage_depth: int = 4) -> RunPaths:
    cache = root / "cache"
    artifacts = root / "artifacts"
    submissions = root / "submissions"
    logs = root / "logs"
    return RunPaths(
        root=root,
        cache=cache,
        artifacts=artifacts,
        submissions=submissions,
        logs=logs,
        errors=root / "errors.jsonl",
        manifest=root / "manifest.json",
        analysis=artifacts / "query_analysis.json",
        global_rrf=artifacts / "global_rrf_top60.json",
        rrf60_results=submissions / "rrf_top60_clean" / "results.json",
        intent_ranked=artifacts / "intent_ranked_hits.json",
        bge_cache=cache / "bge_intent_scores.jsonl",
        tiered_results=submissions / "tiered_rrf12_bge5_rawintent5" / "results.json",
        stage1_diagnostics=artifacts / "stage1_gemma_compact_v4_diagnostics.json",
        penalty_diagnostics=submissions / "stage1_gemma_v4_penalty_cleanup" / "diagnostics.json",
        final_diagnostics=artifacts / "final_collective_diagnostics.json",
        final_results=submissions / "final_collective" / "results.json",
        rescue_results=(
            submissions
            / f"best_final_enforcement_gate_rawintent_top1_rescue_depth{rescue_coverage_depth}"
            / "results.json"
        ),
        generated_results=(
            submissions / f"final_answers_rescue_depth{rescue_coverage_depth}" / "results.json"
        ),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retry_operation(function: Callable[[], object], *, attempts: int = 5):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(30, 2 * attempt))
    assert last_error is not None
    raise last_error


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def acquire_run_lock(run_dir: Path) -> None:
    lock_path = run_dir / ".pipeline.running"
    if lock_path.exists():
        try:
            text = lock_path.read_text(encoding="utf-8")
            pid_line = next(line for line in text.splitlines() if line.startswith("pid="))
            pid = int(pid_line.partition("=")[2])
            os.kill(pid, 0)
        except (OSError, ValueError, StopIteration):
            lock_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Pipeline run is already active with PID {pid}: {lock_path}")
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(f"pid={os.getpid()}\nstarted={utc_now()}\n")
    atexit.register(lambda: lock_path.unlink(missing_ok=True))


def load_questions(path: Path, expected_count: int) -> list[dict]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("Input questions must be a JSON list")
    ids = [str(row.get("id")) for row in rows]
    if len(rows) != expected_count:
        raise ValueError(f"Input question count {len(rows)} != expected {expected_count}")
    if any(row.get("id") is None or not str(row.get("question") or "").strip() for row in rows):
        raise ValueError("Every input row must have a non-empty id and question")
    if len(set(ids)) != len(ids):
        raise ValueError("Input contains duplicate question IDs")
    return rows


def validate_question_rows(
    path: Path,
    expected_ids: set[str],
    *,
    require_articles: bool = False,
    required_fields: tuple[str, ...] = (),
    allowed_articles: set[str] | None = None,
) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing {path}"
    try:
        rows = read_json(path)
    except Exception as exc:
        return False, f"cannot read {path}: {exc}"
    if not isinstance(rows, list):
        return False, f"{path} is not a JSON list"
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        return False, f"{path} contains duplicate IDs"
    missing = expected_ids - set(ids)
    extra = set(ids) - expected_ids
    if missing or extra:
        return False, f"{path}: missing={len(missing)} extra={len(extra)}"
    for row in rows:
        if any(not row.get(field) for field in required_fields):
            return False, f"Q{row.get('id')} missing required fields {required_fields}"
        if require_articles and not row.get("relevant_articles"):
            return False, f"Q{row.get('id')} has no relevant_articles"
        if allowed_articles is not None:
            unknown = [
                article
                for article in row.get("relevant_articles", [])
                if " ".join(str(article).split()) not in allowed_articles
            ]
            if unknown:
                return False, f"Q{row.get('id')} contains articles outside corpus: {unknown[:3]}"
    return True, f"{len(rows)} complete rows"


def validate_intent_rows(
    path: Path,
    expected_ids: set[str],
    allowed_articles: set[str],
) -> tuple[bool, str]:
    ok, detail = validate_question_rows(
        path,
        expected_ids,
        required_fields=("legal_intents", "intent_ranked_hits", "intent_hits_union"),
    )
    if not ok:
        return ok, detail
    for row in read_json(path):
        intents = row["legal_intents"]
        ranked = row["intent_ranked_hits"]
        if not 1 <= len(intents) <= 6 or len(ranked) != len(intents):
            return False, f"Q{row['id']} intent/ranked cardinality mismatch"
        if any(not item.get("ranked_articles") for item in ranked):
            return False, f"Q{row['id']} has an empty per-intent ranking"
        for intent_row in ranked:
            unknown = [
                item.get("article")
                for item in intent_row["ranked_articles"]
                if " ".join(str(item.get("article") or "").split()) not in allowed_articles
            ]
            if unknown:
                return False, f"Q{row['id']} intent hits contain articles outside corpus: {unknown[:3]}"
    return True, f"{len(expected_ids)} complete intent rows"


def load_jsonl_latest(path: Path, key_fn: Callable[[dict], object]) -> dict[object, dict]:
    rows: dict[object, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
                key = key_fn(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            rows[key] = row
    return rows


def validate_bge_cache(
    path: Path,
    intent_path: Path,
    allowed_articles: set[str],
) -> tuple[bool, str]:
    if not intent_path.exists():
        return False, f"missing {intent_path}"
    intent_rows = read_json(intent_path)
    expected = {
        (str(row["id"]), index)
        for row in intent_rows
        for index, _ in enumerate(row.get("legal_intents", []))
    }
    cached = load_jsonl_latest(path, lambda row: (str(row["question_id"]), int(row["intent_index"])))
    valid = {
        key
        for key, row in cached.items()
        if row.get("ranked_articles") and row.get("scored_size") == row.get("keep_size")
    }
    missing = expected - valid
    if missing:
        return False, f"BGE cache missing/invalid jobs={len(missing)} examples={sorted(missing)[:10]}"
    for key in expected:
        unknown = [
            item.get("article")
            for item in cached[key].get("ranked_articles", [])
            if " ".join(str(item.get("article") or "").split()) not in allowed_articles
        ]
        if unknown:
            return False, f"BGE job {key} contains articles outside corpus: {unknown[:3]}"
    return True, f"{len(expected)} complete BGE intent jobs"


def run_process(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = subprocess.list2cmdline(command)
    log.info("Running: %s", printable)
    with log_path.open("a", encoding="utf-8") as logfile:
        logfile.write(f"\n[{utc_now()}] $ {printable}\n")
        logfile.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            logfile.write(line)
        return process.wait()


def run_resumable_stage(
    *,
    name: str,
    base_command: list[str],
    validator: Callable[[], tuple[bool, str]],
    log_path: Path,
    max_passes: int,
    manifest: dict,
    manifest_path: Path,
    errors_path: Path,
) -> None:
    valid, detail = validator()
    if valid:
        log.info("[%s] already complete: %s", name, detail)
        manifest["stages"][name] = {"status": "complete", "detail": detail, "updated_at": utc_now()}
        atomic_write_json(manifest_path, manifest)
        return

    manifest["stages"][name] = {"status": "running", "detail": detail, "updated_at": utc_now()}
    atomic_write_json(manifest_path, manifest)
    for pass_number in range(1, max_passes + 1):
        command = list(base_command)
        if "--resume" not in command:
            command.append("--resume")
        return_code = run_process(command, log_path)
        valid, detail = validator()
        if valid:
            manifest["stages"][name] = {
                "status": "complete",
                "detail": detail,
                "passes": pass_number,
                "updated_at": utc_now(),
            }
            atomic_write_json(manifest_path, manifest)
            return
        log.warning("[%s] pass %d incomplete (exit=%d): %s", name, pass_number, return_code, detail)
        append_jsonl(
            errors_path,
            {
                "timestamp": utc_now(),
                "stage": name,
                "pass": pass_number,
                "exit_code": return_code,
                "detail": detail,
            },
        )
        if pass_number < max_passes:
            time.sleep(min(60, 5 * (2 ** (pass_number - 1))))

    manifest["stages"][name] = {"status": "blocked", "detail": detail, "updated_at": utc_now()}
    atomic_write_json(manifest_path, manifest)
    raise RuntimeError(f"Stage {name} incomplete after {max_passes} passes: {detail}")


def validate_submission_zip(path: Path, expected_ids: set[str], allowed_articles: set[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing {path}"
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["results.json"]:
                return False, f"unexpected ZIP entries: {archive.namelist()}"
            rows = json.loads(archive.read("results.json").decode("utf-8"))
    except Exception as exc:
        return False, f"invalid submission ZIP: {exc}"
    temporary = path.parent / ".zip_validation_results.json"
    try:
        atomic_write_json(temporary, rows)
        return validate_question_rows(
            temporary,
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        )
    finally:
        temporary.unlink(missing_ok=True)


def validate_generated_rows(
    path: Path,
    expected_ids: set[str],
    allowed_articles: set[str],
) -> tuple[bool, str]:
    ok, detail = validate_question_rows(
        path,
        expected_ids,
        require_articles=True,
        required_fields=("question", "answer", "relevant_docs", "relevant_articles"),
        allowed_articles=allowed_articles,
    )
    if not ok:
        return ok, detail
    valid, errors = validate_submission(read_json(path))
    if not valid:
        return False, f"generated submission errors={len(errors)} examples={errors[:3]}"
    return True, f"{len(expected_ids)} complete grounded answers"


def validate_generation_stage(
    results_path: Path,
    metadata_path: Path,
    expected_ids: set[str],
    allowed_articles: set[str],
) -> tuple[bool, str]:
    ok, detail = validate_generated_rows(results_path, expected_ids, allowed_articles)
    if not ok:
        return ok, detail
    if not metadata_path.exists():
        return False, f"missing {metadata_path}"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid generation metadata: {exc}"
    if metadata.get("prompt_version") != PROMPT_VERSION:
        return False, (
            "generation prompt mismatch: "
            f"{metadata.get('prompt_version')!r} != {PROMPT_VERSION!r}"
        )
    if metadata.get("row_count") != len(expected_ids):
        return False, (
            "generation metadata row count mismatch: "
            f"{metadata.get('row_count')!r} != {len(expected_ids)}"
        )
    return True, detail


def validate_generated_zip(
    path: Path,
    expected_ids: set[str],
    allowed_articles: set[str],
) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing {path}"
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["results.json"]:
                return False, f"unexpected ZIP entries: {archive.namelist()}"
            rows = json.loads(archive.read("results.json").decode("utf-8"))
    except Exception as exc:
        return False, f"invalid generated submission ZIP: {exc}"
    temporary = path.parent / ".generated_zip_validation.json"
    try:
        atomic_write_json(temporary, rows)
        return validate_generated_rows(temporary, expected_ids, allowed_articles)
    finally:
        temporary.unlink(missing_ok=True)


def run_deterministic_stage(
    *,
    name: str,
    command: list[str],
    validator: Callable[[], tuple[bool, str]],
    log_path: Path,
    manifest: dict,
    manifest_path: Path,
    errors_path: Path,
) -> None:
    valid, detail = validator()
    if not valid:
        return_code = run_process(command, log_path)
        if return_code:
            append_jsonl(
                errors_path,
                {"timestamp": utc_now(), "stage": name, "exit_code": return_code},
            )
            raise RuntimeError(f"Stage {name} exited with code {return_code}")
        valid, detail = validator()
    if not valid:
        append_jsonl(
            errors_path,
            {"timestamp": utc_now(), "stage": name, "detail": detail},
        )
        raise RuntimeError(f"Stage {name} output validation failed: {detail}")
    manifest["stages"][name] = {"status": "complete", "detail": detail, "updated_at": utc_now()}
    atomic_write_json(manifest_path, manifest)


def ensure_phase_submission(
    *,
    command: list[str],
    results_path: Path,
    expected_ids: set[str],
    allowed_articles: set[str],
    log_path: Path,
) -> None:
    valid, detail = validate_question_rows(
        results_path,
        expected_ids,
        require_articles=True,
        allowed_articles=allowed_articles,
    )
    if valid:
        return
    return_code = run_process(command, log_path)
    if return_code:
        raise RuntimeError(f"Phase submission command exited with code {return_code}")
    valid, detail = validate_question_rows(
        results_path,
        expected_ids,
        require_articles=True,
        allowed_articles=allowed_articles,
    )
    if not valid:
        raise RuntimeError(f"Phase submission validation failed: {detail}")


def preflight(
    *,
    config_path: Path,
    input_path: Path,
    corpus_path: Path,
    bm25_path: Path,
    expected_questions: int,
    expected_articles: int,
) -> tuple[list[dict], dict, set[str]]:
    questions = load_questions(input_path, expected_questions)
    corpus = read_json(corpus_path)
    articles = corpus.get("articles", [])
    chunk_ids = [row.get("chunk_id") for row in articles]
    if len(articles) != expected_articles or corpus.get("total_articles") != expected_articles:
        raise ValueError("Corpus article count/header mismatch")
    if None in chunk_ids or len(set(chunk_ids)) != expected_articles:
        raise ValueError("Corpus chunk IDs are missing or duplicated")
    allowed_articles = {
        " ".join(str(row.get("relevant_article_str") or "").split())
        for row in articles
        if row.get("relevant_article_str")
    }
    if any(not row.get("relevant_article_str") for row in articles):
        raise ValueError("Corpus contains an empty relevant_article_str")

    bm25 = BM25Index.load(bm25_path)
    if len(bm25) != expected_articles or set(bm25._doc_ids) != set(chunk_ids):
        raise ValueError("BM25 IDs do not exactly match the as-of corpus")

    config = load_config(config_path)
    problems = validate_config(config)
    if problems:
        raise ValueError(f"Invalid config: {'; '.join(problems)}")

    qdrant = create_qdrant_client(config)
    collection_info = retry_operation(
        lambda: qdrant.get_collection(config.qdrant.collection)
    )
    vectors = collection_info.config.params.vectors
    if isinstance(vectors, dict):
        raise ValueError("Qdrant collection uses named vectors; expected one unnamed vector")
    actual_size = int(vectors.size)
    actual_distance = str(vectors.distance).split(".")[-1].lower()
    expected_distance = str(config.qdrant.distance).split(".")[-1].lower()
    if actual_size != config.qdrant.vector_size or actual_distance != expected_distance:
        raise ValueError(
            f"Qdrant schema mismatch: size={actual_size}, distance={actual_distance}; "
            f"expected size={config.qdrant.vector_size}, distance={expected_distance}"
        )
    count = retry_operation(
        lambda: qdrant.count(collection_name=config.qdrant.collection, exact=True)
    ).count
    if count != expected_articles:
        raise ValueError(
            f"Qdrant collection {config.qdrant.collection} has {count} points; expected {expected_articles}"
        )
    missing_ids: list[str] = []
    stale_ids: list[str] = []
    expected_point_ids = [str(uuid.UUID(chunk_id)) for chunk_id in chunk_ids]
    article_by_point_id = {
        str(uuid.UUID(row["chunk_id"])): row
        for row in articles
    }
    expected_embedding_model = (
        config.gpu.embed_model if config.backend == "gpu" else config.models.embedder
    )
    for start in range(0, len(expected_point_ids), 256):
        ids = expected_point_ids[start:start + 256]
        points = retry_operation(
            lambda ids=ids: qdrant.retrieve(
                collection_name=config.qdrant.collection,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )
        )
        present = {str(point.id) for point in points}
        missing_ids.extend(point_id for point_id in ids if point_id not in present)
        for point in points:
            point_id = str(point.id)
            payload = point.payload or {}
            article = article_by_point_id[point_id]
            if (
                payload.get("embedding_text_sha256") != embedding_text_sha256(article)
                or payload.get("embedding_backend") != config.backend
                or payload.get("embedding_model") != expected_embedding_model
            ):
                stale_ids.append(point_id)
    if missing_ids:
        raise ValueError(f"Qdrant is missing {len(missing_ids)} expected chunk IDs")
    if stale_ids:
        raise ValueError(
            f"Qdrant has {len(stale_ids)} stale or mismatched embeddings; rebuild/resume collection"
        )

    bge = BgeScorer(config, dry_run=False)
    bge_ok, bge_detail = bge.ping()
    if not bge_ok:
        raise ValueError(f"BGE endpoint preflight failed: {bge_detail}")

    if config.backend == "vertex_ai":
        from legal_rag.backends.vertex import VertexEmbedder, VertexQueryRewriter

        embedder = VertexEmbedder(config, dry_run=False)
        rewriter = VertexQueryRewriter(config, mock=False)
    elif config.backend == "gpu":
        from legal_rag.backends.gpu import make_gpu_components

        embedder, rewriter, _, _ = make_gpu_components(config, mock=False)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")
    vector = embedder.embed_query("kiểm tra kết nối hệ thống pháp luật")
    if len(vector) != config.embedding.dimension:
        raise ValueError(f"Embedding dimension {len(vector)} != {config.embedding.dimension}")
    probe = retry_operation(
        lambda: qdrant.query_points(
            collection_name=config.qdrant.collection,
            query=vector.tolist(),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
    )
    if not probe.points:
        raise ValueError("Qdrant vector search preflight returned no points")
    rewrite_probe = rewriter.rewrite_strict("Người lao động được nghỉ phép năm bao nhiêu ngày?")
    if not rewrite_probe.get("rewritten_query") or not rewrite_probe.get("topic_description"):
        raise ValueError("Query rewrite endpoint returned an incomplete preflight response")
    intent_probe = LegalIntentDecomposer(
        config,
        mock=False,
        chat_complete=getattr(rewriter, "_chat_complete", None),
    ).decompose_strict("Người lao động được nghỉ phép năm bao nhiêu ngày?")
    if not intent_probe.intents:
        raise ValueError("Intent decomposition endpoint returned no intents")

    fingerprint = {
        "input_sha256": sha256_file(input_path),
        "corpus_sha256": sha256_file(corpus_path),
        "bm25_sha256": sha256_file(bm25_path),
        "config_sha256": sha256_file(config_path),
        "questions": expected_questions,
        "articles": expected_articles,
        "qdrant_collection": config.qdrant.collection,
        "runtime_target_sha256": hashlib.sha256(
            json.dumps(
                {
                    "backend": config.backend,
                    "qdrant_url": config.qdrant.url,
                    "qdrant_host": config.qdrant.host,
                    "qdrant_port": config.qdrant.port,
                    "qdrant_collection": config.qdrant.collection,
                    "embedder": config.models.embedder,
                    "reranker": config.models.reranker,
                    "bge_endpoint": config.bge.endpoint_id,
                    "bge_dns": config.bge.dns,
                    "gcp_project": config.vllm.gcp_project,
                    "gcp_location": config.vllm.gcp_location,
                    "gpu_embed_endpoint": config.gpu.embed_endpoint_id,
                    "gpu_embed_dns": config.gpu.embed_dns,
                    "gpu_embed_region": config.gpu.embed_region,
                    "gpu_embed_model": config.gpu.embed_model,
                    "gpu_llm_endpoint": config.gpu.llm_endpoint_id,
                    "gpu_llm_dns": config.gpu.llm_dns,
                    "gpu_llm_region": config.gpu.region,
                    "gpu_llm_model": config.gpu.llm_model,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "code_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "legal_rag/common/backends_common.py",
                "legal_rag/common/article_lookup.py",
                "legal_rag/common/paths.py",
                "legal_rag/common/qdrant.py",
                "legal_rag/backends/bge.py",
                "legal_rag/backends/vertex.py",
                "legal_rag/backends/gpu.py",
                "legal_rag/pipeline.py",
                "legal_rag/retrieval/query_analysis.py",
                "legal_rag/retrieval/global_rrf.py",
                "legal_rag/retrieval/intent_rrf.py",
                "legal_rag/ranking/intent_bge.py",
                "legal_rag/ranking/candidate_union.py",
                "legal_rag/verification/candidate_verifier.py",
                "legal_rag/verification/deterministic_cleanup.py",
                "legal_rag/verification/final_collective.py",
                "legal_rag/verification/enforcement_gate.py",
                "legal_rag/verification/intent_coverage_rescue.py",
                "legal_rag/generation/prompt_builder.py",
                "legal_rag/generation/llm_generator.py",
                "legal_rag/generation/generate_answers.py",
                "legal_rag/output/phase_snapshot.py",
                "legal_rag/orchestration/best_pipeline.py",
                "legal_rag/retrieval/bm25_index.py",
                "legal_rag/retrieval/config.py",
                "legal_rag/retrieval/embedder.py",
                "legal_rag/retrieval/hybrid_search.py",
                "legal_rag/retrieval/intent_decomposer.py",
                "legal_rag/retrieval/query_rewriter.py",
                "legal_rag/output/submission.py",
            )
        },
    }
    return questions, fingerprint, allowed_articles


def prepare_manifest(
    path: Path,
    fingerprint: dict,
    settings: dict,
    *,
    accept_code_change: bool = False,
    accept_runtime_change: bool = False,
    accept_workflow_change: bool = False,
) -> dict:
    if path.exists():
        manifest = read_json(path)
        settings_changed = manifest.get("settings") != settings
        if settings_changed:
            if not accept_workflow_change:
                raise ValueError(
                    "Run manifest workflow settings differ from this command. "
                    "Use --accept-workflow-change only when you intentionally changed later "
                    "pipeline stages and want to reuse validated upstream artifacts."
                )
        fingerprint_changed = manifest.get("fingerprint") != fingerprint
        changed_keys: set[str] = set()
        if fingerprint_changed:
            previous = manifest.get("fingerprint", {})
            changed_keys = {
                key
                for key in set(previous) | set(fingerprint)
                if previous.get(key) != fingerprint.get(key)
            }
            accepted_keys = set()
            if accept_code_change:
                accepted_keys.add("code_sha256")
            if accept_runtime_change:
                accepted_keys.add("runtime_target_sha256")
            unaccepted_keys = changed_keys - accepted_keys
            if unaccepted_keys:
                raise ValueError(
                    "Run manifest fingerprint differs from the current run "
                    f"(changed: {', '.join(sorted(changed_keys))}). "
                    "Use --accept-code-change for an intentional code-only change and/or "
                    "--accept-runtime-change after intentionally redeploying model endpoints. "
                    "Input, corpus, BM25 index, and config cannot be overridden."
                )
        if settings_changed or fingerprint_changed:
            accepted_at = utc_now()
            if settings_changed:
                manifest.setdefault("workflow_change_history", []).append(
                    {
                        "accepted_at": accepted_at,
                        "previous_settings": manifest.get("settings", {}),
                    }
                )
                manifest["settings"] = settings
            if "code_sha256" in changed_keys:
                manifest.setdefault("code_change_history", []).append(
                    {
                        "accepted_at": accepted_at,
                        "previous_code_sha256": previous.get("code_sha256", {}),
                    }
                )
            if "runtime_target_sha256" in changed_keys:
                manifest.setdefault("runtime_change_history", []).append(
                    {
                        "accepted_at": accepted_at,
                        "previous_runtime_target_sha256": previous.get("runtime_target_sha256", ""),
                    }
                )
            if fingerprint_changed:
                manifest["fingerprint"] = fingerprint
            atomic_write_json(path, manifest)
        return manifest
    manifest = {
        "workflow": "best-asof-20260301-v1",
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "settings": settings,
        "stages": {},
    }
    atomic_write_json(path, manifest)
    return manifest


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the proven legal RAG submission workflow end to end.")
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input", default="../R2AIStage1DATA.json")
    parser.add_argument("--corpus", default="corpus/data/corpus_clean_asof_20260301.json")
    parser.add_argument("--bm25-index", default="retrieval/data/bm25_index_asof_20260301.pkl")
    parser.add_argument("--run-dir", default="outputs/runs/best_asof_20260301_v1")
    parser.add_argument("--expected-questions", type=int, default=EXPECTED_QUESTIONS)
    parser.add_argument("--expected-articles", type=int, default=EXPECTED_ARTICLES)
    parser.add_argument("--analysis-workers", type=int, default=12)
    parser.add_argument("--retrieval-workers", type=int, default=20)
    parser.add_argument("--bge-workers", type=int, default=12)
    parser.add_argument("--llm-workers", type=int, default=12)
    parser.add_argument(
        "--rescue-coverage-depth",
        type=int,
        default=4,
        choices=(2, 4),
        help="Raw-intent coverage depth: 4 is best overall; 2 prioritizes article recall",
    )
    parser.add_argument("--max-resume-passes", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--stop-after-stage",
        choices=STAGE_NAMES,
        default="",
        help="Stop cleanly after this stage; rerun with the same run-dir to continue",
    )
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument(
        "--accept-code-change",
        action="store_true",
        help="Resume an existing run after a code-only fingerprint change",
    )
    parser.add_argument(
        "--accept-runtime-change",
        action="store_true",
        help="Resume an existing run after intentionally redeploying model endpoints",
    )
    parser.add_argument(
        "--accept-workflow-change",
        action="store_true",
        help="Reuse validated upstream artifacts after intentionally changing later stages",
    )
    args = parser.parse_args()

    if args.list_stages:
        for stage_name in STAGE_NAMES:
            print(stage_name)
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config_path = (ROOT / args.config).resolve()
    input_path = (ROOT / args.input).resolve()
    corpus_path = (ROOT / args.corpus).resolve()
    bm25_path = (ROOT / args.bm25_index).resolve()
    paths = make_paths(
        (ROOT / args.run_dir).resolve(),
        rescue_coverage_depth=args.rescue_coverage_depth,
    )
    for directory in (paths.root, paths.cache, paths.artifacts, paths.submissions, paths.logs):
        directory.mkdir(parents=True, exist_ok=True)
    acquire_run_lock(paths.root)

    questions, fingerprint, allowed_articles = preflight(
        config_path=config_path,
        input_path=input_path,
        corpus_path=corpus_path,
        bm25_path=bm25_path,
        expected_questions=args.expected_questions,
        expected_articles=args.expected_articles,
    )
    expected_ids = {str(row["id"]) for row in questions}
    settings = {
        "global_source_top_k": GLOBAL_SOURCE_TOP_K,
        "global_top_k": GLOBAL_TOP_K,
        "intent_source_top_k": INTENT_SOURCE_TOP_K,
        "intent_rrf_top_k": INTENT_RRF_TOP_K,
        "tier_rrf_top_k": TIER_RRF_TOP_K,
        "tier_bge_per_intent": TIER_BGE_PER_INTENT,
        "tier_raw_intent_per_intent": TIER_RAW_INTENT_PER_INTENT,
        "stage1_batch": 6,
        "stage1_content_chars": 1800,
        "stage1_prompt_version": "gemma-recall-v4",
        "final_content_chars": 2200,
        "final_batch": 6,
        "final_direct_max": 8,
        "final_min_size": 2,
        "final_preserve_top1": False,
        "final_compact_candidates": True,
        "final_prompt_version": "gemma-precision-v5",
        "rescue_coverage_depth": args.rescue_coverage_depth,
        "generation_prompt_version": PROMPT_VERSION,
        "generation_max_articles": 0,
        "generation_content_chars": 2400,
        "generation_total_content_chars": 48000,
    }
    manifest = prepare_manifest(
        paths.manifest,
        fingerprint,
        settings,
        accept_code_change=args.accept_code_change,
        accept_runtime_change=args.accept_runtime_change,
        accept_workflow_change=args.accept_workflow_change,
    )
    if args.preflight_only:
        log.info("Preflight complete; no pipeline stages were run")
        return

    py = sys.executable
    common = ["--config", str(config_path), "--input", str(input_path)]

    def stop_if_requested(stage_name: str, submission: Path | None = None) -> bool:
        if args.stop_after_stage != stage_name:
            return False
        manifest["status"] = "paused"
        manifest["completed_through"] = stage_name
        manifest["paused_at"] = utc_now()
        if submission is not None:
            manifest["phase_submission"] = str(submission)
        atomic_write_json(paths.manifest, manifest)
        if submission is None:
            log.info("Stopped after %s (this phase has no article submission)", stage_name)
        else:
            log.info("Stopped after %s; phase submission: %s", stage_name, submission)
        return True

    run_resumable_stage(
        name="01_query_analysis",
        base_command=[
            py,
            "-m",
            "legal_rag.retrieval.query_analysis",
            *common,
            "--rewrite-cache",
            str(paths.cache / "rewrite.jsonl"),
            "--intent-cache",
            str(paths.cache / "intents.jsonl"),
            "--output",
            str(paths.analysis),
            "--workers",
            str(args.analysis_workers),
        ],
        validator=lambda: validate_question_rows(
            paths.analysis,
            expected_ids,
            required_fields=("rewritten_query", "topic_description", "legal_intents"),
        ),
        log_path=paths.logs / "01_query_analysis.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("01_query_analysis"):
        return

    rrf_prefix = paths.submissions / "rrf_top"
    run_resumable_stage(
        name="02_global_rrf",
        base_command=[
            py,
            "-m",
            "legal_rag.retrieval.global_rrf",
            *common,
            "--cached-analysis",
            str(paths.analysis),
            "--output",
            str(paths.global_rrf),
            "--submission-prefix",
            str(rrf_prefix),
            "--bm25-index",
            str(bm25_path),
            "--workers",
            str(args.retrieval_workers),
            "--top-k",
            str(GLOBAL_TOP_K),
            "--source-top-k",
            str(GLOBAL_SOURCE_TOP_K),
            "--export-ks",
            str(GLOBAL_TOP_K),
            "--expected-count",
            str(args.expected_articles),
            "--strict-errors",
        ],
        validator=lambda: validate_question_rows(
            paths.rrf60_results,
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "02_global_rrf.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("02_global_rrf", paths.submissions / "rrf_top60_clean" / "submission.zip"):
        return

    run_resumable_stage(
        name="03_raw_intent_retrieval",
        base_command=[
            py,
            "-m",
            "legal_rag.retrieval.intent_rrf",
            *common,
            "--intents-source",
            str(paths.analysis),
            "--cache",
            str(paths.cache / "intent_ranked_hits.jsonl"),
            "--output",
            str(paths.intent_ranked),
            "--bm25-index",
            str(bm25_path),
            "--expected-count",
            str(args.expected_articles),
            "--workers",
            str(args.retrieval_workers),
            "--strict-errors",
        ],
        validator=lambda: validate_intent_rows(paths.intent_ranked, expected_ids, allowed_articles),
        log_path=paths.logs / "03_raw_intent_retrieval.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )

    raw_intent_dir = paths.submissions / "raw_intent_top5_union"
    ensure_phase_submission(
        command=[
            py,
            "-m",
            "legal_rag.output.phase_snapshot",
            "--mode",
            "raw-intent",
            "--input",
            str(input_path),
            "--intent-results",
            str(paths.intent_ranked),
            "--top-each",
            str(TIER_RAW_INTENT_PER_INTENT),
            "--output-dir",
            str(raw_intent_dir),
        ],
        results_path=raw_intent_dir / "results.json",
        expected_ids=expected_ids,
        allowed_articles=allowed_articles,
        log_path=paths.logs / "03_raw_intent_submission.log",
    )
    if stop_if_requested("03_raw_intent_retrieval", raw_intent_dir / "submission.zip"):
        return

    run_resumable_stage(
        name="04_bge_intent_rerank",
        base_command=[
            py,
            "-m",
            "legal_rag.ranking.intent_bge",
            "--mode",
            "run",
            *common,
            "--corpus",
            str(corpus_path),
            "--cache",
            str(paths.bge_cache),
            "--rrf60-submission",
            str(paths.rrf60_results),
            "--intent-results",
            str(paths.intent_ranked),
            "--workers",
            str(args.bge_workers),
            "--strict-errors",
        ],
        validator=lambda: validate_bge_cache(paths.bge_cache, paths.intent_ranked, allowed_articles),
        log_path=paths.logs / "04_bge_intent_rerank.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )

    bge_intent_dir = paths.submissions / "bge_intent_top5_union"
    ensure_phase_submission(
        command=[
            py,
            "-m",
            "legal_rag.output.phase_snapshot",
            "--mode",
            "bge-intent",
            "--input",
            str(input_path),
            "--intent-results",
            str(paths.intent_ranked),
            "--bge-cache",
            str(paths.bge_cache),
            "--top-each",
            str(TIER_BGE_PER_INTENT),
            "--output-dir",
            str(bge_intent_dir),
        ],
        results_path=bge_intent_dir / "results.json",
        expected_ids=expected_ids,
        allowed_articles=allowed_articles,
        log_path=paths.logs / "04_bge_intent_submission.log",
    )
    if stop_if_requested("04_bge_intent_rerank", bge_intent_dir / "submission.zip"):
        return

    tiered_dir = paths.submissions / "tiered_rrf12_bge5_rawintent5"
    run_deterministic_stage(
        name="05_tiered_union",
        command=[
            py,
            "-m",
            "legal_rag.ranking.candidate_union",
            "--input",
            str(input_path),
            "--rrf60-submission",
            str(paths.rrf60_results),
            "--intent-results",
            str(paths.intent_ranked),
            "--intent-ranked-results",
            str(paths.intent_ranked),
            "--bge-cache",
            str(paths.bge_cache),
            "--output-dir",
            str(tiered_dir),
            "--diagnostics",
            str(paths.artifacts / "tiered_union.csv"),
            "--top-b-rrf",
            str(TIER_RRF_TOP_K),
            "--top-n-bge",
            str(TIER_BGE_PER_INTENT),
            "--top-m-intent",
            str(TIER_RAW_INTENT_PER_INTENT),
            "--strict-errors",
        ],
        validator=lambda: validate_question_rows(
            paths.tiered_results,
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "05_tiered_union.log",
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("05_tiered_union", tiered_dir / "submission.zip"):
        return

    stage1_dir = paths.submissions / "stage1_gemma_compact_v4"
    run_resumable_stage(
        name="06_stage1_compact",
        base_command=[
            py,
            "-m",
            "legal_rag.verification.candidate_verifier",
            "--config",
            str(config_path),
            "--input",
            str(paths.tiered_results),
            "--corpus",
            str(corpus_path),
            "--intent-results",
            str(paths.intent_ranked),
            "--cache",
            str(paths.cache / "stage1_gemma_compact_v4.jsonl"),
            "--output-dir",
            str(stage1_dir),
            "--diagnostics",
            str(paths.stage1_diagnostics),
            "--workers",
            str(args.llm_workers),
            "--stage1-compact-candidates",
            "--strict-errors",
        ],
        validator=lambda: validate_question_rows(
            stage1_dir / "results.json",
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "06_stage1_compact.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("06_stage1_compact", stage1_dir / "submission.zip"):
        return

    penalty_dir = paths.submissions / "stage1_gemma_v4_penalty_cleanup"
    run_deterministic_stage(
        name="07_penalty_cleanup",
        command=[
            py,
            "-m",
            "legal_rag.verification.deterministic_cleanup",
            "--input-diagnostics",
            str(paths.stage1_diagnostics),
            "--corpus",
            str(corpus_path),
            "--output-dir",
            str(penalty_dir),
            "--skip-invalid",
            "--strict-errors",
        ],
        validator=lambda: validate_question_rows(
            penalty_dir / "results.json",
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "07_penalty_cleanup.log",
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("07_penalty_cleanup", penalty_dir / "submission.zip"):
        return

    final_dir = paths.submissions / "final_collective"
    run_resumable_stage(
        name="08_final_collective",
        base_command=[
            py,
            "-m",
            "legal_rag.verification.final_collective",
            "--config",
            str(config_path),
            "--input-diagnostics",
            str(paths.penalty_diagnostics),
            "--intent-results",
            str(paths.intent_ranked),
            "--corpus",
            str(corpus_path),
            "--cache",
            str(paths.cache / "final_collective_gemma_v5.jsonl"),
            "--output-dir",
            str(final_dir),
            "--diagnostics",
            str(paths.final_diagnostics),
            "--content-max-chars",
            "2200",
            "--batch-size",
            "6",
            "--direct-max",
            "8",
            "--min-size",
            "2",
            "--workers",
            str(args.llm_workers),
            "--compact-candidates",
            "--strict-errors",
        ],
        validator=lambda: validate_question_rows(
            paths.final_results,
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "08_final_collective.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("08_final_collective", final_dir / "submission.zip"):
        return

    gate_dir = paths.submissions / "best_final_enforcement_gate"
    run_deterministic_stage(
        name="09_enforcement_role_gate",
        command=[
            py,
            "-m",
            "legal_rag.verification.enforcement_gate",
            "--input",
            str(paths.final_results),
            "--corpus",
            str(corpus_path),
            "--output-dir",
            str(gate_dir),
            "--strict-errors",
        ],
        validator=lambda: validate_question_rows(
            gate_dir / "results.json",
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "09_enforcement_role_gate.log",
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("09_enforcement_role_gate", gate_dir / "submission.zip"):
        return

    rescue_dir = paths.rescue_results.parent
    run_deterministic_stage(
        name="10_intent_coverage_rescue",
        command=[
            py,
            "-m",
            "legal_rag.verification.intent_coverage_rescue",
            "--final-results",
            str(gate_dir / "results.json"),
            "--stage1-results",
            str(paths.submissions / "stage1_gemma_v4_penalty_cleanup" / "results.json"),
            "--intent-results",
            str(paths.intent_ranked),
            "--coverage-depth",
            str(args.rescue_coverage_depth),
            "--output-dir",
            str(rescue_dir),
        ],
        validator=lambda: validate_question_rows(
            paths.rescue_results,
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "10_intent_coverage_rescue.log",
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("10_intent_coverage_rescue", rescue_dir / "submission.zip"):
        return

    generation_dir = paths.generated_results.parent
    run_resumable_stage(
        name="11_answer_generation",
        base_command=[
            py,
            "-m",
            "legal_rag.generation.generate_answers",
            "--config",
            str(config_path),
            "--input",
            str(paths.rescue_results),
            "--corpus",
            str(corpus_path),
            "--cache",
            str(paths.cache / f"answer_generation_rescue_depth{args.rescue_coverage_depth}.jsonl"),
            "--output-dir",
            str(generation_dir),
            "--errors",
            str(paths.artifacts / "answer_generation_errors.json"),
            "--workers",
            str(args.llm_workers),
            "--strict-errors",
        ],
        validator=lambda: validate_generation_stage(
            paths.generated_results,
            generation_dir / "generation_metadata.json",
            expected_ids,
            allowed_articles,
        ),
        log_path=paths.logs / "11_answer_generation.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )
    if stop_if_requested("11_answer_generation", generation_dir / "submission.zip"):
        return

    final_zip = generation_dir / "submission.zip"
    zip_ok, zip_detail = validate_generated_zip(final_zip, expected_ids, allowed_articles)
    if not zip_ok:
        raise RuntimeError(zip_detail)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["final_submission"] = str(final_zip)
    atomic_write_json(paths.manifest, manifest)
    log.info("Pipeline complete: %s", final_zip)


if __name__ == "__main__":
    main()
