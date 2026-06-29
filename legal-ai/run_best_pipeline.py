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

from bge_scorer import BgeScorer
from retrieval.bm25_index import BM25Index
from retrieval.config import load_config, validate_config
from retrieval.embedder import embedding_text_sha256
from retrieval.intent_decomposer import LegalIntentDecomposer


ROOT = Path(__file__).resolve().parent
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
    stage2_diagnostics: Path
    final_diagnostics: Path
    final_results: Path


def make_paths(root: Path) -> RunPaths:
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
        stage1_diagnostics=artifacts / "stage1_compact_diagnostics.json",
        penalty_diagnostics=submissions / "stage1_penalty_cleanup" / "diagnostics.json",
        stage2_diagnostics=artifacts / "stage2_compact_diagnostics.json",
        final_diagnostics=artifacts / "final_collective_diagnostics.json",
        final_results=submissions / "final_collective" / "results.json",
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

    from qdrant_client import QdrantClient

    if config.qdrant.url:
        qdrant = QdrantClient(
            url=config.qdrant.url,
            api_key=config.qdrant.api_key or None,
            timeout=120,
        )
    else:
        qdrant = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
            timeout=120,
        )
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
        from vertex_backends import VertexEmbedder, VertexQueryRewriter

        embedder = VertexEmbedder(config, dry_run=False)
        rewriter = VertexQueryRewriter(config, mock=False)
    elif config.backend == "gpu":
        from gpu_backends import make_gpu_components

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
                "backends_common.py",
                "bge_scorer.py",
                "query_analysis_runner.py",
                "pipeline.py",
                "rrf_topk_clean_probe.py",
                "intent_ranked_hits_clean.py",
                "bge_intent_compression_clean.py",
                "bge_intent_tiered_union_clean.py",
                "llm_candidate_verifier.py",
                "stage1_deterministic_cleanup.py",
                "llm_best_collective_filter.py",
                "enforcement_role_postprocess.py",
                "run_best_pipeline.py",
                "vertex_backends.py",
                "gpu_backends.py",
                "retrieval/bm25_index.py",
                "retrieval/embedder.py",
                "retrieval/intent_decomposer.py",
                "retrieval/query_rewriter.py",
            )
        },
    }
    return questions, fingerprint, allowed_articles


def prepare_manifest(path: Path, fingerprint: dict, settings: dict) -> dict:
    if path.exists():
        manifest = read_json(path)
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Run manifest fingerprint differs from current input/corpus/index/config")
        if manifest.get("settings") != settings:
            raise ValueError("Run manifest workflow settings differ from this command")
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
    parser.add_argument("--max-resume-passes", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config_path = (ROOT / args.config).resolve()
    input_path = (ROOT / args.input).resolve()
    corpus_path = (ROOT / args.corpus).resolve()
    bm25_path = (ROOT / args.bm25_index).resolve()
    paths = make_paths((ROOT / args.run_dir).resolve())
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
        "stage2_content_chars": 1600,
        "final_content_chars": 2200,
        "final_batch": 6,
        "final_direct_max": 8,
        "final_preserve_top1": False,
    }
    manifest = prepare_manifest(paths.manifest, fingerprint, settings)
    if args.preflight_only:
        log.info("Preflight complete; no pipeline stages were run")
        return

    py = sys.executable
    common = ["--config", str(config_path), "--input", str(input_path)]

    run_resumable_stage(
        name="01_query_analysis",
        base_command=[
            py,
            str(ROOT / "query_analysis_runner.py"),
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

    rrf_prefix = paths.submissions / "rrf_top"
    run_resumable_stage(
        name="02_global_rrf",
        base_command=[
            py,
            str(ROOT / "rrf_topk_clean_probe.py"),
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

    run_resumable_stage(
        name="03_raw_intent_retrieval",
        base_command=[
            py,
            str(ROOT / "intent_ranked_hits_clean.py"),
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

    run_resumable_stage(
        name="04_bge_intent_rerank",
        base_command=[
            py,
            str(ROOT / "bge_intent_compression_clean.py"),
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

    tiered_dir = paths.submissions / "tiered_rrf12_bge5_rawintent5"
    run_deterministic_stage(
        name="05_tiered_union",
        command=[
            py,
            str(ROOT / "bge_intent_tiered_union_clean.py"),
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

    stage1_dir = paths.submissions / "stage1_compact"
    run_resumable_stage(
        name="06_stage1_compact",
        base_command=[
            py,
            str(ROOT / "llm_candidate_verifier.py"),
            "--config",
            str(config_path),
            "--input",
            str(paths.tiered_results),
            "--corpus",
            str(corpus_path),
            "--intent-results",
            str(paths.intent_ranked),
            "--cache",
            str(paths.cache / "stage1_compact.jsonl"),
            "--output-dir",
            str(stage1_dir),
            "--diagnostics",
            str(paths.stage1_diagnostics),
            "--workers",
            str(args.llm_workers),
            "--stage1-compact-candidates",
            "--stage1-only",
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

    penalty_dir = paths.submissions / "stage1_penalty_cleanup"
    run_deterministic_stage(
        name="07_penalty_cleanup",
        command=[
            py,
            str(ROOT / "stage1_deterministic_cleanup.py"),
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

    stage2_dir = paths.submissions / "stage2_compact"
    run_resumable_stage(
        name="08_stage2_compact",
        base_command=[
            py,
            str(ROOT / "llm_candidate_verifier.py"),
            "--config",
            str(config_path),
            "--stage2-only-from-diagnostics",
            str(paths.penalty_diagnostics),
            "--corpus",
            str(corpus_path),
            "--intent-results",
            str(paths.intent_ranked),
            "--cache",
            str(paths.cache / "stage2_compact.jsonl"),
            "--output-dir",
            str(stage2_dir),
            "--diagnostics",
            str(paths.stage2_diagnostics),
            "--workers",
            str(args.llm_workers),
            "--stage2-compact-candidates",
            "--strict-errors",
        ],
        validator=lambda: validate_question_rows(
            stage2_dir / "results.json",
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "08_stage2_compact.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )

    final_dir = paths.submissions / "final_collective"
    run_resumable_stage(
        name="09_final_collective",
        base_command=[
            py,
            str(ROOT / "llm_best_collective_filter.py"),
            "--config",
            str(config_path),
            "--input-diagnostics",
            str(paths.stage2_diagnostics),
            "--intent-results",
            str(paths.intent_ranked),
            "--corpus",
            str(corpus_path),
            "--cache",
            str(paths.cache / "final_collective.jsonl"),
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
            "3",
            "--workers",
            str(args.llm_workers),
            "--prompt-mode",
            "final_precision",
            "--strict-errors",
        ],
        validator=lambda: validate_question_rows(
            paths.final_results,
            expected_ids,
            require_articles=True,
            allowed_articles=allowed_articles,
        ),
        log_path=paths.logs / "09_final_collective.log",
        max_passes=args.max_resume_passes,
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )

    gate_dir = paths.submissions / "best_final_enforcement_gate"
    run_deterministic_stage(
        name="10_enforcement_role_gate",
        command=[
            py,
            str(ROOT / "enforcement_role_postprocess.py"),
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
        log_path=paths.logs / "10_enforcement_role_gate.log",
        manifest=manifest,
        manifest_path=paths.manifest,
        errors_path=paths.errors,
    )

    final_zip = gate_dir / "submission.zip"
    zip_ok, zip_detail = validate_submission_zip(final_zip, expected_ids, allowed_articles)
    if not zip_ok:
        raise RuntimeError(zip_detail)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["final_submission"] = str(final_zip)
    atomic_write_json(paths.manifest, manifest)
    log.info("Pipeline complete: %s", final_zip)


if __name__ == "__main__":
    main()
