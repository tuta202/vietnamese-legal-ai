from __future__ import annotations

import argparse
import atexit
import os
import json
import logging
import re
import shutil
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.common.article_lookup import ArticleLookup  # noqa: E402
from legal_rag.common.backends_common import retry_transient, strip_code_fences  # noqa: E402
from legal_rag.retrieval.config import load_config, validate_config  # noqa: E402
from legal_rag.output.submission import save_submission  # noqa: E402


log = logging.getLogger("legal_rag.verification.candidate_verifier")

BATCH_SIZE = 6
STAGE1_CONTENT_MAX_CHARS = 1800
STAGE2_CONTENT_MAX_CHARS = 1600
STAGE2_BATCH_SIZE = 8
STAGE2_DIRECT_MAX_ARTICLES = 10
STAGE2_MIN_ARTICLES = 2
STAGE2_EMPTY_SELECTION_RESCUE_TOP = 2
STAGE2_SINGLE_SELECTION_RESCUE_TOP = 1

STAGE1_SYSTEM_PROMPT = """\
You select Vietnamese legal articles for retrieval recall.

Use only:
- question
- legal_intents
- candidate_articles

Keep an article if it may be needed to answer the question or any legal intent:
- direct rule, definition, condition, exception, procedure, deadline, authority
- right, obligation, penalty, legal consequence, required dossier/evidence
- one useful article for each independent legal intent
- both general law and detailed decree/circular if they add different details

Remove only articles that are clearly unrelated, only background, wrong subtask, or duplicate another kept article without adding legal detail.

If unsure, keep the article.
Return exactly one valid JSON object. No markdown. No bullets. No explanation:
{
  "selected_article_keys": [],
  "confidence": "high|medium|low"
}
"""

STAGE2_SYSTEM_PROMPT = """\
You are a recall-safe Vietnamese legal evidence selector.

Goal: improve precision, but recall is more important.

Your task is to keep the legal articles that may be needed to answer the question.

Use only:
- the question
- the legal retrieval intents, if provided
- the content of each candidate article

Do not rely on outside knowledge.
Do not select article IDs outside candidate_articles.

Rules:
1. Keep every article that may be needed to answer any legal intent.
2. Keep articles for all parts of the question: conditions, procedures, deadlines, penalties, remedies, tax, accounting, documents, authority, exceptions, or definitions.
3. Keep both a general law article and a detailed decree/circular article if they add different legal details.
4. Remove only articles that are clearly unrelated, only background, clearly outdated, or duplicate another kept article without adding useful details.
5. If unsure, keep the article.
6. Select only article_id values from candidate_articles.

Return valid JSON only, with no explanation:
{
  "selected_article_ids": [],
  "confidence": "high|medium|low"
}
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_text(value: str) -> str:
    return " ".join((value or "").split())


def doc_from_article_ref(article_ref: str) -> str:
    doc, sep, _dieu = article_ref.rpartition("|")
    return doc if sep else article_ref


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def order_by_reference(items: list[str], reference: list[str]) -> list[str]:
    rank = {item: index for index, item in enumerate(reference)}
    return sorted(
        dedupe_keep_order(items),
        key=lambda item: (rank.get(item, len(rank)), item),
    )


def build_candidate_articles(
    article_refs: list[str],
    lookup: ArticleLookup,
    *,
    content_max_chars: int,
) -> tuple[list[dict], list[str]]:
    candidates: list[dict] = []
    missing: list[str] = []
    for article_ref in article_refs:
        candidate = lookup.verifier_candidate(article_ref, content_max_chars=content_max_chars)
        if candidate is None:
            missing.append(article_ref)
            continue
        candidates.append(candidate)
    return candidates, missing


def chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_user_prompt(
    question: str,
    candidate_articles: list[dict],
    *,
    legal_intents: list[str] | None = None,
) -> str:
    payload = {
        "question": question,
        "candidate_articles": candidate_articles,
    }
    if legal_intents is not None:
        payload["legal_intents"] = legal_intents
    return json.dumps(payload, ensure_ascii=False, indent=2)


def alias_candidates(
    candidates: list[dict],
    *,
    compact: bool = False,
) -> tuple[list[dict], dict[str, str]]:
    key_to_id: dict[str, str] = {}
    aliased: list[dict] = []
    for index, candidate in enumerate(candidates, start=1):
        key = f"A{index}"
        key_to_id[key] = candidate["article_id"]
        if compact:
            law_id = candidate["article_id"].split("|", 1)[0]
            law_name = candidate.get("law_name", "")
            article_number = candidate.get("article_number", "")
            article_title = candidate.get("article_title", "")
            source = f"{law_id} - {law_name}" if law_name else law_id
            article = f"{article_number}. {article_title}".strip(" .")
            aliased.append(
                {
                    "key": key,
                    "source": source,
                    "article": article,
                    "content": candidate.get("article_content", ""),
                }
            )
            continue
        display = dict(candidate)
        display["article_key"] = key
        display.pop("article_id", None)
        aliased.append(display)
    return aliased, key_to_id


def parse_verifier_json(raw: str, allowed_ids: set[str]) -> tuple[dict | None, bool]:
    cleaned = strip_code_fences(raw)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None, False
    if not isinstance(data, dict):
        return None, False
    if "selected_article_ids" not in data:
        return None, False
    selected = data["selected_article_ids"]
    if not isinstance(selected, list):
        return None, False
    normalized_ids = [str(x).strip() for x in selected]
    if any(article_id not in allowed_ids for article_id in normalized_ids):
        return None, False
    selected = normalized_ids
    confidence = str(data.get("confidence", "low")).lower().strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {"selected_article_ids": dedupe_keep_order(selected), "confidence": confidence}, True


def parse_stage1_alias_json(raw: str, key_to_id: dict[str, str]) -> tuple[dict | None, bool]:
    cleaned = strip_code_fences(raw)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        marker = re.search(
            r"selected_article_keys\**\s*:?\s*(?P<keys>(?:.|\n){0,300})",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not marker:
            return None, False
        keys = re.findall(r"\bA\d+\b", marker.group("keys"))
        if not keys:
            return None, False
        data = {"selected_article_keys": keys, "confidence": "low"}
    if not isinstance(data, dict):
        return None, False
    if "selected_article_keys" in data:
        keys = data["selected_article_keys"]
    elif "selected_article_ids" in data:
        keys = data["selected_article_ids"]
    else:
        return None, False
    if not isinstance(keys, list):
        return None, False
    normalized_keys = [str(key).strip() for key in keys]
    if any(key not in key_to_id for key in normalized_keys):
        return None, False
    selected: list[str] = []
    for key in normalized_keys:
        selected.append(key_to_id[key])
    confidence = str(data.get("confidence", "low")).lower().strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "selected_article_keys": normalized_keys,
        "selected_article_ids": dedupe_keep_order(selected),
        "confidence": confidence,
    }, True


def apply_fallback(
    verifier_result: dict | None,
    original_order: list[str],
    *,
    technical_issue: bool = False,
) -> tuple[list[str], bool, str]:
    if verifier_result is None:
        return original_order[:5], True, "parse_fail"

    selected = dedupe_keep_order(verifier_result.get("selected_article_ids", []))

    if not selected:
        return original_order[:3], True, "empty_selection_rescue_top3"
    if technical_issue:
        return dedupe_keep_order(selected + original_order[:5]), True, "technical_issue_rescue_top5"
    if len(selected) == 1:
        return dedupe_keep_order(selected + original_order[:1]), True, "small_selection_rescue_top1"
    return selected, False, "trust_llm_selected"


class VerifierWorker:
    def __init__(self, config) -> None:
        self.config = config
        self.backend = config.backend
        self.client = None
        if self.backend == "vertex_ai":
            from legal_rag.backends.vertex import make_genai_client  # noqa: PLC0415

            self.client = make_genai_client(config)
        elif self.backend != "gpu":
            raise ValueError(f"Unsupported verifier backend: {self.backend}")

    def call(
        self,
        question: str,
        candidate_articles: list[dict],
        *,
        system_prompt: str,
        legal_intents: list[str] | None = None,
    ) -> str:
        user_prompt = build_user_prompt(question, candidate_articles, legal_intents=legal_intents)
        if self.backend == "gpu":
            from legal_rag.backends.gpu import gpu_chat_complete  # noqa: PLC0415

            return gpu_chat_complete(
                self.config,
                system=system_prompt,
                user=user_prompt,
                temperature=0.0,
                max_tokens=2048,
            )

        from google.genai import types  # noqa: PLC0415

        assert self.client is not None
        return retry_transient(
            lambda: getattr(
                self.client.models.generate_content(
                    model=self.config.vllm.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                        max_output_tokens=2048,
                    ),
                ),
                "text",
                "",
            )
            or "",
            attempts=6,
            base=8.0,
        )


_THREAD_LOCAL = threading.local()


def get_worker(config) -> VerifierWorker:
    worker = getattr(_THREAD_LOCAL, "worker", None)
    worker_key = (
        config.backend,
        config.vllm.model if config.backend == "vertex_ai" else config.gpu.llm_model,
        config.gpu.llm_endpoint_id if config.backend == "gpu" else config.vllm.gcp_project,
    )
    if worker is None or getattr(_THREAD_LOCAL, "worker_key", None) != worker_key:
        worker = VerifierWorker(config)
        _THREAD_LOCAL.worker = worker
        _THREAD_LOCAL.worker_key = worker_key
    return worker


def load_cache(path: Path) -> dict[object, dict]:
    if not path.exists():
        return {}
    rows: dict[object, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") is not None and row.get("final_article_ids") is not None:
                rows[row["id"]] = row
    return rows


def append_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def acquire_cache_run_lock(cache_path: Path) -> Path:
    lock_path = cache_path.with_suffix(cache_path.suffix + ".running")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            text = lock_path.read_text(encoding="utf-8")
            pid_line = next((line for line in text.splitlines() if line.startswith("pid=")), "")
            pid = int(pid_line.partition("=")[2])
            os.kill(pid, 0)
        except (OSError, ValueError, StopIteration):
            lock_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                f"Cache lock already exists and PID {pid} is active: {lock_path}"
            )
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Cache lock already exists: {lock_path}. "
            "Another verifier run may still be writing this cache. "
            "If no run is active, delete the .running file and retry."
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\nstarted={datetime.now().isoformat(timespec='seconds')}\n")

    def cleanup() -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(cleanup)
    return lock_path


def prepare_cache_for_run(cache_path: Path, resume: bool) -> None:
    if resume or not cache_path.exists() or cache_path.stat().st_size == 0:
        return
    backup = cache_path.with_suffix(
        cache_path.suffix + ".bak_fresh_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    shutil.move(str(cache_path), str(backup))
    log.warning("Fresh run requested; moved existing cache to %s", backup)


def run_stage2_call(
    config,
    question: str,
    candidate_articles: list[dict],
    legal_intents: list[str],
    *,
    compact_candidates: bool = False,
) -> dict:
    allowed_ids = {c["article_id"] for c in candidate_articles}
    key_to_id: dict[str, str] = {}
    prompt_candidates = candidate_articles
    if compact_candidates:
        prompt_candidates, key_to_id = alias_candidates(candidate_articles, compact=True)
    raw = ""
    verifier_result = None
    parse_ok = False
    error = ""
    try:
        raw = get_worker(config).call(
            question,
            prompt_candidates,
            system_prompt=STAGE2_SYSTEM_PROMPT,
            legal_intents=legal_intents,
        )
        if compact_candidates:
            verifier_result, parse_ok = parse_stage1_alias_json(raw, key_to_id)
        else:
            verifier_result, parse_ok = parse_verifier_json(raw, allowed_ids)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "candidate_article_ids": [c["article_id"] for c in candidate_articles],
        "candidate_article_keys": key_to_id,
        "selected_article_keys": verifier_result.get("selected_article_keys", []) if verifier_result else [],
        "selected_article_ids": verifier_result.get("selected_article_ids", []) if verifier_result else [],
        "confidence": verifier_result.get("confidence", "low") if verifier_result else "low",
        "parse_ok": parse_ok,
        "raw_response": raw[:2000],
        "error": error,
    }


def merge_confidence(confidence_values: list[str], *, has_error: bool) -> str:
    if has_error:
        return "low"
    if confidence_values and all(c == "low" for c in confidence_values):
        return "low"
    if "medium" in confidence_values or "low" in confidence_values:
        return "medium"
    if confidence_values:
        return "high"
    return "low"


def stage2_minimal_select(
    config,
    lookup: ArticleLookup,
    *,
    question: str,
    stage1_article_ids: list[str],
    legal_intents: list[str],
    evidence_order: list[str] | None = None,
    compact_candidates: bool = False,
    strict_errors: bool = False,
) -> tuple[list[str], dict]:
    evidence_order = dedupe_keep_order(evidence_order or stage1_article_ids)
    if len(stage1_article_ids) < STAGE2_MIN_ARTICLES:
        return stage1_article_ids, {
            "enabled": True,
            "skipped": True,
            "reason": f"stage1_size_lt_{STAGE2_MIN_ARTICLES}",
            "selected_article_ids": stage1_article_ids,
            "confidence": "high",
            "parse_ok": True,
            "rounds": [],
            "fallback_used": False,
            "fallback_reason": "not_needed",
        }

    if len(stage1_article_ids) <= 1:
        return stage1_article_ids, {
            "enabled": True,
            "skipped": True,
            "reason": "stage1_size_le1",
            "selected_article_ids": stage1_article_ids,
            "confidence": "high",
            "parse_ok": True,
            "rounds": [],
            "fallback_used": False,
            "fallback_reason": "not_needed",
        }

    stage2_candidates, missing = build_candidate_articles(
        stage1_article_ids,
        lookup,
        content_max_chars=STAGE2_CONTENT_MAX_CHARS,
    )
    if strict_errors and missing:
        raise RuntimeError(f"stage2 missing {len(missing)} hydrated articles")
    hydrated_order = [c["article_id"] for c in stage2_candidates]
    if not stage2_candidates:
        return stage1_article_ids, {
            "enabled": True,
            "skipped": False,
            "reason": "no_hydrated_stage2_candidates",
            "selected_article_ids": stage1_article_ids,
            "confidence": "low",
            "parse_ok": False,
            "rounds": [],
            "missing_article_ids": missing,
            "fallback_used": True,
            "fallback_reason": "stage2_no_hydrated_keep_stage1",
        }

    rounds: list[dict] = []
    if len(stage2_candidates) <= STAGE2_DIRECT_MAX_ARTICLES:
        result = run_stage2_call(
            config,
            question,
            stage2_candidates,
            legal_intents,
            compact_candidates=compact_candidates,
        )
        result["round"] = "direct"
        result["batch_index"] = 0
        rounds.append(result)
        selected = result["selected_article_ids"] if result["parse_ok"] else []
        has_error = bool(result.get("error")) or not result["parse_ok"]
        confidence = merge_confidence([result.get("confidence", "low")], has_error=has_error)
    else:
        intermediate: list[str] = []
        confidence_values: list[str] = []
        has_error = False
        for batch_index, batch in enumerate(chunks(stage2_candidates, STAGE2_BATCH_SIZE)):
            result = run_stage2_call(
                config,
                question,
                batch,
                legal_intents,
                compact_candidates=compact_candidates,
            )
            result["round"] = "group"
            result["batch_index"] = batch_index
            rounds.append(result)
            if result["parse_ok"]:
                intermediate.extend(result["selected_article_ids"])
                confidence_values.append(result.get("confidence", "low"))
            else:
                has_error = True
        intermediate = dedupe_keep_order(intermediate)

        if intermediate:
            final_candidates, final_missing = build_candidate_articles(
                intermediate,
                lookup,
                content_max_chars=STAGE2_CONTENT_MAX_CHARS,
            )
            if final_missing:
                missing.extend(final_missing)
                if strict_errors:
                    raise RuntimeError(
                        f"stage2 global round missing {len(final_missing)} hydrated articles"
                    )
            result = run_stage2_call(
                config,
                question,
                final_candidates,
                legal_intents,
                compact_candidates=compact_candidates,
            )
            result["round"] = "global"
            result["batch_index"] = 0
            rounds.append(result)
            if result["parse_ok"]:
                selected = result["selected_article_ids"]
                confidence_values.append(result.get("confidence", "low"))
            else:
                selected = []
                has_error = True
        else:
            selected = []
            has_error = True
        confidence = merge_confidence(confidence_values, has_error=has_error)

    if strict_errors and has_error:
        raise RuntimeError("stage2 technical issue")

    selected = dedupe_keep_order(selected)
    if not selected:
        rescued = order_by_reference(
            evidence_order[:STAGE2_EMPTY_SELECTION_RESCUE_TOP],
            evidence_order,
        )
        return rescued, {
            "enabled": True,
            "skipped": False,
            "reason": "stage2_empty_or_failed",
            "selected_article_ids": rescued,
            "confidence": confidence,
            "parse_ok": all(r.get("parse_ok") for r in rounds),
            "rounds": rounds,
            "missing_article_ids": missing,
            "fallback_used": True,
            "fallback_reason": f"stage2_empty_rescue_evidence_top{STAGE2_EMPTY_SELECTION_RESCUE_TOP}",
        }

    single_selection_rescue = False
    if len(selected) == 1 and evidence_order:
        selected = dedupe_keep_order(selected + evidence_order[:STAGE2_SINGLE_SELECTION_RESCUE_TOP])
        single_selection_rescue = True
    selected = order_by_reference(selected, evidence_order)

    parse_ok = all(r.get("parse_ok") for r in rounds)
    fallback_used = single_selection_rescue or not parse_ok or any(r.get("error") for r in rounds)
    return selected, {
        "enabled": True,
        "skipped": False,
        "reason": "stage2_selected",
        "selected_article_ids": selected,
        "confidence": confidence,
        "parse_ok": parse_ok,
        "rounds": rounds,
        "missing_article_ids": missing,
        "fallback_used": fallback_used,
        "fallback_reason": (
            f"stage2_single_selection_rescue_top{STAGE2_SINGLE_SELECTION_RESCUE_TOP}"
            if single_selection_rescue
            else "stage2_partial_issue_selected" if fallback_used else "trust_stage2_selected"
        ),
    }


def verify_one(
    config,
    lookup: ArticleLookup,
    row: dict,
    legal_intents: list[str],
    *,
    stage1_only: bool = False,
    strict_errors: bool = False,
    stage1_compact_candidates: bool = False,
    stage2_compact_candidates: bool = False,
) -> dict:
    qid = row["id"]
    question = row.get("question", "")
    original_order = list(row.get("relevant_articles", []))
    candidate_articles, missing = build_candidate_articles(
        original_order,
        lookup,
        content_max_chars=STAGE1_CONTENT_MAX_CHARS,
    )
    if strict_errors and missing:
        raise RuntimeError(f"stage1 missing {len(missing)} hydrated articles for question {qid}")
    allowed_ids = {c["article_id"] for c in candidate_articles}
    hydrated_order = [a for a in original_order if a in allowed_ids]

    batch_results: list[dict] = []
    selected_all: list[str] = []
    confidence_values: list[str] = []
    parse_ok_count = 0
    error_count = 0
    for batch_index, batch in enumerate(chunks(candidate_articles, BATCH_SIZE)):
        aliased_batch, key_to_id = alias_candidates(
            batch,
            compact=stage1_compact_candidates,
        )
        raw = ""
        verifier_result = None
        parse_ok = False
        error = ""
        try:
            raw = get_worker(config).call(
                question,
                aliased_batch,
                system_prompt=STAGE1_SYSTEM_PROMPT,
                legal_intents=legal_intents,
            )
            verifier_result, parse_ok = parse_stage1_alias_json(raw, key_to_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if parse_ok and verifier_result:
            selected_all.extend(verifier_result.get("selected_article_ids", []))
            confidence_values.append(verifier_result.get("confidence", "low"))
            parse_ok_count += 1
        else:
            error_count += 1
        batch_results.append(
            {
                "batch_index": batch_index,
                "candidate_article_ids": [c["article_id"] for c in batch],
                "candidate_article_keys": key_to_id,
                "selected_article_keys": verifier_result.get("selected_article_keys", []) if verifier_result else [],
                "selected_article_ids": verifier_result.get("selected_article_ids", []) if verifier_result else [],
                "confidence": verifier_result.get("confidence", "low") if verifier_result else "low",
                "parse_ok": parse_ok,
                "raw_response": raw[:2000],
                "error": error,
            }
        )

    selected_all = dedupe_keep_order(selected_all)
    parse_fail_count = len(batch_results) - parse_ok_count
    merged_confidence = merge_confidence(confidence_values, has_error=bool(error_count or parse_fail_count))
    verifier_result = {
        "selected_article_ids": selected_all,
        "confidence": merged_confidence,
    }
    parse_ok = parse_ok_count == len(batch_results) and bool(batch_results)

    technical_issue = error_count > 0 or parse_fail_count > 0
    if strict_errors and technical_issue:
        raise RuntimeError(f"stage1 technical issue for question {qid}")
    stage1_final_article_ids, fallback_used, fallback_reason = apply_fallback(
        verifier_result if selected_all or parse_ok else None,
        hydrated_order,
        technical_issue=technical_issue,
    )
    if stage1_only:
        final_article_ids = stage1_final_article_ids
        stage2_result = {
            "enabled": False,
            "skipped": True,
            "reason": "stage1_only",
            "selected_article_ids": stage1_final_article_ids,
            "confidence": "high",
            "parse_ok": True,
            "rounds": [],
            "fallback_used": False,
            "fallback_reason": "not_needed",
        }
    else:
        final_article_ids, stage2_result = stage2_minimal_select(
            config,
            lookup,
            question=question,
            stage1_article_ids=stage1_final_article_ids,
            legal_intents=legal_intents,
            evidence_order=original_order,
            compact_candidates=stage2_compact_candidates,
            strict_errors=strict_errors,
        )
    return {
        "id": qid,
        "question": question,
        "candidate_article_ids": original_order,
        "hydrated_candidate_count": len(candidate_articles),
        "missing_article_ids": missing,
        "verifier_selected_article_ids": verifier_result.get("selected_article_ids", []) if verifier_result else [],
        "verifier_confidence": verifier_result.get("confidence", "low") if verifier_result else "low",
        "parse_ok": parse_ok,
        "parse_ok_batches": parse_ok_count,
        "batch_count": len(batch_results),
        "batch_results": batch_results,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "stage1_final_article_ids": stage1_final_article_ids,
        "stage2": stage2_result,
        "final_article_ids": final_article_ids,
        "raw_response": "",
        "error": "; ".join([b["error"] for b in batch_results if b.get("error")])[:1000],
    }


def verify_stage2_only_one(
    config,
    lookup: ArticleLookup,
    row: dict,
    legal_intents: list[str],
    *,
    stage2_compact_candidates: bool = False,
    strict_errors: bool = False,
) -> dict:
    qid = row["id"]
    question = row.get("question", "")
    stage1_final_article_ids = dedupe_keep_order(row.get("final_article_ids", []))
    final_article_ids, stage2_result = stage2_minimal_select(
        config,
        lookup,
        question=question,
        stage1_article_ids=stage1_final_article_ids,
        legal_intents=legal_intents,
        evidence_order=row.get("candidate_article_ids", stage1_final_article_ids),
        compact_candidates=stage2_compact_candidates,
        strict_errors=strict_errors,
    )
    return {
        "id": qid,
        "question": question,
        "candidate_article_ids": row.get("candidate_article_ids", stage1_final_article_ids),
        "hydrated_candidate_count": len(stage1_final_article_ids),
        "missing_article_ids": [],
        "verifier_selected_article_ids": stage1_final_article_ids,
        "verifier_confidence": row.get("verifier_confidence", "low"),
        "parse_ok": stage2_result.get("parse_ok", False),
        "parse_ok_batches": 0,
        "batch_count": 0,
        "batch_results": [],
        "fallback_used": row.get("fallback_used", False),
        "fallback_reason": row.get("fallback_reason", "stage1_cached"),
        "stage1_final_article_ids": stage1_final_article_ids,
        "stage2": stage2_result,
        "final_article_ids": final_article_ids,
        "raw_response": "",
        "error": "; ".join(
            [
                round_row.get("error", "")
                for round_row in stage2_result.get("rounds", [])
                if round_row.get("error")
            ]
        )[:1000],
    }


def to_submission_row(record: dict) -> dict:
    articles = dedupe_keep_order(record.get("final_article_ids", []))
    docs = dedupe_keep_order([doc_from_article_ref(article) for article in articles])
    return {
        "id": record["id"],
        "question": record.get("question", ""),
        "answer": "",
        "relevant_docs": docs,
        "relevant_articles": articles,
    }


def write_outputs(output_dir: Path, diagnostics_path: Path, questions: list[dict], records: dict) -> list[dict]:
    ordered_records = [
        records.get(
            row["id"],
            {
                "id": row["id"],
                "question": row.get("question", ""),
                "final_article_ids": row.get("relevant_articles", [])[:5],
                "parse_ok": False,
                "fallback_used": True,
                "fallback_reason": "missing_cache_fallback",
            },
        )
        for row in questions
    ]
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(ordered_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    submission = [to_submission_row(row) for row in ordered_records]
    save_submission(submission, output_dir)
    return submission


def load_legal_intents(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    rows = read_json(path)
    out: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row.get("id", ""))
        intents = []
        for item in row.get("legal_intents", []) or []:
            if isinstance(item, dict):
                value = item.get("intent")
            else:
                value = item
            if value:
                intents.append(norm_text(str(value)))
        if qid:
            out[qid] = intents
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Batch-wise bias-free LLM verifier for tiered candidate submissions.")
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input", default="outputs/submission_bge_intent_tiered_rrf12_bge5_rawintent5_clean/results.json")
    parser.add_argument(
        "--stage2-only-from-diagnostics",
        default=None,
        help="Read stage-1 verifier diagnostics and run only stage 2 on final_article_ids.",
    )
    parser.add_argument("--corpus", default="corpus/data/corpus_clean_asof_20260301.json")
    parser.add_argument("--intent-results", default="outputs/intent_ranked_hits_clean_results.json")
    parser.add_argument("--cache", default="cache/llm_stage2_adaptive_min2_rescue1_c1600_prompt_v2.jsonl")
    parser.add_argument("--output-dir", default="outputs/submission_stage2_adaptive_min2_rescue1_c1600_prompt_v2")
    parser.add_argument("--diagnostics", default="outputs/diagnostics_stage2_adaptive_min2_rescue1_c1600_prompt_v2.json")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--stage1-compact-candidates",
        action="store_true",
        help="Use compact key/source/article/content objects in Stage 1 only.",
    )
    parser.add_argument(
        "--stage2-compact-candidates",
        action="store_true",
        help="Use compact A1/A2 key/source/article/content objects in Stage 2.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run only Stage 1 and export stage1_final_article_ids as final_article_ids.",
    )
    parser.add_argument(
        "--strict-errors",
        action="store_true",
        help="Do not cache fallback records for technical errors; leave failed questions for resume.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    problems = validate_config(config)
    if problems:
        raise ValueError(f"Invalid config: {'; '.join(problems)}")

    stage2_only = bool(args.stage2_only_from_diagnostics)
    rows = list(read_json(Path(args.stage2_only_from_diagnostics if stage2_only else args.input)))
    if args.limit:
        rows = rows[: args.limit]
    lookup = ArticleLookup(args.corpus)
    legal_intents_by_id = load_legal_intents(Path(args.intent_results))
    cache_path = Path(args.cache)
    acquire_cache_run_lock(cache_path)
    prepare_cache_for_run(cache_path, args.resume)
    records = load_cache(cache_path) if args.resume else {}
    todo = [row for row in rows if row["id"] not in records]
    log.info(
        "LLM verifier total=%d todo=%d workers=%d input=%s intents=%d stage1_chars=%d stage2_chars=%d stage2_min_articles=%d",
        len(rows),
        len(todo),
        args.workers,
        args.stage2_only_from_diagnostics if stage2_only else args.input,
        len(legal_intents_by_id),
        STAGE1_CONTENT_MAX_CHARS,
        STAGE2_CONTENT_MAX_CHARS,
        STAGE2_MIN_ARTICLES,
    )

    lock = threading.Lock()
    started = time.time()
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        if stage2_only:
            futures = {
                ex.submit(
                    verify_stage2_only_one,
                    config,
                    lookup,
                    row,
                    legal_intents_by_id.get(str(row["id"]), []),
                    stage2_compact_candidates=args.stage2_compact_candidates,
                    strict_errors=args.strict_errors,
                ): row["id"]
                for row in todo
            }
        else:
            futures = {
                ex.submit(
                    verify_one,
                    config,
                    lookup,
                    row,
                    legal_intents_by_id.get(str(row["id"]), []),
                    stage1_only=args.stage1_only,
                    strict_errors=args.strict_errors,
                    stage1_compact_candidates=args.stage1_compact_candidates,
                    stage2_compact_candidates=args.stage2_compact_candidates,
                ): row["id"]
                for row in todo
            }
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:
                if args.strict_errors:
                    log.error("Q%s failed and will be left uncached: %s", qid, exc)
                    processed += 1
                    if processed % 10 == 0 or processed == len(todo):
                        rate = processed / max(time.time() - started, 1e-6)
                        eta = (len(todo) - processed) / rate if rate else 0
                        log.info("%d/%d done (%.2f q/s, ETA %.1f min)", len(records), len(rows), rate, eta / 60)
                    continue
                log.warning("Q%s failed outside verifier: %s", qid, exc)
                source = next(row for row in rows if row["id"] == qid)
                record = {
                    "id": qid,
                    "question": source.get("question", ""),
                    "candidate_article_ids": source.get("relevant_articles", []),
                    "final_article_ids": source.get("relevant_articles", [])[:5],
                    "parse_ok": False,
                    "fallback_used": True,
                    "fallback_reason": "exception_fallback",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records[record["id"]] = record
            append_jsonl(cache_path, record, lock)
            processed += 1
            if processed % 10 == 0 or processed == len(todo):
                rate = processed / max(time.time() - started, 1e-6)
                eta = (len(todo) - processed) / rate if rate else 0
                log.info("%d/%d done (%.2f q/s, ETA %.1f min)", len(records), len(rows), rate, eta / 60)

    missing_after_run = [row["id"] for row in rows if row["id"] not in records]
    if missing_after_run and args.strict_errors:
        raise RuntimeError(
            f"{len(missing_after_run)} questions are not cached yet. "
            f"Resume the same command to retry: {missing_after_run[:20]}"
        )

    submission = write_outputs(Path(args.output_dir), Path(args.diagnostics), rows, records)
    sizes = [len(row["relevant_articles"]) for row in submission]
    diag_rows = list(records.values())
    log.info(
        "Final rows=%d min/max/mean=%d/%d/%.2f parse_ok=%d errors=%d",
        len(submission),
        min(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
        statistics.mean(sizes) if sizes else 0,
        sum(1 for row in diag_rows if row.get("parse_ok")),
        sum(1 for row in diag_rows if row.get("error")),
    )


if __name__ == "__main__":
    main()
