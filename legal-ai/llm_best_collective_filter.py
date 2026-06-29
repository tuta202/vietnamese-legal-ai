from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from article_lookup import ArticleLookup  # noqa: E402
from llm_candidate_verifier import (  # noqa: E402
    acquire_cache_run_lock,
    build_candidate_articles,
    chunks,
    dedupe_keep_order,
    get_worker,
    load_config,
    load_legal_intents,
    order_by_reference,
    prepare_cache_for_run,
    validate_config,
)
from submit import save_submission  # noqa: E402


log = logging.getLogger("llm_best_collective_filter")

SYSTEM_PROMPT = """\
You select the necessary Vietnamese legal articles for one legal question.

Use only:
- question
- legal_intents
- candidate_articles

Keep articles that are needed for the final legal answer:
- direct answer articles
- definitions that are required to understand the answer
- obligation/conduct articles when the question asks about penalty, violation, coercion, or consequence
- exception/condition articles when the question asks about a special case
- one article for each independent legal intent
- both a general law article and a detailed decree/circular article when they add different legal details
- an article whose title directly names the requested issue, unless another selected article fully covers the same issue more specifically

Remove articles that are only:
- same broad topic but wrong subtask
- another dossier/procedure/benefit/fund/program
- general scope, responsibility, implementation, or effective-date background
- clearly redundant because another selected article covers the same legal point and this article adds no legal detail

If unsure, keep the article.
Return valid JSON only:
{
  "selected_article_keys": [],
  "confidence": "high|medium|low"
}
"""

STAGE2_RECALL_PROMPT = """\
You filter Vietnamese legal articles for a recall-safe second stage.

Use only:
- question
- legal_intents
- candidate_articles

Keep an article if it may be needed to answer the question or any legal intent:
- direct rule, definition, condition, exception, procedure, deadline, authority
- right, obligation, violation, penalty, legal consequence, required dossier/evidence
- one useful article for each independent legal intent
- both general law and detailed decree/circular if they add different legal details

Remove only articles that are clearly unrelated, wrong subtask, or fully duplicated by another kept article.
If unsure, keep the article.

Return valid JSON only:
{
  "selected_article_keys": [],
  "confidence": "high|medium|low"
}
"""

PROMPTS = {
    "final_precision": SYSTEM_PROMPT,
    "stage2_recall": STAGE2_RECALL_PROMPT,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def doc_from_article_ref(article_ref: str) -> str:
    doc, sep, _ = str(article_ref).rpartition("|")
    return doc if sep else str(article_ref)


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


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("id", ""))
            if qid:
                out[qid] = row
    return out


def append_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


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


def parse_collective_json(raw: str, key_to_id: dict[str, str]) -> tuple[dict | None, bool]:
    from backends_common import strip_code_fences  # noqa: PLC0415
    import re  # noqa: PLC0415

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
    if "selected_article_keys" not in data:
        return None, False
    keys = data["selected_article_keys"]
    if not isinstance(keys, list):
        return None, False
    normalized_keys = [str(key).strip() for key in keys]
    if any(key not in key_to_id for key in normalized_keys):
        return None, False
    selected: list[str] = []
    for key in normalized_keys:
        article_id = key_to_id[key]
        if article_id not in selected:
            selected.append(article_id)
    confidence = str(data.get("confidence", "low")).lower().strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "selected_article_ids": selected,
        "selected_article_keys": normalized_keys,
        "confidence": confidence,
    }, True


def call_collective(
    config,
    question: str,
    candidates: list[dict],
    legal_intents: list[str],
    *,
    system_prompt: str,
    compact_candidates: bool = False,
) -> dict:
    aliased_candidates, key_to_id = alias_candidates(candidates, compact=compact_candidates)
    raw = ""
    parsed = None
    parse_ok = False
    error = ""
    try:
        raw = get_worker(config).call(
            question,
            aliased_candidates,
            system_prompt=system_prompt,
            legal_intents=legal_intents,
        )
        parsed, parse_ok = parse_collective_json(raw, key_to_id)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {
        "candidate_article_ids": [c["article_id"] for c in candidates],
        "candidate_article_keys": {key: article_id for key, article_id in key_to_id.items()},
        "selected_article_keys": parsed.get("selected_article_keys", []) if parsed else [],
        "selected_article_ids": parsed.get("selected_article_ids", []) if parsed else [],
        "confidence": parsed.get("confidence", "low") if parsed else "low",
        "parse_ok": parse_ok,
        "raw_response": raw[:2000],
        "error": error,
    }


def merge_rounds_selected(rounds: list[dict]) -> list[str]:
    selected: list[str] = []
    for round_row in rounds:
        if round_row.get("parse_ok"):
            selected.extend(round_row.get("selected_article_ids", []))
    return dedupe_keep_order(selected)


def process_question(
    config,
    lookup: ArticleLookup,
    row: dict,
    legal_intents: list[str],
    *,
    content_max_chars: int,
    batch_size: int,
    direct_max: int,
    min_size: int,
    preserve_top1: bool,
    strict_errors: bool,
    system_prompt: str,
    prompt_mode: str,
    compact_candidates: bool,
) -> dict:
    qid = row["id"]
    question = row.get("question", "")
    original = dedupe_keep_order(row.get("final_article_ids", []))
    if len(original) < min_size:
        return {
            **row,
            "final_article_ids": original,
            "collective_filter": {
                "enabled": True,
                "skipped": True,
                "reason": f"final_size_lt_{min_size}",
                "input_article_ids": original,
                "selected_article_ids": original,
                "dropped_article_ids": [],
                "rounds": [],
                "content_max_chars": content_max_chars,
                "preserve_top1": preserve_top1,
                "prompt_mode": prompt_mode,
            },
        }

    candidates, missing = build_candidate_articles(original, lookup, content_max_chars=content_max_chars)
    if strict_errors and missing:
        raise RuntimeError(f"collective missing {len(missing)} hydrated articles for question {qid}")
    if not candidates:
        return {
            **row,
            "final_article_ids": original,
            "collective_filter": {
                "enabled": True,
                "skipped": False,
                "reason": "no_hydrated_candidates_keep_original",
                "input_article_ids": original,
                "selected_article_ids": original,
                "dropped_article_ids": [],
                "missing_article_ids": missing,
                "rounds": [],
                "content_max_chars": content_max_chars,
                "preserve_top1": preserve_top1,
                "prompt_mode": prompt_mode,
            },
        }

    rounds: list[dict] = []
    if len(candidates) <= direct_max:
        result = call_collective(
            config,
            question,
            candidates,
            legal_intents,
            system_prompt=system_prompt,
            compact_candidates=compact_candidates,
        )
        result["round"] = "direct"
        result["batch_index"] = 0
        rounds.append(result)
        selected = result["selected_article_ids"] if result["parse_ok"] else []
        parse_ok = result["parse_ok"]
        has_error = bool(result.get("error")) or not parse_ok
    else:
        intermediate: list[str] = []
        has_error = False
        for batch_index, batch in enumerate(chunks(candidates, batch_size)):
            result = call_collective(
                config,
                question,
                batch,
                legal_intents,
                system_prompt=system_prompt,
                compact_candidates=compact_candidates,
            )
            result["round"] = "batch"
            result["batch_index"] = batch_index
            rounds.append(result)
            if result["parse_ok"]:
                intermediate.extend(result["selected_article_ids"])
            else:
                has_error = True
        intermediate = dedupe_keep_order(intermediate)
        if intermediate:
            final_candidates, final_missing = build_candidate_articles(
                intermediate,
                lookup,
                content_max_chars=content_max_chars,
            )
            missing.extend(final_missing)
            if strict_errors and final_missing:
                raise RuntimeError(
                    f"collective global round missing {len(final_missing)} hydrated articles for question {qid}"
                )
            result = call_collective(
                config,
                question,
                final_candidates,
                legal_intents,
                system_prompt=system_prompt,
                compact_candidates=compact_candidates,
            )
            result["round"] = "global"
            result["batch_index"] = 0
            rounds.append(result)
            selected = result["selected_article_ids"] if result["parse_ok"] else intermediate
            has_error = has_error or bool(result.get("error")) or not result["parse_ok"]
        else:
            selected = []
            has_error = True
        parse_ok = all(r.get("parse_ok") for r in rounds)

    selected = dedupe_keep_order(selected)
    fallback_used = False
    fallback_reason = ""
    if not selected:
        if strict_errors and any((not r.get("parse_ok")) or r.get("error") for r in rounds):
            raise RuntimeError(f"collective call failed for question {qid}")
        selected = original[:2]
        fallback_used = True
        fallback_reason = "empty_or_failed_rescue_top2"
    elif len(selected) == 1:
        selected = dedupe_keep_order(selected + original[:1])
        fallback_used = True
        fallback_reason = "single_selection_rescue_top1"
    elif has_error:
        if strict_errors:
            raise RuntimeError(f"collective partial parse/error for question {qid}")
        fallback_used = True
        fallback_reason = "partial_parse_or_error"

    top1_rescue_used = False
    if preserve_top1 and original:
        before = set(selected)
        selected = dedupe_keep_order(selected + original[:1])
        top1_rescue_used = set(selected) != before

    final = order_by_reference(selected, original)
    dropped = [article_id for article_id in original if article_id not in set(final)]
    return {
        **row,
        "final_article_ids": final,
        "collective_filter": {
            "enabled": True,
            "skipped": False,
            "reason": "collective_selected",
            "input_article_ids": original,
            "selected_article_ids": final,
            "dropped_article_ids": dropped,
            "legal_intents": legal_intents,
            "parse_ok": parse_ok,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "preserve_top1": preserve_top1,
            "top1_rescue_used": top1_rescue_used,
            "missing_article_ids": missing,
            "rounds": rounds,
            "content_max_chars": content_max_chars,
            "batch_size": batch_size,
            "direct_max": direct_max,
            "prompt_mode": prompt_mode,
            "compact_candidates": compact_candidates,
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Collective LLM filter for current best submission diagnostics.")
    parser.add_argument("--config", default="config_vertex_clean.yaml")
    parser.add_argument("--input-diagnostics", default="outputs/submission_drop_penalty_when_not_needed/diagnostics.json")
    parser.add_argument("--intent-results", default="outputs/intent_ranked_hits_clean_results.json")
    parser.add_argument("--corpus", default="corpus/data/corpus_clean_asof_20260301.json")
    parser.add_argument("--cache", default="cache/llm_best_collective_filter_c2200.jsonl")
    parser.add_argument("--output-dir", default="outputs/submission_best_collective_filter_c2200")
    parser.add_argument("--diagnostics", default="outputs/diagnostics_best_collective_filter_c2200.json")
    parser.add_argument("--content-max-chars", type=int, default=2200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--direct-max", type=int, default=8)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=sorted(PROMPTS),
        default="final_precision",
        help="Use final_precision for the proven final filter, or stage2_recall for recall-safe Stage 2.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preserve-top1", action="store_true")
    parser.add_argument(
        "--compact-candidates",
        action="store_true",
        help="Use compact A1/A2 key/source/article/content candidate objects.",
    )
    parser.add_argument(
        "--strict-errors",
        action="store_true",
        help="Do not cache fallback records for LLM errors or parse failures; leave them for resume.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    problems = validate_config(config)
    if problems:
        raise ValueError(f"Invalid config: {'; '.join(problems)}")
    system_prompt = PROMPTS[args.prompt_mode]

    rows = list(read_json(Path(args.input_diagnostics)))
    if args.limit:
        rows = rows[: args.limit]
    intents = load_legal_intents(Path(args.intent_results))
    lookup = ArticleLookup(args.corpus)
    cache_path = Path(args.cache)
    acquire_cache_run_lock(cache_path)
    prepare_cache_for_run(cache_path, args.resume)
    cached = load_cache(cache_path) if args.resume else {}
    todo = [row for row in rows if str(row["id"]) not in cached]
    log.info(
        "collective filter rows=%d cached=%d todo=%d workers=%d content_chars=%d batch=%d direct_max=%d prompt_mode=%s",
        len(rows),
        len(cached),
        len(todo),
        args.workers,
        args.content_max_chars,
        args.batch_size,
        args.direct_max,
        args.prompt_mode,
    )

    lock = threading.Lock()
    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_question,
                config,
                lookup,
                row,
                intents.get(str(row["id"]), []),
                content_max_chars=args.content_max_chars,
                batch_size=args.batch_size,
                direct_max=args.direct_max,
                min_size=args.min_size,
                preserve_top1=args.preserve_top1,
                strict_errors=args.strict_errors,
                system_prompt=system_prompt,
                prompt_mode=args.prompt_mode,
                compact_candidates=args.compact_candidates,
            ): row["id"]
            for row in todo
        }
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # noqa: BLE001
                if args.strict_errors:
                    log.error("Question %s failed and will be left uncached: %s: %s", qid, type(exc).__name__, exc)
                    done += 1
                    if done % 10 == 0 or done == len(todo):
                        rate = done / max(time.time() - started, 1e-6)
                        eta = (len(todo) - done) / rate if rate else 0
                        log.info("%d/%d questions done (%.2f q/s, ETA %.1f min)", done, len(todo), rate, eta / 60)
                    continue
                row = next(r for r in todo if r["id"] == qid)
                record = {
                    **row,
                    "final_article_ids": dedupe_keep_order(row.get("final_article_ids", [])),
                    "collective_filter": {
                        "enabled": True,
                        "skipped": False,
                        "reason": "exception_keep_original",
                        "input_article_ids": dedupe_keep_order(row.get("final_article_ids", [])),
                        "selected_article_ids": dedupe_keep_order(row.get("final_article_ids", [])),
                        "dropped_article_ids": [],
                        "rounds": [],
                        "fallback_used": True,
                        "fallback_reason": "exception_keep_original",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                }
            cached[str(qid)] = record
            append_jsonl(cache_path, record, lock)
            done += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.time() - started, 1e-6)
                eta = (len(todo) - done) / rate if rate else 0
                log.info("%d/%d questions done (%.2f q/s, ETA %.1f min)", done, len(todo), rate, eta / 60)

    missing_after_run = [str(row["id"]) for row in rows if str(row["id"]) not in cached]
    if missing_after_run:
        message = (
            f"{len(missing_after_run)} questions are not cached yet. "
            f"Resume the same command to retry: {', '.join(missing_after_run[:20])}"
        )
        if args.strict_errors:
            raise RuntimeError(message)
        raise KeyError(message)

    records = [cached[str(row["id"])] for row in rows]
    diagnostics_path = Path(args.diagnostics)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    submission = [to_submission_row(row) for row in records]
    save_submission(submission, Path(args.output_dir))
    sizes = [len(row["relevant_articles"]) for row in submission]
    dropped = sum(len(row.get("collective_filter", {}).get("dropped_article_ids", [])) for row in records)
    changed = sum(1 for row in records if row.get("collective_filter", {}).get("dropped_article_ids"))
    parse_ok = sum(
        1
        for row in records
        if row.get("collective_filter", {}).get("skipped")
        or row.get("collective_filter", {}).get("parse_ok")
    )
    log.info(
        "Final rows=%d min/max/mean=%d/%d/%.2f changed=%d dropped=%d parse_or_skip=%d",
        len(submission),
        min(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
        statistics.mean(sizes) if sizes else 0,
        changed,
        dropped,
        parse_ok,
    )


if __name__ == "__main__":
    main()
