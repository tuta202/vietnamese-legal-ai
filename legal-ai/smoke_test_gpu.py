"""
smoke_test_gpu.py — TIP-012 Step 1 gate.

Four cheap live checks against the Gpu endpoints before the expensive
re-embed / full run:
  1a env vars present
  1b embedder returns a valid 4096-d vector
  1c LLM returns a non-empty Vietnamese string
  1d reranker parses JSON scores (not falling back) on >=2/3 tries

Run:
    python smoke_test_gpu.py --config config_gpu.yaml
Exit code 0 = all gates pass; non-zero = at least one failed.
"""
from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retrieval.config import _load_dotenv, load_config  # noqa: E402


def _has_vietnamese(s: str) -> bool:
    """True if the string contains a Vietnamese-specific diacritic letter."""
    for ch in s:
        if "WITH" in unicodedata.name(ch, "") and ch.isalpha():
            return True
    # đ/Đ specifically
    return "đ" in s or "Đ" in s


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_gpu.yaml")
    args = ap.parse_args()

    _load_dotenv()
    import os

    results: list[tuple[str, bool, str]] = []

    # --- 1a env vars -------------------------------------------------------
    needed = ["GCP_PROJECT", "GPU_EMBED_ENDPOINT_ID", "GPU_LLM_ENDPOINT_ID"]
    missing = [k for k in needed if not os.environ.get(k)]
    ok_env = not missing
    results.append((
        "1a env vars", ok_env,
        "all set" if ok_env else f"MISSING: {', '.join(missing)}",
    ))
    for k in needed:
        v = os.environ.get(k, "")
        shown = v if k == "GCP_PROJECT" else (v[:18] + "…" if len(v) > 18 else v)
        print(f"    {k} = {shown or '(empty)'}")
    if not ok_env:
        _report(results)
        return 1

    config = load_config(args.config)
    from gpu_backends import make_gpu_components
    embedder, rewriter, reranker, _gen = make_gpu_components(config, mock=False)

    # --- 1b embedder -------------------------------------------------------
    try:
        t0 = time.time()
        vec = embedder.embed_query(
            "Luật doanh nghiệp 2020 quy định về thành lập công ty"
        )
        ms = _ms(t0)
        import numpy as np
        ok_emb = (
            hasattr(vec, "shape")
            and vec.shape == (config.embedding.dimension,)
            and not np.isnan(vec).any()
            and float(np.abs(vec).sum()) > 0.0
        )
        results.append((
            "1b embedder", ok_emb,
            f"dim={getattr(vec, 'shape', '?')}, {ms}ms"
            + ("" if ok_emb else " — INVALID (wrong dim / NaN / all-zero)"),
        ))
    except Exception as e:
        results.append(("1b embedder", False, f"EXCEPTION: {type(e).__name__}: {e}"))

    # --- 1c LLM ------------------------------------------------------------
    try:
        t0 = time.time()
        raw = rewriter._chat_complete([
            {"role": "user", "content":
                "Theo pháp luật Việt Nam, điều kiện thành lập doanh nghiệp là gì? "
                "Trả lời ngắn gọn."},
        ])
        ms = _ms(t0)
        ok_llm = bool(raw and raw.strip()) and _has_vietnamese(raw)
        results.append((
            "1c LLM", ok_llm,
            f"{len(raw)} chars, vi={_has_vietnamese(raw)}, {ms}ms",
        ))
        print(f"    LLM sample: {raw.strip()[:120]!r}")
    except Exception as e:
        results.append(("1c LLM", False, f"EXCEPTION: {type(e).__name__}: {e}"))

    # --- 1d reranker JSON --------------------------------------------------
    q = "Điều kiện thành lập doanh nghiệp tại Việt Nam?"
    docs = [
        {"chunk_id": "x1", "dieu_number": "Điều 17", "dieu_title": "Quyền thành lập doanh nghiệp",
         "content": "Tổ chức, cá nhân có quyền thành lập và quản lý doanh nghiệp tại Việt Nam."},
        {"chunk_id": "x2", "dieu_number": "Điều 168", "dieu_title": "Khai thác khoáng sản",
         "content": "Việc khai thác khoáng sản phải có giấy phép của cơ quan nhà nước."},
        {"chunk_id": "x3", "dieu_number": "Điều 27", "dieu_title": "Đăng ký doanh nghiệp",
         "content": "Doanh nghiệp được cấp Giấy chứng nhận đăng ký doanh nghiệp khi đủ điều kiện."},
    ]
    success = 0
    times = []
    for _ in range(3):
        try:
            t0 = time.time()
            scores = reranker._llm_scores(q, docs)
            times.append(_ms(t0))
            if scores:
                success += 1
        except Exception as e:
            print(f"    rerank try EXCEPTION: {type(e).__name__}: {e}")
    ok_rr = success >= 2
    avg_ms = int(sum(times) / len(times)) if times else 0
    results.append((
        "1d reranker", ok_rr,
        f"{success}/3 parsed JSON, avg {avg_ms}ms"
        + ("" if ok_rr else " — Llama not emitting valid JSON"),
    ))

    return _report(results)


def _report(results: list[tuple[str, bool, str]]) -> int:
    print("\n" + "=" * 60)
    print("SMOKE TEST — TIP-012 Gate 1")
    print("=" * 60)
    all_ok = True
    for name, ok, detail in results:
        mark = "✅" if ok else "❌"
        all_ok = all_ok and ok
        print(f"  {mark}  {name:<14} {detail}")
    print("=" * 60)
    print("GATE 1:", "PASS — proceed to re-embed" if all_ok else "FAIL — do NOT proceed")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
