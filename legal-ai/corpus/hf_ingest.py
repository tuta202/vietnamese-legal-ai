"""
hf_ingest.py — TIP-CORPUS-001 two-tier corpus builder.

Reads the cached th1nhng0 parquet files (see hf_download.py), filters by
effect-status / doc-type / SME-domain, converts HTML→text (tier A) or uses
plain text (tier B), parses each Điều into an Article (reusing the existing
LegalTextParser — schema unchanged), formats citations, dedups (tier A > B),
and writes an expanded corpus.json + SOURCES.md.

Design notes / deviations from TIP (documented in Completion Report):
  • Uses pyarrow to read parquet directly instead of datasets.load_dataset,
    because datasets streaming raises ArrowInvalid 'large_string -> string'
    on the oversized content columns. Download/cache is via hf_download.py
    (hf_hub_download — resumable).
  • Tier B content (3.5GB) is streamed in row-group batches; only ids that
    survive the tier-B metadata filter are materialised (bounded memory).

CLI:
    python corpus/hf_ingest.py                       # full build (A+B)
    python corpus/hf_ingest.py --tier A              # tier A only
    python corpus/hf_ingest.py --limit-docs 500      # quick smoke build
    python corpus/hf_ingest.py --config corpus/hf_config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
from bs4 import BeautifulSoup

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
from parser import LegalTextParser  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf_ingest")

_DEFAULT_CONFIG = _HERE / "hf_config.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str | None) -> str:
    return (s or "").strip()


def _casefold_in(value: str, keep_list_cf: set[str]) -> bool:
    return _norm(value).casefold() in keep_list_cf


def html_to_text(html: str) -> str:
    """Strip HTML to plain text; collapse whitespace handled later by parser."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # Drop script/style noise
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class HFIngest:
    def __init__(self, config: dict, limit_docs: int | None = None) -> None:
        self.cfg = config
        self.limit_docs = limit_docs
        self.cache = _HERE / config["cache_dir"]
        self.parser = LegalTextParser()

        f = config["filters"]
        self.exclude_code_patterns = [p.casefold() for p in f.get("exclude_code_patterns", [])]
        self.doc_type_keep_cf = {t.casefold() for t in f["doc_type_keep_vi"]}
        self.legal_type_map = {k.casefold(): v for k, v in f["legal_type_map_en_vi"].items()}
        self.domain_kw = [k.casefold() for k in f["domain_keywords"]]
        self.domain_apply = f["domain_apply"]
        self.effect_keep = {
            tier: {s.casefold() for s in lst}
            for tier, lst in f["effect_keep"].items()
        }
        self.name_variant = config["citation"]["name_variant"]
        self.normalize_names = config["citation"].get("normalize_names_from_hf", False)
        self.prefer_tier = config["dedup"]["prefer_tier"]
        self.drop_khoan = config["output"].get("drop_khoan_list", False)
        self._title_map: dict[str, str] = {}   # code(casefold) -> official HF title

        # stats
        self.stats: dict = {
            "0": {"articles": 0},                 # original corpus (preferred)
            "A": {"meta_rows": 0, "kept_meta": 0, "docs_parsed": 0, "no_content": 0,
                  "no_articles": 0, "articles": 0},
            "B": {"meta_rows": 0, "kept_meta": 0, "docs_parsed": 0, "no_content": 0,
                  "no_articles": 0, "articles": 0},
            "dedup_dropped": 0,
            "type_mapping": {},   # raw doc_type -> kept(normalised)/dropped
            "scan_pdf_skipped": 0,
        }
        self._seen_keys: set[str] = set()      # dedup keys (code|dieu)
        self._articles: list[dict] = []        # final article dicts

    # ------------------------------------------------------------------
    # Domain / type filters
    # ------------------------------------------------------------------

    def _domain_match(self, haystack: str) -> bool:
        h = haystack.casefold()
        return any(kw in h for kw in self.domain_kw)

    def _normalise_type(self, raw_type: str, tier: str) -> str | None:
        """Return the canonical Vietnamese type if kept, else None."""
        rt = _norm(raw_type)
        if not rt:
            return None
        if tier == "B":
            mapped = self.legal_type_map.get(rt.casefold())
            if mapped:
                self.stats["type_mapping"][f"B:{rt}"] = f"KEEP→{mapped}"
                return mapped
            self.stats["type_mapping"].setdefault(f"B:{rt}", "DROP")
            return None
        # tier A — already Vietnamese; case-normalise against keep list
        if rt.casefold() in self.doc_type_keep_cf:
            # canonical casing = the keep-list entry
            canon = next(t for t in self.cfg["filters"]["doc_type_keep_vi"]
                         if t.casefold() == rt.casefold())
            self.stats["type_mapping"][f"A:{rt}"] = f"KEEP→{canon}"
            return canon
        self.stats["type_mapping"].setdefault(f"A:{rt}", "DROP")
        return None

    # ------------------------------------------------------------------
    # Metadata filtering → returns {id: law_meta}
    # ------------------------------------------------------------------

    def filter_meta(self, tier: str) -> dict[str, dict]:
        fld = self.cfg["fields"][tier]
        meta_path = self.cache / self.cfg["tiers"][tier]["meta"]
        cols = ["id", fld["doc_code"], fld["doc_type"], fld["title"], fld["effect"], *fld["sectors"]]
        tbl = pq.read_table(meta_path, columns=cols)
        rows = tbl.to_pylist()
        self.stats[tier]["meta_rows"] = len(rows)

        keep_effect = self.effect_keep[tier]
        apply_domain = self.domain_apply[tier]

        kept: dict[str, dict] = {}
        for r in rows:
            # effect status
            if not _casefold_in(_norm(r.get(fld["effect"])), keep_effect):
                continue
            # doc type
            ctype = self._normalise_type(r.get(fld["doc_type"]), tier)
            if ctype is None:
                continue
            # doc code (must exist to cite)
            code = _norm(r.get(fld["doc_code"]))
            if not code:
                continue
            # exclude local-government documents (HĐND/UBND/HU/TU) — central only
            code_cf = code.casefold()
            if any(p in code_cf for p in self.exclude_code_patterns):
                self.stats[tier]["excluded_local"] = self.stats[tier].get("excluded_local", 0) + 1
                continue
            title = _norm(r.get(fld["title"]))
            # domain
            if apply_domain:
                sectors = " ".join(_norm(r.get(s)) for s in fld["sectors"])
                if not self._domain_match(f"{title} {sectors}"):
                    continue
            doc_id = str(r["id"])
            kept[doc_id] = {"id": code, "type": ctype, "name": title or code}

        self.stats[tier]["kept_meta"] = len(kept)
        log.info("Tier %s: %d meta rows → %d kept after filters",
                 tier, len(rows), len(kept))
        return kept

    # ------------------------------------------------------------------
    # Content streaming + parse
    # ------------------------------------------------------------------

    def ingest_tier(self, tier: str) -> None:
        if not self.cfg["tiers"][tier]["enabled"]:
            log.info("Tier %s disabled — skipping", tier)
            return
        keep = self.filter_meta(tier)
        if not keep:
            return

        tcfg = self.cfg["tiers"][tier]
        content_path = self.cache / tcfg["content"]
        if not content_path.exists():
            log.warning("Tier %s content parquet missing (%s) — skipping content",
                        tier, content_path)
            return
        cfield = tcfg["content_field"]
        is_html = tcfg["content_is_html"]

        pf = pq.ParquetFile(content_path)
        processed = 0
        seen_ids: set[str] = set()   # content parquet has duplicate id rows; parse once
        for batch in pf.iter_batches(batch_size=2000, columns=["id", cfield]):
            d = batch.to_pydict()
            ids = d["id"]
            contents = d[cfield]
            for doc_id, raw in zip(ids, contents):
                sid = str(doc_id)
                meta = keep.get(sid)
                if meta is None:
                    continue
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                if self.limit_docs and processed >= self.limit_docs:
                    log.info("Tier %s: hit --limit-docs=%d", tier, self.limit_docs)
                    self._finalise_tier_stats(tier, processed)
                    return
                processed += 1
                if not _norm(raw):
                    self.stats[tier]["no_content"] += 1
                    self.stats["scan_pdf_skipped"] += 1
                    continue
                text = html_to_text(raw) if is_html else raw
                arts = self.parser.parse(text, meta)
                if not arts:
                    self.stats[tier]["no_articles"] += 1
                    continue
                self._add_dicts([a.to_dict() for a in arts], tier)

        self._finalise_tier_stats(tier, processed)

    def _finalise_tier_stats(self, tier: str, processed: int) -> None:
        self.stats[tier]["docs_parsed"] = processed
        log.info("Tier %s: parsed %d docs → %d articles (no_content=%d, no_articles=%d)",
                 tier, processed, self.stats[tier]["articles"],
                 self.stats[tier]["no_content"], self.stats[tier]["no_articles"])

    # ------------------------------------------------------------------
    # Citation variant + dedup
    # ------------------------------------------------------------------

    def _normalize_name(self, art: dict) -> None:
        """Override law_name with the official full trích yếu from HF metadata
        (type-prefix stripped + first letter capitalised). Keeps law_type.
        No-op if disabled or the code isn't in the title map (keeps existing name)."""
        if not self.normalize_names:
            return
        raw = self._title_map.get(_norm(art["law_id"]).casefold())
        if not raw:
            return
        name = LegalTextParser._strip_type_prefix(raw, art["law_type"]).strip()
        if name:
            art["law_name"] = name[0].upper() + name[1:]

    def _apply_citation_variant(self, art: dict) -> None:
        """(Re)build the citation strings from law_id/type/name so they stay in
        sync after any name normalisation."""
        lid, lt, ln, dn = (art["law_id"], art["law_type"],
                           art["law_name"], art["dieu_number"])
        doc = f"{lid}|{lt} {lid} {ln}" if self.name_variant == "loai_makh_title" \
            else f"{lid}|{lt} {ln}"
        art["relevant_doc_str"] = doc
        art["relevant_article_str"] = f"{doc}|{dn}"

    def _add_dicts(self, dicts: list[dict], tier: str) -> None:
        for art in dicts:
            key = f"{_norm(art['law_id']).casefold()}|{_norm(art['dieu_number']).casefold()}"
            if key in self._seen_keys:
                self.stats["dedup_dropped"] += 1
                continue
            self._seen_keys.add(key)
            self._normalize_name(art)
            self._apply_citation_variant(art)
            if self.drop_khoan and art.get("khoan_list"):
                art["khoan_list"] = []
            self._articles.append(art)
            self.stats[tier]["articles"] += 1

    # ------------------------------------------------------------------
    # Build + write
    # ------------------------------------------------------------------

    def _ensure_backup(self) -> Path | None:
        """Back up the existing corpus.json → corpus_v1_1044.json BEFORE any
        overwrite, so the original hand-crawled corpus is preserved and can be
        unioned in as preferred tier 0. Never overwrites an existing backup."""
        out = self.cfg["output"]
        corpus_path = _HERE / out["corpus"]
        backup_path = _HERE / out["backup_v1"]
        if backup_path.exists():
            return backup_path
        if corpus_path.exists():
            backup_path.write_text(corpus_path.read_text(encoding="utf-8"),
                                   encoding="utf-8")
            log.info("Backed up existing corpus → %s", backup_path.name)
            return backup_path
        return None

    def _load_tier0(self) -> None:
        """Union the original corpus (preferred) so HF tiers only ADD new
        articles — guarantees no regression on the hand-crawled core laws,
        which also carry the gold-matching citation strings."""
        backup_path = _HERE / self.cfg["output"]["backup_v1"]
        if not backup_path.exists():
            log.warning("No tier-0 backup (%s) — skipping original-corpus union",
                        backup_path.name)
            return
        data = json.loads(backup_path.read_text(encoding="utf-8"))
        arts = data["articles"] if isinstance(data, dict) else data
        self._add_dicts(arts, "0")
        log.info("Tier 0 (original corpus): added %d articles (preferred)",
                 self.stats["0"]["articles"])

    def _build_title_map(self) -> None:
        """Map mã văn bản → official HF title (trích yếu). Tier B first, tier A
        overwrites (A preferred — cleaner Vietnamese metadata)."""
        if not self.normalize_names:
            return
        for tier in ("B", "A"):
            fld = self.cfg["fields"][tier]
            path = self.cache / self.cfg["tiers"][tier]["meta"]
            if not path.exists():
                continue
            tbl = pq.read_table(path, columns=[fld["doc_code"], fld["title"]])
            for r in tbl.to_pylist():
                code = _norm(r.get(fld["doc_code"]))
                title = _norm(r.get(fld["title"]))
                if code and title:
                    self._title_map[code.casefold()] = title
        log.info("Title map built: %d codes (for name normalisation)",
                 len(self._title_map))

    def build(self) -> None:
        self._ensure_backup()
        self._build_title_map()
        self._load_tier0()
        order = ["A", "B"] if self.prefer_tier == "A" else ["B", "A"]
        for tier in order:
            self.ingest_tier(tier)

    def write(self) -> None:
        out = self.cfg["output"]
        corpus_path = _HERE / out["corpus"]

        payload = {
            "version": "2.0",
            "total_articles": len(self._articles),
            "articles": self._articles,
        }
        corpus_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        log.info("Wrote %d articles → %s", len(self._articles), corpus_path)

        self._write_sources(corpus_path)

    def _write_sources(self, corpus_path: Path) -> None:
        src = self.cfg["source"]
        out = self.cfg["output"]
        sources_path = _HERE / out["sources_md"]
        s = self.stats
        lines = [
            "# Corpus Sources — TIP-CORPUS-001",
            "",
            f"- **Dataset:** `{src['repo']}` (HuggingFace)",
            f"- **License:** {src['license']}",
            f"- **Origin:** {src['origin']}",
            f"- **DOI:** {src.get('doi','—')}",
            f"- **Pinned revision:** `{self.cfg.get('dataset_revision', 'main')}`",
            f"- **Downloaded / built:** {date.today().isoformat()}",
            f"- **Citation name variant:** `{self.cfg['citation']['name_variant']}`",
            "",
            "## Article counts",
            "",
            "| Tier | meta rows | kept (filtered) | docs parsed | articles |",
            "|------|-----------|-----------------|-------------|----------|",
            f"| 0 (original, preferred) | — | — | — | {s['0']['articles']} |",
            f"| A (core, HTML)    | {s['A']['meta_rows']} | {s['A']['kept_meta']} | {s['A']['docs_parsed']} | {s['A']['articles']} |",
            f"| B (recall net)    | {s['B']['meta_rows']} | {s['B']['kept_meta']} | {s['B']['docs_parsed']} | {s['B']['articles']} |",
            "",
            f"- **Total articles (post-dedup):** {len(self._articles)}",
            f"- **Dedup dropped (tier preference 0 > A > B):** {s['dedup_dropped']}",
            f"- **Docs skipped (no content / scan-only PDF):** {s['scan_pdf_skipped']}",
            "",
            "## Doc-type mapping (raw → kept/dropped)",
            "",
            "| tier:raw_type | decision |",
            "|---------------|----------|",
        ]
        for k, v in sorted(s["type_mapping"].items()):
            lines.append(f"| {k} | {v} |")
        lines += [
            "",
            "## Filters applied",
            "",
            f"- **Effect keep:** A={self.cfg['filters']['effect_keep']['A']}, "
            f"B={self.cfg['filters']['effect_keep']['B']}",
            f"- **Doc-type keep (VI):** {self.cfg['filters']['doc_type_keep_vi']}",
            f"- **Domain filter applied:** {self.domain_apply}",
            f"- **Domain keywords:** {len(self.domain_kw)} terms "
            "(title + sectors substring match; see hf_config.json)",
        ]
        sources_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote provenance → %s", sources_path)

    def print_summary(self) -> None:
        s = self.stats
        print("\n" + "=" * 60)
        print("  HF INGEST SUMMARY")
        print("=" * 60)
        print(f"  Tier 0 (original union): articles={s['0']['articles']}")
        for tier in ("A", "B"):
            t = s[tier]
            print(f"  Tier {tier}: meta={t['meta_rows']} kept={t['kept_meta']} "
                  f"docs={t['docs_parsed']} articles={t['articles']}")
        print(f"  Dedup dropped: {s['dedup_dropped']}")
        print(f"  No-content skipped: {s['scan_pdf_skipped']}")
        print(f"  TOTAL ARTICLES: {len(self._articles)}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="TIP-CORPUS-001 two-tier corpus builder")
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--tier", choices=["A", "B"], default=None,
                    help="build only one tier (default: both)")
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="cap docs per tier (quick smoke build)")
    ap.add_argument("--no-write", action="store_true",
                    help="build in memory but do not write corpus.json")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.tier:
        for t in ("A", "B"):
            config["tiers"][t]["enabled"] = (t == args.tier)

    t0 = time.time()
    ing = HFIngest(config, limit_docs=args.limit_docs)
    ing.build()
    ing.print_summary()
    if not args.no_write:
        ing.write()
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
