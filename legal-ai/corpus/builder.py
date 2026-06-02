"""
CorpusBuilder — reads crawler_config.json + raw .txt files,
parses them into Article objects, and writes corpus.json.

CLI:
    python corpus/builder.py
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Allow running as 'python corpus/builder.py' from project root
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from parser import Article, LegalTextParser

CONFIG_PATH = _ROOT / "crawler" / "crawler_config.json"
RAW_DIR = _ROOT / "corpus" / "data" / "raw"
OUTPUT_PATH = _ROOT / "corpus" / "data" / "corpus.json"

# Expected Điều counts from TIP-001B (raw regex, includes cross-refs)
_TIP001B_COUNTS = {
    "59/2020/QH14":   712,
    "04/2017/QH14":    34,
    "38/2019/QH14":   416,
    "13/2008/QH12":    48,
    "14/2008/QH12":    48,
    "45/2019/QH14":   668,
    "61/2020/QH14":   346,
    "01/2021/NĐ-CP":  414,
    "80/2021/NĐ-CP":  134,
    "123/2020/NĐ-CP": 218,
    "12/2022/NĐ-CP":  446,
    "78/2014/TT-BTC":  96,
    "219/2013/TT-BTC": 134,
}


class CorpusBuilder:

    def __init__(self):
        self._parser = LegalTextParser()
        self._config: list[dict] = []
        self._articles: list[Article] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_config(self) -> list[dict]:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self._config = data["laws"]
        return self._config

    def build_from_raw(self) -> list[Article]:
        if not self._config:
            self.load_config()

        self._articles = []
        for entry in self._config:
            safe_id = entry["id"].replace("/", "_").replace(" ", "_")
            raw_file = RAW_DIR / f"{safe_id}.txt"

            if not raw_file.exists():
                print(f"  [WARN] Missing raw file: {raw_file.name}", flush=True)
                continue

            raw_text = raw_file.read_text(encoding="utf-8")
            parsed = self._parser.parse(raw_text, entry)
            self._articles.extend(parsed)

            print(
                f"  {entry['id']:<22}  parsed={len(parsed):>4}  "
                f"({raw_file.stat().st_size // 1024}KB)",
                flush=True,
            )

        return self._articles

    def save(self, path: Path | str = OUTPUT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0",
            "total_articles": len(self._articles),
            "articles": [a.to_dict() for a in self._articles],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved {len(self._articles)} articles → {path}", flush=True)

    @staticmethod
    def load(path: Path | str = OUTPUT_PATH) -> list[dict]:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["articles"]

    def stats(self) -> None:
        """Print per-law article counts vs TIP-001B expected counts."""
        if not self._articles:
            print("No articles loaded. Run build_from_raw() first.")
            return

        # Group by law_id
        counts: dict[str, int] = {}
        for a in self._articles:
            counts[a.law_id] = counts.get(a.law_id, 0) + 1

        print()
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║  CORPUS STATS — TIP-002                                       ║")
        print("╠══════════════════╦═══════════╦══════════╦══════════╦══════════╣")
        print("║ law_id           ║ parsed    ║ tip001b  ║ diff     ║ law_name ║")
        print("╠══════════════════╬═══════════╬══════════╬══════════╬══════════╣")

        total_parsed = 0
        for entry in self._config:
            lid = entry["id"]
            parsed = counts.get(lid, 0)
            expected = _TIP001B_COUNTS.get(lid, "?")
            diff = (parsed - expected) if isinstance(expected, int) else "?"
            total_parsed += parsed
            name_short = entry["name"][:18]
            print(
                f"║ {lid:<16} ║ {parsed:>9} ║ {str(expected):>8} "
                f"║ {str(diff):>8} ║ {name_short:<8} ║"
            )

        print("╠══════════════════╬═══════════╬══════════╬══════════╬══════════╣")
        print(f"║ {'TOTAL':<16} ║ {total_parsed:>9} ║ {'3814':>8} ║ {'':>8} ║ {'':>8} ║")
        print("╚══════════════════╩═══════════╩══════════╩══════════╩══════════╝")
        print()
        print("Note: TIP-001B counts include cross-references (raw 'Điều \\d+').")
        print("      Parser counts = actual article headers only (monotonic filter).")

        # Dedup check
        chunk_ids = [a.chunk_id for a in self._articles]
        dupes = len(chunk_ids) - len(set(chunk_ids))
        print(f"\nchunk_id unique: {len(set(chunk_ids))}  duplicates: {dupes}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  CorpusBuilder — TIP-002")
    print("=" * 60)
    t0 = time.time()

    builder = CorpusBuilder()
    builder.load_config()

    print(f"\nBuilding from {RAW_DIR} ...")
    builder.build_from_raw()

    builder.save(OUTPUT_PATH)
    builder.stats()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
