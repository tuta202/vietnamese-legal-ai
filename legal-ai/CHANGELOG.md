# Changelog

## TIP-MEASURE-001 — 2026-06-05

Bundle cleanup + re-embed prep for the public-leaderboard F2 measurement.

### Reproducibility
- HF dataset **`th1nhng0/vietnamese-legal-documents`** pinned to revision
  **`0a39ad7eae8e6c188cb225c4b1443c3b346461d8`** (as of 2026-06-05).
  Downloaders (`corpus/hf_download.py`, `corpus/hf_fetch_large.py`) read this
  from `corpus/hf_config.json` → `dataset_revision`.
  **Verify:** a clean rebuild (`hf_download.py` + `hf_fetch_large.py` + `hf_ingest.py`)
  must reproduce exactly **113,508 articles**.

### Changes
- Pinned dataset revision in config + both downloaders.
- `corpus.json` rebuilt with `drop_khoan_list=true` → **436 MB** (was 780 MB),
  still 113,508 articles, schema unchanged (`khoan_list` emptied — it duplicated
  `content` and is unused downstream).
- Deleted deprecated eval data `eval/data/eval_set_50.json`, `eval_set_100.json`
  (recoverable from git `ecae399`). `eval/synthetic_qa.py` + `eval/evaluator.py`
  kept for code reference. F2 now measured via the public leaderboard.

### KNOWN BUGS in the deprecated eval data (`synthetic_qa.py` lineage)
> Not used post-TIP-MEASURE-001 — recorded for reference only.
- `86/2015/QH13` mislabeled **"Luật KH&CN"** — correct is **"Luật An toàn thông tin mạng"**
  (the expanded corpus carries the correct name).
- `45/2020/QH14` — typo for `45/2019/QH14` (**Bộ luật Lao động**); `45/2020/QH14`
  does not exist in the source dataset.

`synthetic_qa.py` is kept for code reference only; its generated eval data is
deprecated in favour of leaderboard F2.
