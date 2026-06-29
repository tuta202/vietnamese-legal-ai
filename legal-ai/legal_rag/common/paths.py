"""Stable project paths; runtime artifacts remain outside the source package."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DATA_DIR = PROJECT_ROOT / "corpus" / "data"
INDEX_DATA_DIR = PROJECT_ROOT / "retrieval" / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
