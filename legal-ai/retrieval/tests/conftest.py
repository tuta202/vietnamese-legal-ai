"""Add legal-ai/ root to sys.path so `from retrieval.*` imports resolve."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
