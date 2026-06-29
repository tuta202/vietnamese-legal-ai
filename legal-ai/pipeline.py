"""Backward-compatible imports for the online pipeline API."""

from legal_rag.pipeline import LegalAIPipeline, PipelineState, main

__all__ = ["LegalAIPipeline", "PipelineState", "main"]


if __name__ == "__main__":
    main()
