"""Qdrant client construction shared by indexing and retrieval stages."""

from __future__ import annotations

from typing import Any


def create_qdrant_client(config: Any, *, timeout: int = 120):
    from qdrant_client import QdrantClient

    qdrant = config.qdrant
    if qdrant.url:
        return QdrantClient(
            url=qdrant.url,
            api_key=qdrant.api_key or None,
            timeout=timeout,
        )
    return QdrantClient(host=qdrant.host, port=qdrant.port, timeout=timeout)
