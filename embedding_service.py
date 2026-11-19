"""
Dedicated CPU Embedding Service.
Runs the platform's single embedding model (nomic-ai/nomic-embed-text-v1.5) on CPU via FastEmbed.
Auto-downloads weights on startup if not present; requires 0 GPU VRAM.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from config import settings

logger = logging.getLogger(__name__)


class DedicatedEmbeddingEngine:
    def __init__(self, model_id: str = settings.EMBEDDING_MODEL_ID):
        self.model_id = model_id
        self._engine: Any = None
        self._lock = asyncio.Lock()

    def warmup(self) -> None:
        """Initialize and pull embedding model on startup if not already cached."""
        if self._engine is not None:
            return

        logger.info("Initializing dedicated CPU embedding model '%s' via FastEmbed...", self.model_id)
        try:
            from fastembed import TextEmbedding

            self._engine = TextEmbedding(model_name=self.model_id)
            logger.info("Embedding model '%s' ready.", self.model_id)
        except ImportError:
            logger.warning("fastembed not installed. Falling back to mock embeddings for local development.")
            self._engine = None
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self.model_id, exc)
            self._engine = None

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of strings on CPU asynchronously."""
        return await asyncio.to_thread(self._sync_embed, texts)

    def _sync_embed(self, texts: List[str]) -> List[List[float]]:
        if self._engine is None:
            self.warmup()

        if self._engine is not None:
            embeddings_gen = self._engine.embed(texts)
            return [emb.tolist() for emb in embeddings_gen]

        # Deterministic mock fallback (768 dimensions for nomic-embed)
        results = []
        for text in texts:
            h = hash(text) & 0xFFFFFFFF
            base = float(h % 1000) / 1000.0
            vec = [(base + (i * 0.001)) % 1.0 for i in range(768)]
            results.append(vec)
        return results


# Global singleton instance
embedding_service = DedicatedEmbeddingEngine()
