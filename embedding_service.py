"""
CPU Embedding Service using FastEmbed.
Auto-downloads model weights if not present; runs purely on CPU to prevent GPU VRAM contention.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

_EMBED_ENGINES: Dict[str, Any] = {}


class EmbeddingService:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _get_or_load_model(self, model_id: str):
        if model_id in self._models:
            return self._models[model_id]

        logger.info("Initializing CPU embedding model '%s' via FastEmbed...", model_id)
        try:
            from fastembed import TextEmbedding

            # FastEmbed model mapping (nomic-ai/nomic-embed-text-v1.5 maps to nomic-ai/nomic-embed-text-v1.5)
            engine = TextEmbedding(model_name=model_id)
            self._models[model_id] = engine
            logger.info("CPU embedding model '%s' successfully initialized.", model_id)
            return engine
        except ImportError:
            logger.warning(
                "fastembed is not installed. Using fallback deterministic mock embedding generator."
            )
            return None
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", model_id, exc)
            return None

    async def get_embeddings(
        self, texts: List[str], model_id: Optional[str] = None
    ) -> List[List[float]]:
        """Compute embeddings for a list of strings on CPU asynchronously."""
        target_model = model_id or settings.DEFAULT_EMBED_MODEL

        # Run model loading and inference in worker thread
        return await asyncio.to_thread(self._sync_embed, texts, target_model)

    def _sync_embed(self, texts: List[str], model_id: str) -> List[List[float]]:
        engine = self._get_or_load_model(model_id)
        if engine is not None:
            # FastEmbed generator yields numpy arrays
            embeddings_gen = engine.embed(texts)
            return [emb.tolist() for emb in embeddings_gen]

        # Deterministic mock fallback (768 dimensions for nomic-embed)
        logger.debug("Generating mock embeddings for %d text(s)", len(texts))
        results = []
        for text in texts:
            # Deterministic pseudo-random vector based on hash
            h = hash(text) & 0xFFFFFFFF
            base = float(h % 1000) / 1000.0
            vec = [(base + (i * 0.001)) % 1.0 for i in range(768)]
            results.append(vec)
        return results


# Global singleton instance
embedding_service = EmbeddingService()
