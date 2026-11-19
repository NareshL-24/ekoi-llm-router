"""
Embedding endpoints supporting both Ollama (/api/embeddings) and OpenAI (/v1/embeddings) schemas.
Runs CPU-based FastEmbed models with zero GPU VRAM impact.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from embedding_service import embedding_service
from models_catalog import get_model

logger = logging.getLogger(__name__)

router = APIRouter()


class OllamaEmbeddingRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]


class OpenAIEmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]


@router.post("/api/embeddings")
async def embed_ollama(request: OllamaEmbeddingRequest):
    """Ollama compatible embedding endpoint."""
    model_id = (request.model or settings.DEFAULT_EMBED_MODEL).strip()
    
    # Text or list of texts
    if isinstance(request.prompt, list):
        texts = request.prompt
    else:
        texts = [request.prompt]

    try:
        embeddings = await embedding_service.get_embeddings(texts, model_id)
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

        # Ollama expects single "embedding" if prompt was string
        if isinstance(request.prompt, str):
            return JSONResponse({"embedding": embeddings[0]})
        return JSONResponse({"embeddings": embeddings})

    except Exception as exc:
        logger.error("Embedding generation failed for '%s': %s", model_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/v1/embeddings")
async def embed_openai(request: OpenAIEmbeddingRequest):
    """OpenAI compatible embedding endpoint."""
    model_id = (request.model or settings.DEFAULT_EMBED_MODEL).strip()

    if isinstance(request.input, list):
        texts = request.input
    else:
        texts = [request.input]

    try:
        embeddings = await embedding_service.get_embeddings(texts, model_id)
        data = [
            {
                "object": "embedding",
                "embedding": emb,
                "index": i,
            }
            for i, emb in enumerate(embeddings)
        ]

        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "model": model_id,
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in texts),
                    "total_tokens": sum(len(t.split()) for t in texts),
                },
            }
        )
    except Exception as exc:
        logger.error("OpenAI embedding generation failed for '%s': %s", model_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
