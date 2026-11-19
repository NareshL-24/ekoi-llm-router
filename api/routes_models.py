"""
Model catalog & health routes.
Exposes Ollama tags, OpenAI models, router catalog, and GPU health stats.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from models_catalog import list_models
from scheduler import scheduler
from vllm_controller import get_gpu_memory_info

router = APIRouter()


@router.get("/api/tags")
async def ollama_tags():
    """
    Ollama-compatible /api/tags endpoint.
    Used by health-check and model discovery clients expecting Ollama response format.
    """
    all_models = list_models()
    models_list = []
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for m in all_models:
        models_list.append(
            {
                "name": m.id,
                "model": m.id,
                "modified_at": now_iso,
                "size": 0,
                "digest": m.id,
                "details": {
                    "parent_model": "",
                    "format": "safetensors",
                    "family": m.family,
                    "parameter_size": "",
                    "quantization_level": m.quantization or "none",
                },
            }
        )

    return JSONResponse({"models": models_list})


@router.get("/api/models")
@router.get("/models")
async def router_models(model_type: Optional[str] = Query(None, description="Filter by 'llm' or 'embedding'")):
    """
    Master model registry endpoint queried dynamically by platform-services.
    """
    models = list_models(model_type=model_type)
    return JSONResponse(
        {
            "models": [m.to_dict() for m in models]
        }
    )


@router.get("/v1/models")
async def openai_models():
    """OpenAI compatible /v1/models endpoint."""
    all_models = list_models()
    data = [
        {
            "id": m.id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ekoi-llm-router",
            "permission": [],
            "root": m.id,
            "parent": None,
        }
        for m in all_models
    ]
    return JSONResponse({"object": "list", "data": data})


@router.get("/health")
async def health_check():
    """Router health check and GPU telemetry."""
    return JSONResponse(
        {
            "status": "healthy",
            "scheduler": scheduler.get_status(),
            "gpu": get_gpu_memory_info(),
        }
    )
