"""
Model catalog & health routes.
Exposes configurable model listing with customizable field selection,
Ollama tags, OpenAI models, and GPU health stats.
"""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from models_catalog import get_default_model, list_models, registry
from scheduler import scheduler
from vllm_controller import get_gpu_memory_info, vllm_controller

router = APIRouter()


@router.get("/api/models")
@router.get("/models")
async def get_models(
    fields: Optional[str] = Query(
        None,
        description="Comma-separated list of fields to include (e.g. 'id,name' or 'id,name,size,quantization,good_for')",
    ),
    type: Optional[str] = Query(None, description="Filter by 'llm' or 'embedding'"),
    include_disabled: bool = Query(False, description="Whether to include disabled models"),
):
    """
    Central model endpoint. Returns all registered models from models.json.
    Supports dynamic projection via 'fields' parameter and type filtering.
    Internal fields (pull, enabled, path) are excluded unless explicitly requested.
    """
    field_list: Optional[List[str]] = None
    if fields:
        field_list = [f.strip() for f in fields.split(",") if f.strip()]

    models = list_models(
        model_type=type,
        include_disabled=include_disabled,
        fields=field_list,
    )
    return JSONResponse({"models": models})


@router.get("/api/models/default")
async def get_default(
    type: str = Query("llm", description="Model type: 'llm' or 'embedding'"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include"),
):
    """Return the designated default model for the requested type."""
    model_entry = get_default_model(type)
    if not model_entry:
        raise HTTPException(status_code=404, detail=f"No default model configured for type '{type}'")

    field_list: Optional[List[str]] = None
    if fields:
        field_list = [f.strip() for f in fields.split(",") if f.strip()]

    return JSONResponse({"model": model_entry.to_dict(fields=field_list)})


@router.get("/api/tags")
async def ollama_tags():
    """
    Ollama-compatible /api/tags endpoint.
    Used by clients expecting Ollama response format.
    """
    all_entries = registry.get_all_entries()
    models_list = []
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for m in all_entries:
        if not m.enabled:
            continue
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
                    "family": m.type,
                    "parameter_size": m.size,
                    "quantization_level": m.quantization,
                },
            }
        )

    return JSONResponse({"models": models_list})


@router.get("/v1/models")
async def openai_models():
    """OpenAI compatible /v1/models endpoint."""
    all_entries = registry.get_all_entries()
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
        for m in all_entries
        if m.enabled
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


@router.post("/api/models/unload")
@router.post("/unload")
async def unload_model():
    """Immediately unload current active model and release GPU VRAM."""
    current = scheduler.active_model
    if not current and not vllm_controller.is_running:
        return JSONResponse({
            "status": "no_active_model",
            "message": "No model currently loaded in GPU VRAM.",
            "gpu": get_gpu_memory_info(),
        })

    await vllm_controller.stop()
    return JSONResponse({
        "status": "unloaded",
        "unloaded_model": current,
        "gpu": get_gpu_memory_info(),
    })
