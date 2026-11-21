"""
Chat endpoints supporting both Ollama (/chat, /api/chat) and OpenAI (/v1/chat/completions) schemas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from models_catalog import get_default_model, get_model
from scheduler import scheduler
from vllm_controller import vllm_controller

logger = logging.getLogger(__name__)

router = APIRouter()


class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OllamaChatMessage] = Field(default_factory=list)
    stream: bool = False
    options: Optional[Dict[str, Any]] = None


def _build_openai_payload_from_ollama(req: OllamaChatRequest, target_model: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": target_model,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
    }
    if req.options:
        if "temperature" in req.options:
            payload["temperature"] = float(req.options["temperature"])
        if "top_p" in req.options:
            payload["top_p"] = float(req.options["top_p"])
        if "num_predict" in req.options:
            payload["max_tokens"] = int(req.options["num_predict"])
    return payload


async def _mock_stream_ollama(model: str) -> AsyncGenerator[bytes, None]:
    words = ["This", " is", " a", " mock", " response", " from", " ekoi-llm-router."]
    for word in words:
        chunk = {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": word},
            "done": False,
        }
        yield (json.dumps(chunk) + "\n").encode("utf-8")
        await asyncio.sleep(0.05)
    done_chunk = {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }
    yield (json.dumps(done_chunk) + "\n").encode("utf-8")


async def _stream_vllm_to_ollama(
    client: httpx.AsyncClient, response: httpx.Response, target_model: str
) -> AsyncGenerator[bytes, None]:
    try:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    done_chunk = {
                        "model": target_model,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                    }
                    yield (json.dumps(done_chunk) + "\n").encode("utf-8")
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content_piece = delta.get("content", "")
                    if content_piece:
                        ollama_chunk = {
                            "model": target_model,
                            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "message": {"role": "assistant", "content": content_piece},
                            "done": False,
                        }
                        yield (json.dumps(ollama_chunk) + "\n").encode("utf-8")
                except json.JSONDecodeError:
                    pass
    finally:
        await response.aclose()


# ─────────────────────────────────────────────────────────────
# Ollama compatible endpoints: /chat and /api/chat
# ─────────────────────────────────────────────────────────────
@router.post("/chat")
@router.post("/api/chat")
async def chat_ollama(request: OllamaChatRequest):
    default_entry = get_default_model("llm")
    fallback_id = default_entry.id if default_entry else ""
    target_model = (request.model or fallback_id).strip()
    model_entry = get_model(target_model)
    if not model_entry or model_entry.type != "llm":
        raise HTTPException(
            status_code=404,
            detail=f"Model '{target_model}' not found or is not a text generation LLM.",
        )

    # Acquire sequential slot (will wait in queue if another model is loaded)
    await scheduler.acquire_slot(target_model)

    if settings.MOCK_MODE:
        if request.stream:
            async def _cleanup_mock_stream():
                try:
                    async for chunk in _mock_stream_ollama(target_model):
                        yield chunk
                finally:
                    await scheduler.release_slot(target_model)

            return StreamingResponse(_cleanup_mock_stream(), media_type="application/x-ndjson")

        try:
            return JSONResponse(
                {
                    "model": target_model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {
                        "role": "assistant",
                        "content": f"[MOCK] Response from {target_model} generated by ekoi-llm-router.",
                    },
                    "done": True,
                }
            )
        finally:
            await scheduler.release_slot(target_model)

    # Forward to internal vLLM instance
    openai_payload = _build_openai_payload_from_ollama(request, target_model)
    vllm_url = f"{vllm_controller.base_url}/v1/chat/completions"

    if request.stream:
        try:
            client = httpx.AsyncClient(timeout=None)
            req = client.build_request("POST", vllm_url, json=openai_payload)
            resp = await client.send(req, stream=True)
            if resp.status_code != 200:
                await resp.aclose()
                await client.aclose()
                await scheduler.release_slot(target_model)
                raise HTTPException(status_code=resp.status_code, detail="vLLM streaming error")

            async def _cleanup_stream():
                try:
                    async for chunk in _stream_vllm_to_ollama(client, resp, target_model):
                        yield chunk
                finally:
                    await client.aclose()
                    await scheduler.release_slot(target_model)

            return StreamingResponse(_cleanup_stream(), media_type="application/x-ndjson")
        except Exception:
            await scheduler.release_slot(target_model)
            raise

    # Non-streaming request: guaranteed slot release via finally block
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(vllm_url, json=openai_payload)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"vLLM error: {resp.text}")

            vllm_json = resp.json()
            assistant_content = ""
            choices = vllm_json.get("choices", [])
            if choices:
                assistant_content = choices[0].get("message", {}).get("content", "")

            return JSONResponse(
                {
                    "model": target_model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                    "done": True,
                }
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat request failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await scheduler.release_slot(target_model)


# ─────────────────────────────────────────────────────────────
# OpenAI compatible endpoint: /v1/chat/completions
# ─────────────────────────────────────────────────────────────
@router.post("/v1/chat/completions")
async def chat_openai(request: Request):
    body = await request.json()
    default_entry = get_default_model("llm")
    fallback_id = default_entry.id if default_entry else ""
    target_model = (body.get("model") or fallback_id).strip()
    is_stream = bool(body.get("stream", False))

    model_entry = get_model(target_model)
    if not model_entry or model_entry.type != "llm":
        raise HTTPException(
            status_code=404,
            detail=f"Model '{target_model}' not found or is not a text generation LLM.",
        )

    # Acquire sequential slot (will wait in queue if another model is loaded)
    await scheduler.acquire_slot(target_model)

    if settings.MOCK_MODE:
        if is_stream:
            async def _mock_openai_sse():
                try:
                    chunk_id = f"chatcmpl-mock-{int(time.time())}"
                    words = ["Mock", " OpenAI", " stream", " output."]
                    for word in words:
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": target_model,
                            "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"
                finally:
                    await scheduler.release_slot(target_model)

            return StreamingResponse(_mock_openai_sse(), media_type="text/event-stream")

        try:
            return JSONResponse(
                {
                    "id": f"chatcmpl-mock-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": target_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"[MOCK] Response from {target_model}",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        finally:
            await scheduler.release_slot(target_model)

    vllm_url = f"{vllm_controller.base_url}/v1/chat/completions"

    if is_stream:
        try:
            client = httpx.AsyncClient(timeout=None)
            req = client.build_request("POST", vllm_url, json=body)
            resp = await client.send(req, stream=True)
            if resp.status_code != 200:
                await resp.aclose()
                await client.aclose()
                await scheduler.release_slot(target_model)
                raise HTTPException(status_code=resp.status_code, detail="vLLM stream error")

            async def _stream_openai():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
                    await client.aclose()
                    await scheduler.release_slot(target_model)

            return StreamingResponse(_stream_openai(), media_type="text/event-stream")
        except Exception:
            await scheduler.release_slot(target_model)
            raise

    # Non-streaming request: guaranteed slot release via finally block
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(vllm_url, json=body)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return JSONResponse(resp.json())
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat completions failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await scheduler.release_slot(target_model)
