"""
Main entry point for ekoi-llm-router.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_chat import router as chat_router
from api.routes_embed import router as embed_router
from api.routes_models import router as models_router
from config import settings
from embedding_service import embedding_service
from models_catalog import get_default_model, list_models
from vllm_controller import vllm_controller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ekoi-llm-router")


@asynccontextmanager
async def lifespan(app: FastAPI):
    def_llm = get_default_model("llm")
    logger.info("==================================================")
    logger.info("   Starting ekoi-llm-router (port %d)", settings.PORT)
    logger.info("   Default LLM: %s", def_llm.id if def_llm else "None")
    logger.info("   Single Dedicated Embedder: %s", settings.EMBEDDING_MODEL_ID)
    logger.info("   Mock Mode: %s", settings.MOCK_MODE)
    logger.info("==================================================")

    # Warmup and pull single dedicated embedding model on CPU
    embedding_service.warmup()

    # Log discovered LLMs
    llm_models = list_models("llm")
    logger.info("Loaded %d LLM(s) in catalog from models.json.", len(llm_models))

    yield

    logger.info("Shutting down ekoi-llm-router...")
    await vllm_controller.stop()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ekoi-llm-router",
        description="Sequential vLLM Model Orchestrator with Ollama & OpenAI API compatibility",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(chat_router)
    app.include_router(embed_router)
    app.include_router(models_router)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
