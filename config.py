"""
Configuration settings for ekoi-llm-router.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Settings:
    # Router public server
    HOST: str = os.getenv("ROUTER_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("ROUTER_PORT", "8080"))

    # Internal vLLM server
    VLLM_INTERNAL_HOST: str = os.getenv("VLLM_INTERNAL_HOST", "127.0.0.1")
    VLLM_INTERNAL_PORT: int = int(os.getenv("VLLM_INTERNAL_PORT", "8001"))
    VLLM_STARTUP_TIMEOUT: int = int(os.getenv("VLLM_STARTUP_TIMEOUT", "180"))
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95"))
    MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))

    # Server model directories
    QWEN_MODELS_DIR: str = os.getenv(
        "QWEN_MODELS_DIR", "/home/ror-technologies/Quantization/models"
    )
    LLAMA_MODELS_DIR: str = os.getenv(
        "LLAMA_MODELS_DIR", "/home/ror-technologies/IT-helpdesk-agent/quantized_models"
    )
    HF_CACHE_DIR: str = os.getenv("HF_HOME", "")

    # Default models
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "Qwen2.5-7B-Instruct_INT4_AWQ")
    DEFAULT_EMBED_MODEL: str = os.getenv(
        "DEFAULT_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"
    )

    # Scheduler policies
    IDLE_UNLOAD_TIMEOUT: int = int(os.getenv("IDLE_UNLOAD_TIMEOUT", "300"))  # 5 minutes
    
    # Mock / Emulation mode (for non-GPU local testing without spawning vLLM)
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")

    # CORS
    CORS_ORIGINS: List[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]
    )

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.VLLM_INTERNAL_HOST}:{self.VLLM_INTERNAL_PORT}"


settings = Settings()
