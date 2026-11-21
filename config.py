"""
Configuration settings for ekoi-llm-router.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
    MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    DISABLE_FLASHINFER_AUTOTUNE: bool = os.getenv("VLLM_DISABLE_FLASHINFER_AUTOTUNE", "true").lower() in ("true", "1", "yes")
    VLLM_ENFORCE_EAGER: bool = os.getenv("VLLM_ENFORCE_EAGER", "false").lower() in ("true", "1", "yes")

    # Dedicated single embedding model on CPU
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "nomic-ai/nomic-embed-text-v1.5")

    # Scheduler policies
    # 0 = keep loaded permanently (no unload). Default: 300s (5 minutes)
    IDLE_UNLOAD_TIMEOUT: int = int(os.getenv("IDLE_UNLOAD_TIMEOUT", "300"))
    
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
