"""
Master model catalog and registry for ekoi-llm-router.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings


@dataclass
class ModelEntry:
    id: str
    name: str
    type: str  # "llm" or "embedding"
    description: str
    family: str
    path: str
    quantization: Optional[str] = None
    dtype: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "family": self.family,
            "active": self.active,
        }


def _build_catalog() -> Dict[str, ModelEntry]:
    catalog: Dict[str, ModelEntry] = {}

    qwen_dir = settings.QWEN_MODELS_DIR
    llama_dir = settings.LLAMA_MODELS_DIR

    # ─────────────────────────────────────────────────────────────
    # Qwen 2.5 Models
    # ─────────────────────────────────────────────────────────────
    qwen_models = [
        # 3B
        ModelEntry(
            id="Qwen2.5-3B-Instruct_FP16_Baseline",
            name="Qwen 2.5 3B Instruct (FP16 Baseline)",
            type="llm",
            description="3B parameter instruct baseline in FP16/BF16",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-3B-Instruct_FP16_Baseline"),
            dtype="bfloat16",
        ),
        ModelEntry(
            id="Qwen2.5-3B-Instruct_FP8_Static_per_tensor",
            name="Qwen 2.5 3B Instruct (FP8 Static per-tensor)",
            type="llm",
            description="3B parameter instruct model with FP8 static quantization",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-3B-Instruct_FP8_Static_per_tensor"),
            quantization="fp8",
        ),
        ModelEntry(
            id="Qwen2.5-3B-Instruct_INT4_AWQ",
            name="Qwen 2.5 3B Instruct (INT4 AWQ)",
            type="llm",
            description="3B parameter instruct model with 4-bit AWQ quantization",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-3B-Instruct_INT4_AWQ"),
            quantization="awq",
        ),
        # 7B
        ModelEntry(
            id="Qwen2.5-7B-Instruct_FP16_Baseline",
            name="Qwen 2.5 7B Instruct (FP16 Baseline)",
            type="llm",
            description="7B parameter instruct baseline in FP16/BF16",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-7B-Instruct_FP16_Baseline"),
            dtype="bfloat16",
        ),
        ModelEntry(
            id="Qwen2.5-7B-Instruct_FP8_Static_per_tensor",
            name="Qwen 2.5 7B Instruct (FP8 Static per-tensor)",
            type="llm",
            description="7B parameter instruct model with FP8 static quantization",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-7B-Instruct_FP8_Static_per_tensor"),
            quantization="fp8",
        ),
        ModelEntry(
            id="Qwen2.5-7B-Instruct_INT4_AWQ",
            name="Qwen 2.5 7B Instruct (INT4 AWQ)",
            type="llm",
            description="7B parameter instruct model with 4-bit AWQ quantization (Default)",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-7B-Instruct_INT4_AWQ"),
            quantization="awq",
        ),
        # 14B
        ModelEntry(
            id="Qwen2.5-14B-Instruct_FP16_Baseline",
            name="Qwen 2.5 14B Instruct (FP16 Baseline)",
            type="llm",
            description="14B parameter instruct baseline in FP16/BF16",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-14B-Instruct_FP16_Baseline"),
            dtype="bfloat16",
        ),
        ModelEntry(
            id="Qwen2.5-14B-Instruct_FP8_Static_per_tensor",
            name="Qwen 2.5 14B Instruct (FP8 Static per-tensor)",
            type="llm",
            description="14B parameter instruct model with FP8 static quantization",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-14B-Instruct_FP8_Static_per_tensor"),
            quantization="fp8",
        ),
        ModelEntry(
            id="Qwen2.5-14B-Instruct_INT4_AWQ",
            name="Qwen 2.5 14B Instruct (INT4 AWQ)",
            type="llm",
            description="14B parameter instruct model with 4-bit AWQ quantization",
            family="qwen2.5",
            path=str(Path(qwen_dir) / "Qwen2.5-14B-Instruct_INT4_AWQ"),
            quantization="awq",
        ),
    ]

    for m in qwen_models:
        catalog[m.id] = m

    # ─────────────────────────────────────────────────────────────
    # Llama Models (Quantized local folders)
    # ─────────────────────────────────────────────────────────────
    llama_models = [
        # Llama-3.2 1B
        ModelEntry(
            id="meta-llama__Llama-3.2-1B-Instruct-AWQ",
            name="Llama 3.2 1B Instruct (AWQ)",
            type="llm",
            description="Llama-3.2 1B with AWQ 4-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-1B-Instruct-AWQ"),
            quantization="awq",
        ),
        ModelEntry(
            id="meta-llama__Llama-3.2-1B-Instruct-BNB-4bit",
            name="Llama 3.2 1B Instruct (BNB 4-bit)",
            type="llm",
            description="Llama-3.2 1B with BitsAndBytes 4-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-1B-Instruct-BNB-4bit"),
            quantization="bitsandbytes",
            extra_args=["--load-format", "bitsandbytes"],
        ),
        ModelEntry(
            id="meta-llama__Llama-3.2-1B-Instruct-BNB-8bit",
            name="Llama 3.2 1B Instruct (BNB 8-bit)",
            type="llm",
            description="Llama-3.2 1B with BitsAndBytes 8-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-1B-Instruct-BNB-8bit"),
            quantization="bitsandbytes",
            extra_args=["--load-format", "bitsandbytes"],
        ),
        ModelEntry(
            id="meta-llama__Llama-3.2-1B-Instruct-GPTQ-4bit",
            name="Llama 3.2 1B Instruct (GPTQ 4-bit)",
            type="llm",
            description="Llama-3.2 1B with GPTQ 4-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-1B-Instruct-GPTQ-4bit"),
            quantization="gptq",
        ),
        ModelEntry(
            id="meta-llama__Llama-3.2-1B-Instruct-GPTQ-8bit",
            name="Llama 3.2 1B Instruct (GPTQ 8-bit)",
            type="llm",
            description="Llama-3.2 1B with GPTQ 8-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-1B-Instruct-GPTQ-8bit"),
            quantization="gptq",
        ),
        # Llama-3.2 3B
        ModelEntry(
            id="meta-llama__Llama-3.2-3B-Instruct-AWQ",
            name="Llama 3.2 3B Instruct (AWQ)",
            type="llm",
            description="Llama-3.2 3B with AWQ 4-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-3B-Instruct-AWQ"),
            quantization="awq",
        ),
        ModelEntry(
            id="meta-llama__Llama-3.2-3B-Instruct-BNB-4bit",
            name="Llama 3.2 3B Instruct (BNB 4-bit)",
            type="llm",
            description="Llama-3.2 3B with BitsAndBytes 4-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-3B-Instruct-BNB-4bit"),
            quantization="bitsandbytes",
            extra_args=["--load-format", "bitsandbytes"],
        ),
        ModelEntry(
            id="meta-llama__Llama-3.2-3B-Instruct-BNB-8bit",
            name="Llama 3.2 3B Instruct (BNB 8-bit)",
            type="llm",
            description="Llama-3.2 3B with BitsAndBytes 8-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-3B-Instruct-BNB-8bit"),
            quantization="bitsandbytes",
            extra_args=["--load-format", "bitsandbytes"],
        ),
        ModelEntry(
            id="meta-llama__Llama-3.2-3B-Instruct-GPTQ-4bit",
            name="Llama 3.2 3B Instruct (GPTQ 4-bit)",
            type="llm",
            description="Llama-3.2 3B with GPTQ 4-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-3B-Instruct-GPTQ-4bit"),
            quantization="gptq",
        ),
        ModelEntry(
            id="meta-llama__Llama-3.2-3B-Instruct-GPTQ-8bit",
            name="Llama 3.2 3B Instruct (GPTQ 8-bit)",
            type="llm",
            description="Llama-3.2 3B with GPTQ 8-bit quantization",
            family="llama-3.2",
            path=str(Path(llama_dir) / "meta-llama__Llama-3.2-3B-Instruct-GPTQ-8bit"),
            quantization="gptq",
        ),
        # Meta-Llama-3 8B
        ModelEntry(
            id="meta-llama__Meta-Llama-3-8B-Instruct-AWQ",
            name="Meta Llama 3 8B Instruct (AWQ)",
            type="llm",
            description="Meta-Llama-3 8B with AWQ 4-bit quantization",
            family="llama-3",
            path=str(Path(llama_dir) / "meta-llama__Meta-Llama-3-8B-Instruct-AWQ"),
            quantization="awq",
        ),
        ModelEntry(
            id="meta-llama__Meta-Llama-3-8B-Instruct-BNB-4bit",
            name="Meta Llama 3 8B Instruct (BNB 4-bit)",
            type="llm",
            description="Meta-Llama-3 8B with BitsAndBytes 4-bit quantization",
            family="llama-3",
            path=str(Path(llama_dir) / "meta-llama__Meta-Llama-3-8B-Instruct-BNB-4bit"),
            quantization="bitsandbytes",
            extra_args=["--load-format", "bitsandbytes"],
        ),
        ModelEntry(
            id="meta-llama__Meta-Llama-3-8B-Instruct-BNB-8bit",
            name="Meta Llama 3 8B Instruct (BNB 8-bit)",
            type="llm",
            description="Meta-Llama-3 8B with BitsAndBytes 8-bit quantization",
            family="llama-3",
            path=str(Path(llama_dir) / "meta-llama__Meta-Llama-3-8B-Instruct-BNB-8bit"),
            quantization="bitsandbytes",
            extra_args=["--load-format", "bitsandbytes"],
        ),
        ModelEntry(
            id="meta-llama__Meta-Llama-3-8B-Instruct-GPTQ-4bit",
            name="Meta Llama 3 8B Instruct (GPTQ 4-bit)",
            type="llm",
            description="Meta-Llama-3 8B with GPTQ 4-bit quantization",
            family="llama-3",
            path=str(Path(llama_dir) / "meta-llama__Meta-Llama-3-8B-Instruct-GPTQ-4bit"),
            quantization="gptq",
        ),
        ModelEntry(
            id="meta-llama__Meta-Llama-3-8B-Instruct-GPTQ-8bit",
            name="Meta Llama 3 8B Instruct (GPTQ 8-bit)",
            type="llm",
            description="Meta-Llama-3 8B with GPTQ 8-bit quantization",
            family="llama-3",
            path=str(Path(llama_dir) / "meta-llama__Meta-Llama-3-8B-Instruct-GPTQ-8bit"),
            quantization="gptq",
        ),
    ]

    for m in llama_models:
        catalog[m.id] = m

    # ─────────────────────────────────────────────────────────────
    # HuggingFace standard download directory models
    # ─────────────────────────────────────────────────────────────
    hf_models = [
        ModelEntry(
            id="meta-llama/Llama-3.2-1B-Instruct",
            name="Llama 3.2 1B Instruct (HuggingFace)",
            type="llm",
            description="Standard HuggingFace downloaded Llama-3.2 1B Instruct",
            family="llama-3.2",
            path="meta-llama/Llama-3.2-1B-Instruct",
        ),
        ModelEntry(
            id="meta-llama/Llama-3.2-3B-Instruct",
            name="Llama 3.2 3B Instruct (HuggingFace)",
            type="llm",
            description="Standard HuggingFace downloaded Llama-3.2 3B Instruct",
            family="llama-3.2",
            path="meta-llama/Llama-3.2-3B-Instruct",
        ),
        ModelEntry(
            id="meta-llama/Meta-Llama-3-8B-Instruct",
            name="Meta Llama 3 8B Instruct (HuggingFace)",
            type="llm",
            description="Standard HuggingFace downloaded Meta-Llama-3 8B Instruct",
            family="llama-3",
            path="meta-llama/Meta-Llama-3-8B-Instruct",
        ),
    ]

    for m in hf_models:
        catalog[m.id] = m

    # ─────────────────────────────────────────────────────────────
    # Embedding Models (Served concurrently on CPU)
    # ─────────────────────────────────────────────────────────────
    embed_models = [
        ModelEntry(
            id="nomic-ai/nomic-embed-text-v1.5",
            name="Nomic Embed Text v1.5",
            type="embedding",
            description="High-performance embedding model running locally on CPU via FastEmbed",
            family="nomic",
            path="nomic-ai/nomic-embed-text-v1.5",
        ),
    ]

    for m in embed_models:
        catalog[m.id] = m

    return catalog


CATALOG: Dict[str, ModelEntry] = _build_catalog()


def get_model(model_id: str) -> Optional[ModelEntry]:
    """Retrieve model entry by exact identifier."""
    if not model_id:
        return None
    return CATALOG.get(model_id.strip())


def list_models(model_type: Optional[str] = None) -> List[ModelEntry]:
    """List all registered models, optionally filtered by type ('llm' or 'embedding')."""
    models = list(CATALOG.values())
    if model_type:
        models = [m for m in models if m.type == model_type]
    return models


def resolve_vllm_command(model_entry: ModelEntry) -> List[str]:
    """Generate CLI command arguments for launching vLLM."""
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_entry.path,
        "--served-model-name",
        model_entry.id,
        "--host",
        settings.VLLM_INTERNAL_HOST,
        "--port",
        str(settings.VLLM_INTERNAL_PORT),
        "--gpu-memory-utilization",
        str(settings.GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(settings.MAX_MODEL_LEN),
        "--trust-remote-code",
    ]

    if model_entry.quantization:
        cmd.extend(["--quantization", model_entry.quantization])

    if model_entry.dtype:
        cmd.extend(["--dtype", model_entry.dtype])

    if model_entry.extra_args:
        cmd.extend(model_entry.extra_args)

    return cmd
