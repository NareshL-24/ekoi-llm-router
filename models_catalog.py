"""
Master model catalog loader for ekoi-llm-router.
Loads models strictly from models.json (central source of truth) with zero hardcoded model IDs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import settings

logger = logging.getLogger(__name__)

MODELS_JSON_PATH = Path(__file__).resolve().parent / "models.json"

# Fields exposed publicly by default (excluding internal operational details like pull, enabled, path)
PUBLIC_DEFAULT_FIELDS: Set[str] = {
    "id",
    "name",
    "type",
    "size",
    "quantization",
    "advantages",
    "disadvantages",
    "good_for",
    "default",
}


@dataclass
class ModelEntry:
    id: str
    name: str
    type: str  # "llm" or "embedding"
    size: str
    quantization: str
    local_or_pull: str
    pull: bool
    enabled: bool
    default: bool
    advantages: List[str]
    disadvantages: List[str]
    good_for: List[str]
    path: str
    quantization_flag: Optional[str] = None
    dtype: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)

    def to_dict(self, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Serialize model entry to dict.
        If fields is provided, only include those specified fields.
        Otherwise, include standard public fields (excluding internal operational fields).
        """
        full_dict = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "size": self.size,
            "quantization": self.quantization,
            "local_or_pull": self.local_or_pull,
            "pull": self.pull,
            "enabled": self.enabled,
            "default": self.default,
            "advantages": self.advantages,
            "disadvantages": self.disadvantages,
            "good_for": self.good_for,
            "path": self.path,
        }

        if fields:
            # Return only requested keys
            req_set = {f.strip() for f in fields if f.strip()}
            return {k: full_dict[k] for k in req_set if k in full_dict}

        # Default clean public response
        return {k: full_dict[k] for k in PUBLIC_DEFAULT_FIELDS if k in full_dict}


class ModelRegistry:
    def __init__(self, json_path: Path = MODELS_JSON_PATH):
        self.json_path = json_path
        self._models: Dict[str, ModelEntry] = {}
        self.load()

    def load(self) -> None:
        """Parse models.json and resolve environment directory templates."""
        if not self.json_path.exists():
            logger.error("models.json not found at %s", self.json_path)
            self._models = {}
            return

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_models = data.get("models", [])

            parsed: Dict[str, ModelEntry] = {}
            for item in raw_models:
                m_id = item.get("id")
                if not m_id:
                    continue

                entry = ModelEntry(
                    id=m_id,
                    name=item.get("name", m_id),
                    type=item.get("type", "llm"),
                    size=item.get("size", "unknown"),
                    quantization=item.get("quantization", "None"),
                    local_or_pull=item.get("local_or_pull", "local"),
                    pull=bool(item.get("pull", False)),
                    enabled=bool(item.get("enabled", True)),
                    default=bool(item.get("default", False)),
                    advantages=item.get("advantages", []),
                    disadvantages=item.get("disadvantages", []),
                    good_for=item.get("good_for", []),
                    path=item.get("path", ""),
                    quantization_flag=item.get("quantization_flag"),
                    dtype=item.get("dtype"),
                    extra_args=item.get("extra_args", []),
                )
                parsed[m_id] = entry

            self._models = parsed
            logger.info("Loaded %d models from %s", len(self._models), self.json_path)

        except Exception as exc:
            logger.error("Failed to parse models.json: %s", exc)
            self._models = {}

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Retrieve model entry by exact identifier."""
        if not model_id:
            return None
        return self._models.get(model_id.strip())

    def get_default_model(self, model_type: str = "llm") -> Optional[ModelEntry]:
        """Return the model marked with default: true for the given type."""
        for m in self._models.values():
            if m.type == model_type and m.default and m.enabled:
                return m
        # Fallback to first enabled model of that type
        for m in self._models.values():
            if m.type == model_type and m.enabled:
                return m
        return None

    def list_models(
        self,
        model_type: Optional[str] = None,
        include_disabled: bool = False,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List models formatted with specified fields or default public fields.
        Filters out disabled models by default.
        """
        results: List[Dict[str, Any]] = []
        for m in self._models.values():
            if not include_disabled and not m.enabled:
                continue
            if model_type and m.type != model_type:
                continue
            results.append(m.to_dict(fields=fields))
        return results

    def get_all_entries(self) -> List[ModelEntry]:
        return list(self._models.values())


# Global registry instance
registry = ModelRegistry()


def get_model(model_id: str) -> Optional[ModelEntry]:
    return registry.get_model(model_id)


def get_default_model(model_type: str = "llm") -> Optional[ModelEntry]:
    return registry.get_default_model(model_type)


def list_models(
    model_type: Optional[str] = None,
    include_disabled: bool = False,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    return registry.list_models(
        model_type=model_type,
        include_disabled=include_disabled,
        fields=fields,
    )


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

    if model_entry.quantization_flag:
        cmd.extend(["--quantization", model_entry.quantization_flag])

    if model_entry.dtype:
        cmd.extend(["--dtype", model_entry.dtype])

    if model_entry.extra_args:
        cmd.extend(model_entry.extra_args)

    return cmd
