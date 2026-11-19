"""
Unit tests for ekoi-llm-router models.json catalog, field projection, and command generation.
Runs without GPU or vLLM dependency.
"""

import os
import sys
from pathlib import Path
import unittest

# Ensure router directory is on path
router_root = Path(__file__).resolve().parent.parent
if str(router_root) not in sys.path:
    sys.path.insert(0, str(router_root))

os.environ["MOCK_MODE"] = "true"

from models_catalog import get_default_model, get_model, list_models, registry, resolve_vllm_command
from embedding_service import embedding_service
from config import settings


class TestModelsCatalog(unittest.TestCase):
    def test_catalog_loaded_from_json(self):
        all_entries = registry.get_all_entries()
        self.assertGreaterEqual(len(all_entries), 27)

        # Check default LLM dynamically
        default_llm = get_default_model("llm")
        self.assertIsNotNone(default_llm)
        self.assertTrue(default_llm.default)
        self.assertEqual(default_llm.type, "llm")
        self.assertEqual(default_llm.id, "Qwen2.5-7B-Instruct_INT4_AWQ")
        self.assertEqual(default_llm.name, "Qwen 2.5 7B Instruct (INT4 AWQ)")
        self.assertEqual(default_llm.quantization, "INT4 AWQ")
        self.assertIn("General Candidate Screening", default_llm.good_for)

    def test_dedicated_embedding_service(self):
        # Single dedicated embedder configured in settings
        self.assertEqual(settings.EMBEDDING_MODEL_ID, "nomic-ai/nomic-embed-text-v1.5")
        self.assertEqual(embedding_service.model_id, "nomic-ai/nomic-embed-text-v1.5")

        # Test sync embedding mock output
        results = embedding_service._sync_embed(["Hello world", "Test sentence"])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 768)

    def test_fields_projection(self):
        # Default public fields: excludes pull, enabled, path
        models = list_models(model_type="llm")
        first = models[0]
        self.assertIn("id", first)
        self.assertIn("name", first)
        self.assertIn("size", first)
        self.assertIn("quantization", first)
        self.assertIn("advantages", first)
        self.assertIn("good_for", first)
        self.assertNotIn("pull", first)
        self.assertNotIn("enabled", first)
        self.assertNotIn("path", first)

        # Explicit projection: only id and name
        projected = list_models(model_type="llm", fields=["id", "name"])
        first_proj = projected[0]
        self.assertEqual(set(first_proj.keys()), {"id", "name"})

    def test_vllm_command_generation(self):
        model = get_model("Qwen2.5-7B-Instruct_INT4_AWQ")
        cmd = resolve_vllm_command(model)
        self.assertIn("vllm.entrypoints.openai.api_server", cmd)
        self.assertIn("--quantization", cmd)
        self.assertIn("awq", cmd)
        self.assertIn("--port", cmd)
        self.assertIn(str(settings.VLLM_INTERNAL_PORT), cmd)


if __name__ == "__main__":
    unittest.main()
