"""
Unit tests for ekoi-llm-router catalog, scheduler, and API routes.
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

from models_catalog import CATALOG, get_model, list_models, resolve_vllm_command
from config import settings


class TestModelsCatalog(unittest.TestCase):
    def test_catalog_contains_expected_models(self):
        self.assertGreaterEqual(len(CATALOG), 28)

        # Check default Qwen model
        qwen_7b_awq = get_model("Qwen2.5-7B-Instruct_INT4_AWQ")
        self.assertIsNotNone(qwen_7b_awq)
        self.assertEqual(qwen_7b_awq.quantization, "awq")
        self.assertEqual(qwen_7b_awq.type, "llm")
        self.assertIn("Qwen2.5-7B-Instruct_INT4_AWQ", qwen_7b_awq.path)

        # Check Qwen FP16
        qwen_14b_fp16 = get_model("Qwen2.5-14B-Instruct_FP16_Baseline")
        self.assertIsNotNone(qwen_14b_fp16)
        self.assertIsNone(qwen_14b_fp16.quantization)

        # Check Llama BNB
        llama_bnb = get_model("meta-llama__Llama-3.2-3B-Instruct-BNB-4bit")
        self.assertIsNotNone(llama_bnb)
        self.assertEqual(llama_bnb.quantization, "bitsandbytes")

        # Check Embedding model
        nomic_embed = get_model("nomic-ai/nomic-embed-text-v1.5")
        self.assertIsNotNone(nomic_embed)
        self.assertEqual(nomic_embed.type, "embedding")

    def test_vllm_command_generation(self):
        model = get_model("Qwen2.5-7B-Instruct_INT4_AWQ")
        cmd = resolve_vllm_command(model)
        self.assertIn("vllm.entrypoints.openai.api_server", cmd)
        self.assertIn("--quantization", cmd)
        self.assertIn("awq", cmd)
        self.assertIn("--port", cmd)
        self.assertIn(str(settings.VLLM_INTERNAL_PORT), cmd)

    def test_filter_models_by_type(self):
        llms = list_models("llm")
        embeddings = list_models("embedding")
        self.assertTrue(all(m.type == "llm" for m in llms))
        self.assertTrue(all(m.type == "embedding" for m in embeddings))
        self.assertEqual(len(embeddings), 1)


if __name__ == "__main__":
    unittest.main()
