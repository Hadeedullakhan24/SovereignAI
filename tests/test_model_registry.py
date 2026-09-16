"""Unit and Integration Tests for Local LLM Registry and Model Catalog.

Verifies:
    1. LLMRegistry singleton initialization and default registrations.
    2. Model specifications catalog entries (Qwen2.5, SmolLM2, Phi-3.5, etc.).
    3. Model alias resolution across friendly shortcuts.
    4. Physical local model path resolution for installed models.
    5. Installation status detection for present vs uninstalled models.
    6. LLMFactory model creation, caching, and deterministic test model instantiation.
    7. Integration with AgentModelRegistry and models.yaml catalog.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from rag_engine.generation.models.model_registry import LLMRegistry
from rag_engine.generation.models.model_factory import LLMFactory
from rag_engine.generation.models.deterministic_test_llm import DeterministicTestLLM
from agent.model_registry import AgentModelRegistry, get_agent_registry


class TestModelRegistry(unittest.TestCase):
    """Test suite for local model registry and catalog."""

    def setUp(self) -> None:
        self.registry = LLMRegistry.get_instance()
        self.factory = LLMFactory.get_instance()

    def test_registry_singleton(self) -> None:
        """LLMRegistry.get_instance() returns a singleton instance."""
        reg1 = LLMRegistry.get_instance()
        reg2 = LLMRegistry.get_instance()
        self.assertIs(reg1, reg2)

    def test_deterministic_test_registered(self) -> None:
        """Deterministic test model is registered and resolves to DeterministicTestLLM."""
        cls = self.registry.get("deterministic_test")
        self.assertIsNotNone(cls)
        self.assertEqual(cls, DeterministicTestLLM)
        self.assertTrue(self.registry.is_model_installed("deterministic_test"))
        self.assertTrue(self.registry.is_model_installed("mock"))

    def test_smollm2_registered_and_resolves(self) -> None:
        """SmolLM2-1.7B-Instruct is registered with correct aliases and spec."""
        canonical = self.registry.resolve_alias("smollm2")
        self.assertEqual(canonical, "HuggingFaceTB/SmolLM2-1.7B-Instruct")

        spec = self.registry.get_specification("smollm2")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["family"], "smollm")
        self.assertEqual(spec["local_folder"], "smollm2-1.7b-instruct")

        # Must resolve to physical folder on disk
        path = self.registry.find_local_model_path("smollm2")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_dir())
        self.assertTrue(self.registry.is_model_installed("smollm2"))
        self.assertTrue(self.registry.is_model_installed("smollm2-1.7b-instruct"))

    def test_phi35_registered_and_resolves(self) -> None:
        """Phi-3.5-mini-instruct is registered with correct aliases and spec."""
        canonical = self.registry.resolve_alias("phi3.5")
        self.assertEqual(canonical, "microsoft/Phi-3.5-mini-instruct")

        spec = self.registry.get_specification("phi3.5")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["family"], "phi3.5")
        self.assertEqual(spec["local_folder"], "phi-3.5-mini-instruct")

        # Must resolve to physical folder on disk
        path = self.registry.find_local_model_path("phi3.5")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_dir())
        self.assertTrue(self.registry.is_model_installed("phi3.5"))
        self.assertTrue(self.registry.is_model_installed("phi-3.5-mini-instruct"))

    def test_qwen25_registered_and_resolves(self) -> None:
        """Qwen2.5-1.5B-Instruct is registered and resolves to local weights."""
        canonical = self.registry.resolve_alias("qwen")
        self.assertEqual(canonical, "Qwen/Qwen2.5-1.5B-Instruct")

        path = self.registry.find_local_model_path("qwen2.5-1.5b-instruct")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_dir())
        self.assertTrue(self.registry.is_model_installed("qwen2.5-1.5b-instruct"))

    def test_uninstalled_model_reported_correctly(self) -> None:
        """Uninstalled model specs report is_model_installed=False."""
        self.assertFalse(self.registry.is_model_installed("TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
        self.assertFalse(self.registry.is_model_installed("tinyllama"))
        self.assertFalse(self.registry.is_model_installed("Qwen/Qwen2.5-7B-Instruct"))

    def test_factory_creates_test_model(self) -> None:
        """LLMFactory creates and caches DeterministicTestLLM instances cleanly."""
        model = self.factory.create("deterministic_test")
        self.assertIsInstance(model, DeterministicTestLLM)
        # Should return cached instance on second call
        model2 = self.factory.create("deterministic_test")
        self.assertIs(model, model2)

    def test_agent_model_registry_alignment(self) -> None:
        """AgentModelRegistry aligns with installed disk models."""
        agent_reg = get_agent_registry()
        installed_names = [r.name for r in agent_reg.ready()]
        self.assertIn("qwen2.5-1.5b-instruct", installed_names)
        self.assertIn("smollm2-1.7b-instruct", installed_names)
        self.assertIn("phi-3.5-mini-instruct", installed_names)


if __name__ == "__main__":
    unittest.main()
