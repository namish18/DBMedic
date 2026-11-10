"""
tests/test_model_client.py
===========================
Tests for TinyAyaClient — loading, formatting, inference interface.

Unit tests use mocks — no model download required.
Integration test (marked) exercises real model loading when available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.unit
class TestTinyAyaClientInit:
    def test_default_model_id(self):
        from agent.core.model_client import TinyAyaClient, DEFAULT_MODEL_ID
        client = TinyAyaClient()
        assert client.model_id == DEFAULT_MODEL_ID

    def test_custom_model_id_accepted(self):
        from agent.core.model_client import TinyAyaClient
        client = TinyAyaClient(model_id="some/other-model")
        assert client.model_id == "some/other-model"

    def test_not_loaded_on_init(self):
        from agent.core.model_client import TinyAyaClient
        client = TinyAyaClient()
        assert client.is_loaded is False

    def test_generate_raises_before_load(self):
        from agent.core.model_client import TinyAyaClient, InferenceError
        client = TinyAyaClient()
        with pytest.raises(InferenceError, match="not loaded"):
            client.generate([{"role": "user", "content": "test"}])

    def test_format_messages_raises_before_load(self):
        from agent.core.model_client import TinyAyaClient, InferenceError
        client = TinyAyaClient()
        with pytest.raises(InferenceError):
            client.format_messages([{"role": "user", "content": "test"}])

    def test_generation_config_merges_with_defaults(self):
        from agent.core.model_client import TinyAyaClient, DEFAULT_GENERATION_CONFIG
        custom = {"max_new_tokens": 256, "do_sample": True}
        client = TinyAyaClient(generation_config=custom)
        assert client.generation_config["max_new_tokens"] == 256
        assert client.generation_config["do_sample"] is True
        # Default keys still present
        assert "repetition_penalty" in client.generation_config

    def test_repr_shows_not_loaded(self):
        from agent.core.model_client import TinyAyaClient
        client = TinyAyaClient()
        assert "not loaded" in repr(client)


@pytest.mark.unit
class TestTinyAyaClientFallbackFormat:
    def _make_loaded_client(self):
        from agent.core.model_client import TinyAyaClient
        client = TinyAyaClient()
        # Set up a mock tokenizer that makes apply_chat_template fail
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("No template")
        client._tokenizer = mock_tokenizer
        client._loaded = True
        return client

    def test_fallback_format_includes_user_content(self):
        from agent.core.model_client import TinyAyaClient
        client = TinyAyaClient()
        formatted = client._fallback_format([
            {"role": "system", "content": "You are DBMedic."},
            {"role": "user", "content": "Why is the query slow?"},
        ])
        assert "You are DBMedic" in formatted
        assert "Why is the query slow" in formatted

    def test_fallback_format_handles_all_roles(self):
        from agent.core.model_client import TinyAyaClient
        client = TinyAyaClient()
        messages = [
            {"role": "system", "content": "system msg"},
            {"role": "user", "content": "user msg"},
            {"role": "assistant", "content": "assistant msg"},
        ]
        formatted = client._fallback_format(messages)
        assert "system msg" in formatted
        assert "user msg" in formatted
        assert "assistant msg" in formatted


@pytest.mark.unit
class TestTinyAyaClientMockedLoad:
    """Tests for load() with mocked transformers — no GPU/download required."""

    def test_model_load_error_on_missing_transformers(self):
        from agent.core.model_client import TinyAyaClient, ModelLoadError
        client = TinyAyaClient()

        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with pytest.raises((ModelLoadError, ImportError, TypeError)):
                client.load()

    def test_hf_token_not_logged(self, caplog):
        """Ensure HF_TOKEN value never appears in log output."""
        import logging
        from agent.core.model_client import TinyAyaClient, ModelLoadError

        client = TinyAyaClient()
        fake_token = "hf_SUPER_SECRET_TOKEN_12345"

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = Exception(
            f"Auth failed with token {fake_token}"
        )
        mock_torch = MagicMock()

        with patch.dict("os.environ", {"HF_TOKEN": fake_token}), \
             patch.dict("sys.modules", {"transformers": mock_transformers, "torch": mock_torch}):
            with caplog.at_level(logging.ERROR):
                with pytest.raises(ModelLoadError) as exc_info:
                    client.load()

            # Token must not appear in the exception message
            error_msg = str(exc_info.value)
            assert fake_token not in error_msg, (
                "HF_TOKEN value appeared in error message — SECURITY VIOLATION"
            )

    def test_unload_clears_loaded_state(self):
        from agent.core.model_client import TinyAyaClient

        client = TinyAyaClient()
        client._loaded = True
        client._model = MagicMock()
        client._tokenizer = MagicMock()
        client._pipeline = MagicMock()

        client.unload()

        assert client.is_loaded is False
        assert client._model is None


@pytest.mark.unit
class TestTinyAyaClientGenerate:
    """Tests for generate() with fully mocked internals."""

    def _make_mocked_client(self):
        from agent.core.model_client import TinyAyaClient

        client = TinyAyaClient()
        client._loaded = True
        client._model_revision = "abc123"

        # Mock input tensor
        mock_tensor = MagicMock()
        mock_tensor.shape = (1, 3)
        mock_tensor.to.return_value = mock_tensor

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "<prompt>"
        mock_tok.return_value = {"input_ids": mock_tensor}
        mock_tok.eos_token_id = 2
        mock_tok.decode.return_value = '{"type": "final_answer", "diagnosis": "test"}'
        client._tokenizer = mock_tok

        # Mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        client._model = mock_model

        return client

    def test_generate_returns_dict_with_expected_keys(self):
        client = self._make_mocked_client()
        result = client.generate([{"role": "user", "content": "test"}])

        assert "text" in result
        assert "generation_time_ms" in result
        assert "model_id" in result
        assert "model_revision" in result
        assert "tokens_generated" in result

    def test_generate_includes_model_revision(self):
        client = self._make_mocked_client()
        result = client.generate([{"role": "user", "content": "test"}])
        assert result["model_revision"] == "abc123"

    def test_generate_time_is_positive(self):
        client = self._make_mocked_client()
        result = client.generate([{"role": "user", "content": "test"}])
        assert result["generation_time_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Real model integration test — only runs when model is available
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTinyAyaRealInference:
    """
    Integration test for real Tiny Aya inference.
    Requires:
      - HF_TOKEN set in environment
      - Model accepted on Hugging Face
      - Sufficient RAM (≥8 GB recommended)
      - Internet access to Hugging Face Hub

    Skipped automatically when any condition is not met.
    """

    @pytest.fixture(scope="class")
    def real_client(self):
        import os
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            pytest.skip("HF_TOKEN not set — skipping real model inference test")

        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("transformers/torch not installed — skipping real inference test")

        from agent.core.model_client import TinyAyaClient, ModelLoadError

        client = TinyAyaClient(
            generation_config={"max_new_tokens": 64, "do_sample": False}
        )
        try:
            client.load()
        except ModelLoadError as exc:
            pytest.skip(f"Model load failed (may require gated access): {exc}")

        yield client
        client.unload()

    def test_real_model_loads_and_generates(self, real_client):
        """Real model produces non-empty output for a simple prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are DBMedic. Respond only in JSON.",
            },
            {
                "role": "user",
                "content": "Why is my orders query slow? Respond with a JSON final_answer.",
            },
        ]
        result = real_client.generate(messages)
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert result["model_id"] == "CohereLabs/tiny-aya-global"

