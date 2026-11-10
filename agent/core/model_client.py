"""
agent/core/model_client.py
==========================
Hugging Face Transformers client for CohereLabs/tiny-aya-global.

Responsibilities:
  - Load Tiny Aya Global from Hugging Face Hub.
  - Format messages using the model's official chat template.
  - Generate model responses with configurable parameters.
  - Decode generated tokens.
  - Expose a reusable inference interface.

Security:
  - HF_TOKEN is loaded from environment variables only.
  - Tokens are never logged.
  - Model loading failures are handled gracefully.

Hardware:
  - Supports device_map="auto" for GPU/CPU selection.
  - Falls back to CPU if no GPU is available.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"

DEFAULT_GENERATION_CONFIG: dict[str, Any] = {
    "max_new_tokens": 512,
    "do_sample": False,          # Deterministic for reproducible baselines
    "temperature": 1.0,          # Required even when do_sample=False by some versions
    "repetition_penalty": 1.1,
    "pad_token_id": None,        # Set dynamically after tokenizer load
    "eos_token_id": None,        # Set dynamically after tokenizer load
}


class ModelLoadError(RuntimeError):
    """Raised when the model cannot be loaded."""


class InferenceError(RuntimeError):
    """Raised when model inference fails."""


class TinyAyaClient:
    """
    Client wrapper for CohereLabs/tiny-aya-global.

    Usage:
        client = TinyAyaClient()
        client.load()
        response = client.generate(messages)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        generation_config: dict[str, Any] | None = None,
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.generation_config = {
            **DEFAULT_GENERATION_CONFIG,
            **(generation_config or {}),
        }
        self.device_map = device_map
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._loaded = False
        self._model_revision: str | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the model and tokenizer from Hugging Face Hub.

        Requires HF_TOKEN in environment for gated model access.
        Never logs the token value.
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        except ImportError as exc:
            raise ModelLoadError(
                "transformers and torch are required for model inference. "
                "Install with: pip install transformers torch accelerate"
            ) from exc

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            logger.info("HF_TOKEN found in environment (not logged)")
        else:
            logger.warning(
                "HF_TOKEN not set. Gated model access may fail. "
                "Set HF_TOKEN in your .env file."
            )

        logger.info("Loading model: %s", self.model_id)
        logger.info("Device map: %s", self.device_map)

        try:
            # Determine device
            if self.device_map == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Resolved device: %s", device)
            else:
                device = self.device_map

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=hf_token,
                trust_remote_code=False,
            )

            # Load model
            logger.info("Loading model weights (this may take several minutes)...")
            import torch
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=hf_token,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )

            # Try to get model revision/commit hash
            try:
                from huggingface_hub import model_info
                info = model_info(self.model_id, token=hf_token)
                self._model_revision = info.sha
                logger.info("Model commit SHA: %s", self._model_revision)
            except Exception:
                self._model_revision = "unknown"

            # Build pipeline for simpler generation interface
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device_map=self.device_map,
            )

            # Set dynamic token ids
            self.generation_config["pad_token_id"] = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            self.generation_config["eos_token_id"] = self._tokenizer.eos_token_id

            self._loaded = True
            logger.info("Model loaded successfully: %s", self.model_id)

        except Exception as exc:
            # Never include the token in error messages
            msg = str(exc)
            if hf_token and hf_token in msg:
                msg = msg.replace(hf_token, "[REDACTED]")
            raise ModelLoadError(f"Failed to load model '{self.model_id}': {msg}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_revision(self) -> str | None:
        return self._model_revision

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        """
        Format messages using the model's official chat template.

        Parameters
        ----------
        messages:
            List of {"role": "...", "content": "..."} dicts.

        Returns
        -------
        str: Formatted prompt string ready for tokenization.
        """
        if not self._tokenizer:
            raise InferenceError("Model not loaded. Call load() first.")

        try:
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted
        except Exception as exc:
            # Fallback: manual formatting
            logger.warning("Chat template application failed (%s), using fallback", exc)
            return self._fallback_format(messages)

    def _fallback_format(self, messages: list[dict[str, str]]) -> str:
        """Fallback message formatter when chat template is unavailable."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|SYSTEM|>{content}<|END|>")
            elif role == "user":
                parts.append(f"<|USER|>{content}<|END|>")
            elif role == "assistant":
                parts.append(f"<|ASSISTANT|>{content}<|END|>")
        parts.append("<|ASSISTANT|>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]],
        generation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a model response for the given messages.

        Parameters
        ----------
        messages:
            Conversation history as list of {"role": ..., "content": ...} dicts.
        generation_config:
            Optional overrides for generation parameters.

        Returns
        -------
        dict:
            - text: str (generated text, excluding prompt)
            - raw_output: str (full decoded output)
            - generation_time_ms: float
            - model_id: str
            - model_revision: str | None
        """
        if not self._loaded:
            raise InferenceError("Model not loaded. Call load() first.")

        config = {**self.generation_config, **(generation_config or {})}

        # Remove None values to avoid issues
        config = {k: v for k, v in config.items() if v is not None}

        try:
            prompt = self.format_messages(messages)

            start_time = time.perf_counter()

            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            # Move to model device if possible
            try:
                device = next(self._model.parameters()).device
                input_ids = input_ids.to(device)
            except Exception:
                pass  # If device transfer fails, proceed on CPU

            prompt_len = input_ids.shape[1]

            # Remove pipeline-incompatible params for direct model.generate call
            gen_kwargs = {k: v for k, v in config.items()
                         if k not in ("pad_token_id",) or v is not None}
            if "pad_token_id" not in gen_kwargs or gen_kwargs.get("pad_token_id") is None:
                gen_kwargs["pad_token_id"] = self._tokenizer.eos_token_id

            try:
                import torch
                no_grad_ctx = torch.no_grad()
            except ImportError:
                from contextlib import nullcontext
                no_grad_ctx = nullcontext()

            with no_grad_ctx:
                outputs = self._model.generate(
                    input_ids,
                    **gen_kwargs,
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Decode only the new tokens (exclude prompt)
            new_tokens = outputs[0][prompt_len:]
            generated_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            logger.debug(
                "Generated %d tokens in %.1f ms",
                len(new_tokens),
                elapsed_ms,
            )

            return {
                "text": generated_text.strip(),
                "raw_output": full_output,
                "generation_time_ms": round(elapsed_ms, 1),
                "model_id": self.model_id,
                "model_revision": self._model_revision,
                "tokens_generated": len(new_tokens),
            }

        except Exception as exc:
            raise InferenceError(f"Model inference failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            del self._model
            del self._tokenizer
            del self._pipeline
            self._model = None
            self._tokenizer = None
            self._pipeline = None
            self._loaded = False
            logger.info("Model unloaded from memory")

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"TinyAyaClient(model_id='{self.model_id}', status={status})"

