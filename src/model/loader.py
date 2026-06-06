"""
src/model/loader.py

Loads the fine-tuned CodeLlama-7B model with LoRA adapters.
Supports 4-bit quantization (QLoRA) for memory efficiency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_MODEL_NAME = os.getenv("HF_MODEL_NAME", "microsoft/phi-2")
MODEL_PATH      = os.getenv("MODEL_PATH", "./models/codellama-review-lora")
USE_4BIT        = os.getenv("USE_4BIT", "true").lower() == "true"
DEVICE          = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


# ─── BitsAndBytes 4-bit Config ───────────────────────────────────────────────

def _get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ─── Loader ──────────────────────────────────────────────────────────────────

class ModelLoader:
    """
    Loads CodeLlama base + LoRA adapter weights.

    Usage:
        loader = ModelLoader()
        model, tokenizer = loader.load()
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL_NAME,
        adapter_path: str = MODEL_PATH,
        use_4bit: bool = USE_4BIT,
        device: str = DEVICE,
    ):
        self.base_model   = base_model
        self.adapter_path = adapter_path
        self.use_4bit     = use_4bit
        self.device       = device
        self._model       = None
        self._tokenizer   = None

    def load(self):
        """Load model + tokenizer. Returns (model, tokenizer)."""
        if self._model is not None:
            return self._model, self._tokenizer

        logger.info(f"Loading tokenizer from {self.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        logger.info(
            f"Loading base model {self.base_model} "
            f"{'(4-bit)' if self.use_4bit else '(fp16)'}"
        )
        model_kwargs = dict(
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            device_map="auto",
        )
        if self.use_4bit:
            model_kwargs["quantization_config"] = _get_bnb_config()
        else:
            model_kwargs["torch_dtype"] = torch.float16

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, **model_kwargs
        )

        # Apply LoRA adapter if it exists
        adapter = Path(self.adapter_path)
        if adapter.exists():
            logger.info(f"Loading LoRA adapter from {adapter}")
            model = PeftModel.from_pretrained(base, str(adapter))
            model = model.merge_and_unload()   # merge for faster inference
            logger.info("LoRA adapter merged into base model")
        else:
            logger.warning(
                f"Adapter not found at {adapter}. Using base model only. "
                "Run training/finetune.py first."
            )
            model = base

        model.eval()
        self._model     = model
        self._tokenizer = tokenizer
        logger.info("Model ready ✓")
        return model, tokenizer

    def to_device(self, device: Optional[str] = None):
        """Move model to a specific device."""
        d = device or self.device
        if self._model:
            self._model = self._model.to(d)
        return self


# Singleton instance (import and call .load())
_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
