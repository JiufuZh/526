from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(cfg: Dict[str, Any], adapter_path: Optional[str] = None):
    model_name = cfg["model"].get("base_model") or cfg["model"].get("tokenizer_name")
    lora_cfg = cfg.get("lora", {})
    use_4bit = bool(lora_cfg.get("use_4bit", False))

    quantization_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except Exception as exc:
            raise RuntimeError("4-bit QLoRA requested, but bitsandbytes quantization could not be configured.") from exc

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.get("training", {}).get("bf16", True) else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def add_lora_adapter(model, cfg: Dict[str, Any]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("use_4bit", False):
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg.get("target_modules", [])),
    )
    return get_peft_model(model, peft_cfg)
