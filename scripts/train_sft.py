from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import TrainingArguments

from defect_detection.config import ensure_dirs, load_yaml
from defect_detection.data import CausalDefectDataset, balanced_subset, fractional_subset, load_or_download_dataset, make_data_collator
from defect_detection.io_utils import save_json, set_seed
from defect_detection.llm_utils import add_lora_adapter, load_causal_lm, load_tokenizer
from defect_detection.losses import OptionalLossTrainer


def compute_balanced_class_weights(labels):
    counts = torch.bincount(torch.tensor(list(map(int, labels)), dtype=torch.long), minlength=2).float()
    total = counts.sum().clamp_min(1.0)
    return (total / (2.0 * counts.clamp_min(1.0))).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["project"].get("seed", 42))
    set_seed(seed)
    run_dir = ensure_dirs(cfg)
    save_json(cfg, run_dir / "resolved_config.json")

    tokenizer = load_tokenizer(cfg["model"].get("tokenizer_name", cfg["model"]["base_model"]))
    model = load_causal_lm(cfg)
    if cfg.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if cfg.get("lora", {}).get("enabled", True):
        model = add_lora_adapter(model, cfg)
        model.print_trainable_parameters()

    ds = load_or_download_dataset(cfg)
    train_ds = ds["train"]
    train_ds = fractional_subset(train_ds, float(cfg["training"].get("train_fraction", 1.0)), seed=seed)
    smoke_n = cfg["training"].get("balanced_smoke_n")
    if smoke_n:
        train_ds = balanced_subset(train_ds, int(smoke_n), label_column=cfg["data"].get("label_column", "target"), seed=seed)
    eval_ds = ds["validation"]

    train_dataset = CausalDefectDataset(train_ds, tokenizer, cfg, split="train")
    eval_dataset = CausalDefectDataset(eval_ds, tokenizer, cfg, split="validation")
    collator = make_data_collator(tokenizer)

    tcfg = cfg["training"]
    class_weights = None
    if bool(tcfg.get("class_weighting", False)):
        label_col = cfg["data"].get("label_column", "target")
        class_weights = compute_balanced_class_weights(train_ds[label_col])
        save_json({"class_weights": {"non_defective": class_weights[0], "defective": class_weights[1]}}, run_dir / "class_weights.json")

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=float(tcfg.get("epochs", 3)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 0.3)),
        logging_steps=int(tcfg.get("logging_steps", 20)),
        save_steps=int(tcfg.get("save_steps", 500)),
        eval_steps=int(tcfg.get("eval_steps", 500)),
        eval_strategy=str(tcfg.get("eval_strategy", "steps")),
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        bf16=bool(tcfg.get("bf16", True)),
        fp16=bool(tcfg.get("fp16", False)),
        report_to=str(tcfg.get("report_to", "none")),
        run_name=str(tcfg.get("run_name", "sft_run")),
        remove_unused_columns=False,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        optim="paged_adamw_8bit" if cfg.get("lora", {}).get("use_4bit", False) else "adamw_torch",
        lr_scheduler_type="cosine",
        logging_dir=str(run_dir / "logs"),
    )

    trainer = OptionalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        focal_loss_gamma=tcfg.get("focal_loss_gamma"),
        label_smoothing=float(tcfg.get("label_smoothing", 0.0) or 0.0),
        class_weights=class_weights,
    )
    trainer.train()
    final_dir = run_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final adapter/model to {final_dir}")


if __name__ == "__main__":
    main()
