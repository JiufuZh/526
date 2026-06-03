from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from defect_detection.config import ensure_dirs, load_yaml
from defect_detection.data import load_or_download_dataset
from defect_detection.io_utils import save_json, set_seed
from defect_detection.metrics import compute_binary_metrics


class IntLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "labels" in inputs:
            inputs["labels"] = inputs["labels"].long()
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    seed = int(cfg["project"].get("seed", 42))
    set_seed(seed)
    run_dir = ensure_dirs({**cfg, "training": {"run_name": cfg["encoder"]["run_name"]}})
    model_dir = Path(args.model_dir) if args.model_dir else run_dir / "final_model"

    ds = load_or_download_dataset(cfg)
    if args.split not in ds:
        raise ValueError(f"Unknown split {args.split!r}; available splits: {sorted(ds.keys())}")

    ecfg = cfg["encoder"]
    text_col = cfg["data"].get("text_column", "func")
    label_col = cfg["data"].get("label_column", "target")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    def tok(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=int(ecfg.get("max_length", 512)))

    tokenized = ds.map(tok, batched=True)
    tokenized = tokenized.rename_column(label_col, "labels") if label_col != "labels" else tokenized

    def cast_labels(batch):
        return {"labels": [int(label) for label in batch["labels"]]}

    tokenized = tokenized.map(cast_labels, batched=True)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    drop_cols = [c for c in tokenized[args.split].column_names if c not in keep_cols]
    eval_ds = tokenized[args.split].remove_columns(drop_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        attn_implementation=str(ecfg.get("attn_implementation", "eager")),
    )
    model.config.problem_type = "single_label_classification"
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collator(features):
        batch = base_collator(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].long()
        return batch

    training_args = TrainingArguments(
        output_dir=str(run_dir / "eval_tmp"),
        per_device_eval_batch_size=int(ecfg.get("per_device_eval_batch_size", 16)),
        bf16=bool(ecfg.get("bf16", True)),
        fp16=bool(ecfg.get("fp16", False)),
        report_to=str(ecfg.get("report_to", "none")),
    )
    trainer = IntLabelTrainer(model=model, args=training_args, data_collator=collator)
    preds = trainer.predict(eval_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    metrics = compute_binary_metrics(preds.label_ids, y_pred)
    metrics = {f"{args.split}_{key}": value for key, value in metrics.items()}
    save_json(metrics, run_dir / f"{args.split}_metrics.json")
    pd.DataFrame({"y_true": preds.label_ids, "y_pred": y_pred}).to_csv(
        run_dir / f"{args.split}_predictions.csv", index=False
    )
    print(metrics)


if __name__ == "__main__":
    main()
