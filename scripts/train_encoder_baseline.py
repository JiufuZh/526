from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from defect_detection.config import ensure_dirs, load_yaml
from defect_detection.data import load_or_download_dataset
from defect_detection.io_utils import save_json, set_seed
from defect_detection.metrics import compute_binary_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["project"].get("seed", 42))
    set_seed(seed)
    run_dir = ensure_dirs({**cfg, "training": {"run_name": cfg["encoder"]["run_name"]}})
    ds = load_or_download_dataset(cfg)
    ecfg = cfg["encoder"]
    text_col = cfg["data"].get("text_column", "func")
    label_col = cfg["data"].get("label_column", "target")

    tokenizer = AutoTokenizer.from_pretrained(ecfg["model_name"], use_fast=True)

    def tok(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=int(ecfg.get("max_length", 512)))

    tokenized = ds.map(tok, batched=True)
    tokenized = tokenized.rename_column(label_col, "labels") if label_col != "labels" else tokenized
    keep_cols = {"input_ids", "attention_mask", "labels"}
    drop_cols = [c for c in tokenized["train"].column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(drop_cols)

    model = AutoModelForSequenceClassification.from_pretrained(ecfg["model_name"], num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_binary_metrics(labels, preds)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=float(ecfg.get("epochs", 3)),
        learning_rate=float(ecfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(ecfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(ecfg.get("per_device_eval_batch_size", 16)),
        gradient_accumulation_steps=int(ecfg.get("gradient_accumulation_steps", 1)),
        weight_decay=float(ecfg.get("weight_decay", 0.01)),
        warmup_ratio=float(ecfg.get("warmup_ratio", 0.06)),
        logging_steps=int(ecfg.get("logging_steps", 50)),
        eval_strategy="steps",
        eval_steps=int(ecfg.get("eval_steps", 500)),
        save_steps=int(ecfg.get("save_steps", 500)),
        save_total_limit=int(ecfg.get("save_total_limit", 2)),
        bf16=bool(ecfg.get("bf16", True)),
        fp16=bool(ecfg.get("fp16", False)),
        report_to=str(ecfg.get("report_to", "none")),
        run_name=str(ecfg.get("run_name", "encoder_baseline")),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(tokenized["validation"])
    save_json(metrics, run_dir / "validation_metrics.json")
    preds = trainer.predict(tokenized["validation"])
    y_pred = np.argmax(preds.predictions, axis=-1)
    pd.DataFrame({"y_true": preds.label_ids, "y_pred": y_pred}).to_csv(run_dir / "validation_predictions.csv", index=False)
    trainer.save_model(str(run_dir / "final_model"))
    tokenizer.save_pretrained(str(run_dir / "final_model"))
    print(metrics)


if __name__ == "__main__":
    main()
