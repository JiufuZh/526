from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from defect_detection.config import ensure_dirs, load_yaml
from defect_detection.data import load_or_download_dataset
from defect_detection.io_utils import save_json, set_seed
from defect_detection.label_norm import is_conforming_output, normalize_label
from defect_detection.llm_utils import load_causal_lm, load_tokenizer
from defect_detection.metrics import compute_binary_metrics
from defect_detection.prompts import build_prompt
from defect_detection.scoring import score_two_labels


def generate_predictions(model, tokenizer, prompts, max_new_tokens=8, batch_size=4):
    model.eval()
    preds, texts = [], []
    for i in tqdm(range(0, len(prompts), batch_size), desc="generating"):
        batch_prompts = prompts[i:i + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_width = enc["input_ids"].shape[1]
        for row in out:
            gen_ids = row[input_width:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            texts.append(text)
            preds.append(normalize_label(text))
    return preds, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--mode", default=None, choices=["generate", "logprob"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    run_dir = ensure_dirs(cfg)

    tokenizer = load_tokenizer(cfg["model"].get("tokenizer_name", cfg["model"]["base_model"]))
    model = load_causal_lm(cfg, adapter_path=args.adapter)
    ds = load_or_download_dataset(cfg)
    split_ds = ds[args.split]
    text_col = cfg["data"].get("text_column", "func")
    label_col = cfg["data"].get("label_column", "target")
    variant = cfg["model"].get("prompt_variant", "detailed")
    prompts = [build_prompt(x, variant=variant) for x in split_ds[text_col]]
    y_true = list(map(int, split_ds[label_col]))

    mode = args.mode or cfg.get("evaluation", {}).get("classification_mode", "logprob")
    if mode == "logprob":
        y_pred = score_two_labels(model, tokenizer, prompts, max_prompt_length=int(cfg["model"].get("max_length", 512)))
        raw_outputs = ["<logprob_scoring>" for _ in y_pred]
        nonconforming_rate = 0.0
    else:
        y_pred, raw_outputs = generate_predictions(
            model,
            tokenizer,
            prompts,
            max_new_tokens=int(cfg["evaluation"].get("generation_max_new_tokens", 8)),
            batch_size=int(cfg["evaluation"].get("inference_batch_size", 4)),
        )
        nonconforming_rate = 1.0 - sum(is_conforming_output(x) for x in raw_outputs) / max(1, len(raw_outputs))

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["nonconforming_output_rate"] = float(nonconforming_rate)
    if args.adapter:
        metrics["adapter_path"] = args.adapter
    metrics["classification_mode"] = mode
    save_json(metrics, run_dir / f"{args.split}_metrics.json")
    pd.DataFrame({
        "id": split_ds["id"] if "id" in split_ds.column_names else list(range(len(split_ds))),
        "y_true": y_true,
        "y_pred": y_pred,
        "raw_output": raw_outputs,
    }).to_csv(run_dir / f"{args.split}_predictions.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
