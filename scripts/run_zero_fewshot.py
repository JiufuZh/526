from __future__ import annotations

import argparse

from defect_detection.config import load_yaml
from defect_detection.data import load_or_download_dataset
from defect_detection.io_utils import save_json, set_seed
from defect_detection.llm_utils import load_causal_lm, load_tokenizer
from defect_detection.metrics import compute_binary_metrics
from defect_detection.prompts import build_fewshot_block, build_prompt
from defect_detection.scoring import score_two_labels


def pick_balanced_fewshot(train_ds, k, label_col, seed=42):
    half = k // 2
    pos = train_ds.filter(lambda ex: int(ex[label_col]) == 1).shuffle(seed=seed).select(range(half))
    neg = train_ds.filter(lambda ex: int(ex[label_col]) == 0).shuffle(seed=seed).select(range(k - half))
    return list(pos) + list(neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--shots", type=int, default=0)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    ds = load_or_download_dataset(cfg)
    tokenizer = load_tokenizer(cfg["model"].get("tokenizer_name", cfg["model"]["base_model"]))
    model = load_causal_lm(cfg)
    text_col = cfg["data"].get("text_column", "func")
    label_col = cfg["data"].get("label_column", "target")

    fewshot_block = None
    if args.shots > 0:
        examples = pick_balanced_fewshot(ds["train"], args.shots, label_col, int(cfg["project"].get("seed", 42)))
        fewshot_block = build_fewshot_block(examples, text_column=text_col, label_column=label_col)

    prompts = [build_prompt(ex[text_col], variant=cfg["model"].get("prompt_variant", "detailed"), fewshot_block=fewshot_block) for ex in ds[args.split]]
    y_true = list(map(int, ds[args.split][label_col]))
    y_pred = score_two_labels(model, tokenizer, prompts, max_prompt_length=int(cfg["model"].get("max_length", 512)))
    metrics = compute_binary_metrics(y_true, y_pred)
    out_name = f"zero_fewshot_{args.shots}_shot_{args.split}_metrics.json"
    save_json(metrics, f"outputs/{out_name}")
    print(metrics)


if __name__ == "__main__":
    main()
