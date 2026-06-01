from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from defect_detection.config import load_yaml
from defect_detection.data import load_or_download_dataset
from defect_detection.io_utils import save_json
from defect_detection.prompts import build_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    ds = load_or_download_dataset(cfg)
    split_ds = ds[args.split]
    text_col = cfg["data"].get("text_column", "func")
    tokenizer_name = cfg["model"].get("tokenizer_name", cfg["model"].get("base_model"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)
    variant = cfg["model"].get("prompt_variant", "detailed")

    lengths = []
    for ex in split_ds:
        prompt = build_prompt(ex[text_col], variant=variant)
        lengths.append(len(tokenizer(prompt, add_special_tokens=True).input_ids))

    arr = np.asarray(lengths)
    summary = {
        "split": args.split,
        "n": int(len(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "coverage_256": float((arr <= 256).mean()),
        "coverage_512": float((arr <= 512).mean()),
        "coverage_1024": float((arr <= 1024).mean()),
    }
    out_dir = Path(cfg["data"].get("cache_dir", "data/processed/code_x_glue_cc_defect_detection"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_dir / f"token_profile_{args.split}.json")
    pd.DataFrame({"length": lengths}).to_csv(out_dir / f"token_lengths_{args.split}.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
