from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

from defect_detection.error_analysis import summarize_code_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-per-type", type=int, default=20)
    parser.add_argument("--text-column", default="func")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(args.dataset_cache)[args.split]
    pred = pd.read_csv(args.predictions)
    df = ds.to_pandas().reset_index(drop=True)
    merged = pd.concat([df, pred[["y_true", "y_pred"]].reset_index(drop=True)], axis=1)
    merged["error_type"] = "correct"
    merged.loc[(merged.y_true == 0) & (merged.y_pred == 1), "error_type"] = "false_positive"
    merged.loc[(merged.y_true == 1) & (merged.y_pred == 0), "error_type"] = "false_negative"

    samples = []
    for et in ["false_positive", "false_negative"]:
        part = merged[merged.error_type == et].head(args.n_per_type).copy()
        samples.append(part)
    sampled = pd.concat(samples, ignore_index=True) if samples else merged.head(0)
    feature_rows = sampled[args.text_column].apply(summarize_code_features).apply(pd.Series)
    sampled = pd.concat([sampled.reset_index(drop=True), feature_rows.reset_index(drop=True)], axis=1)
    sampled.to_csv(out_dir / "misclassified_samples.csv", index=False)

    taxonomy = sampled.groupby("error_type")[["line_count", "branch_count", "pointer_ops", "array_accesses", "null_checks", "malloc_free_calls", "string_api_calls"]].mean().reset_index()
    taxonomy.to_csv(out_dir / "error_taxonomy_summary.csv", index=False)
    print(taxonomy.to_string(index=False))


if __name__ == "__main__":
    main()
