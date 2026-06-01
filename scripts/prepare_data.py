from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defect_detection.config import load_yaml
from defect_detection.data import load_or_download_dataset
from defect_detection.io_utils import save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    ds = load_or_download_dataset(cfg)
    text_col = cfg["data"].get("text_column", "func")
    label_col = cfg["data"].get("label_column", "target")

    rows = []
    for split, split_ds in ds.items():
        counts = pd.Series(split_ds[label_col]).value_counts().to_dict()
        rows.append({
            "split": split,
            "n": len(split_ds),
            "defective": int(counts.get(1, 0)),
            "non_defective": int(counts.get(0, 0)),
            "defective_pct": float(counts.get(1, 0) / len(split_ds)) if len(split_ds) else 0.0,
            "duplicate_seen_before": int(sum(split_ds["is_exact_duplicate_seen_before"])) if "is_exact_duplicate_seen_before" in split_ds.column_names else 0,
        })
    out_dir = Path(cfg["data"].get("cache_dir", "data/processed/code_x_glue_cc_defect_detection"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "split_summary.csv", index=False)
    save_json({"splits": rows, "text_column": text_col, "label_column": label_col}, out_dir / "data_summary.json")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
