from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    row = {"source_file": str(path)}
    for key, value in obj.items():
        if isinstance(value, (int, float, str)):
            row[key] = value
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-glob", default="outputs/**/*.json")
    parser.add_argument("--output", default="outputs/report_ready_metrics.csv")
    args = parser.parse_args()
    files = sorted(Path().glob(args.metrics_glob))
    rows = [flatten_metrics(p) for p in files if p.name.endswith("metrics.json")]
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    md = df.to_markdown(index=False) if not df.empty else "No metrics found."
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(df)


if __name__ == "__main__":
    main()
