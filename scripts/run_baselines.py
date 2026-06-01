from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from defect_detection.config import ensure_dirs, load_yaml
from defect_detection.data import load_or_download_dataset
from defect_detection.io_utils import save_json, set_seed
from defect_detection.metrics import compute_binary_metrics


def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b|==|!=|<=|>=|->", ngram_range=(1, 2), max_features=120000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=120000)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    run_dir = ensure_dirs({**cfg, "training": {"run_name": "baselines"}})
    ds = load_or_download_dataset(cfg)
    text_col = cfg["data"].get("text_column", "func")
    label_col = cfg["data"].get("label_column", "target")

    X_train = list(ds["train"][text_col])
    y_train = list(map(int, ds["train"][label_col]))
    X_eval = list(ds[args.eval_split][text_col])
    y_eval = list(map(int, ds[args.eval_split][label_col]))

    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "tfidf_logreg": Pipeline([
            ("tfidf", make_features()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, solver="saga")),
        ]),
        "tfidf_linear_svm": Pipeline([
            ("tfidf", make_features()),
            ("clf", LinearSVC(class_weight="balanced", max_iter=5000)),
        ]),
    }

    all_metrics = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_eval)
        metrics = compute_binary_metrics(y_eval, pred)
        all_metrics[name] = metrics
        pd.DataFrame({"y_true": y_eval, "y_pred": pred}).to_csv(run_dir / f"{name}_{args.eval_split}_predictions.csv", index=False)
        save_json(metrics, run_dir / f"{name}_{args.eval_split}_metrics.json")
        print(name, metrics)

    pd.DataFrame([{"model": k, **v} for k, v in all_metrics.items()]).to_csv(run_dir / f"baseline_{args.eval_split}_summary.csv", index=False)


if __name__ == "__main__":
    main()
