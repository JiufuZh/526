from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, Any]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "defective_precision": float(precision_binary),
        "defective_recall": float(recall_binary),
        "defective_f1": float(f1_binary),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def metrics_for_trainer(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_binary_metrics(labels, preds)
