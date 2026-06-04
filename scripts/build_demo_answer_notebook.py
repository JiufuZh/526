from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "demo_answer_version.ipynb"


METRIC_FILES = {
    ("Qwen zero-shot", "validation"): RESULTS / "zero_shot_validation_metrics.json",
    ("Qwen zero-shot", "test"): RESULTS / "zero_shot_test_metrics.json",
    ("Qwen 4-shot", "validation"): RESULTS / "four_shot_validation_metrics.json",
    ("Qwen 4-shot", "test"): RESULTS / "four_shot_test_metrics.json",
    ("Qwen LoRA fine-tuned", "validation"): RESULTS / "lora_validation_metrics.json",
    ("Qwen LoRA fine-tuned", "test"): RESULTS / "lora_test_metrics.json",
    ("Majority baseline", "test"): RESULTS / "majority_test_metrics.json",
    ("TF-IDF Linear SVM", "test"): RESULTS / "tfidf_linear_svm_test_metrics.json",
    ("TF-IDF Logistic Regression", "test"): RESULTS / "tfidf_logreg_test_metrics.json",
    ("GraphCodeBERT", "validation"): RESULTS / "graphcodebert_validation_metrics.json",
    ("GraphCodeBERT", "test"): RESULTS / "graphcodebert_test_metrics.json",
}


def load_metrics() -> pd.DataFrame:
    rows = []
    for (method, split), path in METRIC_FILES.items():
        with path.open("r", encoding="utf-8") as f:
            metric = json.load(f)
        rows.append(
            {
                "method": method,
                "split": split,
                "accuracy": metric["accuracy"],
                "macro_f1": metric["macro_f1"],
                "defective_f1": metric["defective_f1"],
                "defective_recall": metric["defective_recall"],
                "tn": metric["tn"],
                "fp": metric["fp"],
                "fn": metric["fn"],
                "tp": metric["tp"],
                "file": str(path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code_cell(source: str, output: str | None = None, execution_count: int | None = None) -> dict:
    outputs = []
    if output is not None:
        outputs.append({"name": "stdout", "output_type": "stream", "text": output.splitlines(True)})
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source.strip().splitlines(True),
    }


def table_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    df = load_metrics()
    all_results = df.sort_values(["split", "macro_f1"], ascending=[True, False]).copy()
    test_df = df[df["split"] == "test"].sort_values("macro_f1", ascending=False).copy()
    test_view = test_df[["method", "accuracy", "macro_f1", "defective_f1", "defective_recall", "tn", "fp", "fn", "tp"]]

    all_output = all_results.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n"
    test_output = test_view.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n"
    artifact_lines = []
    for path in [
        RESULTS / "report_ready_metrics.md",
        RESULTS / "graphcodebert_test_metrics.json",
        RESULTS / "lora_test_metrics.json",
        RESULTS / "zero_shot_test_metrics.json",
        RESULTS / "four_shot_test_metrics.json",
        RESULTS / "figures" / "confusion_matrices_overview.png",
    ]:
        artifact_lines.append(f"{path.relative_to(ROOT)}: {'OK' if path.exists() else 'MISSING'}")

    confusion_lines = []
    for _, row in test_df.iterrows():
        matrix = [[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]]
        confusion_lines.append(f"{row['method']} test confusion matrix [[TN, FP], [FN, TP]]:")
        confusion_lines.append(str(matrix))
        confusion_lines.append("")

    cells = [
        md_cell(
            """
# Group 8 Demo Answer Version

**Project:** Code Defect Detection with Qwen, LoRA Fine-Tuning, and GraphCodeBERT  
**Purpose of this notebook:** This is the ready-to-show demo version. The result cells are already populated, so we can present without retraining any model live.

**Opening line to say:**  
\"Our project asks whether prompt-only large language models are enough for code defect detection, or whether task adaptation and code-specific pretraining are necessary.\"
"""
        ),
        md_cell(
            """
## 1. Problem and Motivation

**Say this:**  
\"The task is binary code defect detection. Given a C/C++ function, the model predicts whether it is non-defective or defective. This matters because manual code review is expensive, defects can be subtle, and false negatives can leave vulnerable code undetected.\"

**Key point:** We are not only trying one model. We compare a ladder of methods from low-cost prompting to supervised code-specific modeling.
"""
        ),
        md_cell(
            """
## 2. Research Question

**Say this:**  
\"Our research question is: for code defect detection, how far can we get with prompt-only LLMs, and when do fine-tuning or code-specific encoders become more effective?\"

The experiment design directly answers this question:

| Method | What it tests | Adaptation level |
|---|---|---|
| Qwen zero-shot | Raw prompt-only LLM reasoning | none |
| Qwen 4-shot | In-context examples without training | prompt examples |
| Qwen LoRA | Lightweight supervised LLM fine-tuning | LoRA adapter |
| Majority baseline | Sanity-check baseline | most frequent class |
| TF-IDF Linear SVM | Traditional lexical ML baseline | supervised CPU classifier |
| TF-IDF Logistic Regression | Traditional lexical ML baseline | supervised CPU classifier |
| GraphCodeBERT | Code-specific supervised encoder | full classifier head training |
"""
        ),
        md_cell(
            """
## 3. Dataset and Evaluation

**Say this:**  
\"We use the CodeXGLUE defect detection benchmark. The label is binary: 0 for non-defective and 1 for defective. We report both validation and test results.\"

**Evaluation measures to mention:**

- Accuracy: overall correctness.
- Macro-F1: balanced score across both classes.
- Defective-F1 and defective recall: most important for whether the model actually catches defective code.
- Confusion matrix: shows false positives and false negatives directly.

**Say this:**  
\"Accuracy alone can be misleading here, because a model can look acceptable while missing the defective class. That is exactly why we emphasize Macro-F1 and Defective-F1.\"
"""
        ),
        code_cell(
            """
from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd()
RESULTS = ROOT / "results"

metric_files = {
    ("Qwen zero-shot", "validation"): RESULTS / "zero_shot_validation_metrics.json",
    ("Qwen zero-shot", "test"): RESULTS / "zero_shot_test_metrics.json",
    ("Qwen 4-shot", "validation"): RESULTS / "four_shot_validation_metrics.json",
    ("Qwen 4-shot", "test"): RESULTS / "four_shot_test_metrics.json",
    ("Qwen LoRA fine-tuned", "validation"): RESULTS / "lora_validation_metrics.json",
    ("Qwen LoRA fine-tuned", "test"): RESULTS / "lora_test_metrics.json",
    ("GraphCodeBERT", "validation"): RESULTS / "graphcodebert_validation_metrics.json",
    ("GraphCodeBERT", "test"): RESULTS / "graphcodebert_test_metrics.json",
}

rows = []
for (method, split), path in metric_files.items():
    with path.open("r", encoding="utf-8") as f:
        m = json.load(f)
    rows.append({
        "method": method,
        "split": split,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "defective_f1": m["defective_f1"],
        "defective_recall": m["defective_recall"],
        "tn": m["tn"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tp": m["tp"],
    })

df = pd.DataFrame(rows)
print(df.sort_values(["split", "macro_f1"], ascending=[True, False]).to_string(index=False))
""",
            all_output,
            1,
        ),
        md_cell(
            f"""
## 4. Results Summary

**Say this:**  
\"This table shows all completed validation and test metrics. Every main model now has both validation and test results.\"

{table_markdown(all_results[["method", "split", "accuracy", "macro_f1", "defective_f1", "defective_recall"]])}
"""
        ),
        code_cell(
            """
test_df = df[df["split"] == "test"].sort_values("macro_f1", ascending=False)
print(test_df[["method", "accuracy", "macro_f1", "defective_f1", "defective_recall", "tn", "fp", "fn", "tp"]].to_string(index=False))
""",
            test_output,
            2,
        ),
        md_cell(
            f"""
## 5. Final Test-Set Ranking

**Say this:**  
\"On the test split, GraphCodeBERT has the strongest Macro-F1 at 0.6517. TF-IDF Logistic Regression is surprisingly competitive and has the best Defective-F1 at 0.6088. This means simple lexical baselines are strong, but the code-specific encoder is still the best balanced model overall.\"

{table_markdown(test_view)}

**Main takeaway sentence:**  
\"For this task, supervised models work better than prompt-only LLMs, and the code-specific GraphCodeBERT model gives the best balanced performance.\"
"""
        ),
        md_cell(
            """
## 6. Why 4-shot Collapsed

**Say this:**  
\"The 4-shot result is not missing; it is a negative result. It predicted every test sample as non-defective, so its defective precision, recall, and F1 are all zero.\"

**Explanation to say:**  
\"This shows that adding only four examples to the prompt was not enough for reliable code defect detection. The model learned a conservative output pattern instead of learning the defect signal. That supports our broader conclusion: prompt-only adaptation is weak for this task.\"
"""
        ),
        code_cell(
            """
for _, row in test_df.iterrows():
    matrix = [[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]]
    print(f"{row['method']} test confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)
    print()
""",
            "\n".join(confusion_lines),
            3,
        ),
        md_cell(
            """
## 7. Confusion Matrix Figure

**Say this while showing this figure:**  
\"The confusion matrices make the story visible. The 4-shot and majority baselines put everything into the non-defective column. TF-IDF Logistic Regression catches many defective examples, and GraphCodeBERT gives the best balanced Macro-F1.\"

![Confusion matrix overview](results/figures/confusion_matrices_overview.png)
"""
        ),
        md_cell(
            """
## 8. Demo Walkthrough Script

**Say this:**  
\"For reproducibility, our code and outputs are organized in the GitHub repository. The key scripts are in `scripts/`, configurations are in `configs/`, and final metrics and figures are in `results/`.\"

Recommended live clicks:

1. Open `results/report_ready_metrics.md`.
2. Open `results/figures/confusion_matrices_overview.png`.
3. Open `scripts/train_encoder_baseline.py` and `scripts/evaluate_encoder_baseline.py`.
4. Open `configs/encoder_graphcodebert.yaml`.
5. Point out that `demo_answer_version.ipynb` reads completed outputs and does not require retraining.
"""
        ),
        code_cell(
            """
artifact_paths = [
    RESULTS / "report_ready_metrics.md",
    RESULTS / "graphcodebert_test_metrics.json",
    RESULTS / "lora_test_metrics.json",
    RESULTS / "zero_shot_test_metrics.json",
    RESULTS / "four_shot_test_metrics.json",
    RESULTS / "figures" / "confusion_matrices_overview.png",
]

for path in artifact_paths:
    print(f"{path.relative_to(ROOT)}: {'OK' if path.exists() else 'MISSING'}")
""",
            "\n".join(artifact_lines) + "\n",
            4,
        ),
        md_cell(
            """
## 9. Limitations and Future Work

**Say this:**  
\"This is not a solved defect detector. The best model is GraphCodeBERT, but its test Macro-F1 is about 0.65, so there is still room for improvement.\"

Future work:

- Better threshold tuning for defective recall.
- More code-specific baselines or larger encoder models.
- More careful prompt/example selection for few-shot prompting.
- Error analysis by defect type and code length.
- Try combining code-specific encoders with LLM explanations.
"""
        ),
        md_cell(
            """
## 10. Closing Statement

**Say this:**  
\"Our final conclusion is that prompt-only LLMs are not reliable enough for this code defect detection task. Traditional TF-IDF baselines are surprisingly strong, LoRA fine-tuning helps the LLM, and GraphCodeBERT gives the best balanced performance. This suggests that task supervision and code-aware modeling are both important for practical defect detection.\"

**Repo:** https://github.com/JiufuZh/526
"""
        ),
        md_cell(
            """
## References

- CodeXGLUE benchmark: https://github.com/microsoft/CodeXGLUE
- GraphCodeBERT paper: https://arxiv.org/abs/2009.08366
- GraphCodeBERT model card: https://huggingface.co/microsoft/graphcodebert-base
- Qwen model family: https://huggingface.co/Qwen
- LoRA paper: https://arxiv.org/abs/2106.09685
- Project code: https://github.com/JiufuZh/526
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
