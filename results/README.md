# Experiment Results Snapshot

This folder contains lightweight artifacts copied from the Tillicum run directory:

`/gpfs/projects/imt526a/group8/final`

Large model artifacts are intentionally not committed, including LoRA adapter weights,
checkpoints, cached datasets, and Hugging Face model files.

## Completed Results

| Experiment | Split | Accuracy | Macro-F1 | Defective-F1 | Notes |
|---|---:|---:|---:|---:|---|
| Qwen zero-shot | validation | 0.5201 | 0.5086 | 0.4332 | Frozen prompt baseline |
| Qwen zero-shot | test | 0.5264 | 0.5180 | 0.4545 | Frozen prompt baseline |
| Qwen 4-shot | validation | 0.5655 | 0.3612 | 0.0000 | Collapsed to non-defective predictions |
| Qwen 4-shot | test | 0.5406 | 0.3509 | 0.0000 | Collapsed to non-defective predictions |
| Qwen LoRA fine-tuned | validation | 0.5556 | 0.5509 | 0.5045 | Main fine-tuned LLM result |
| Qwen LoRA fine-tuned | test | 0.5523 | 0.5507 | 0.5236 | Main fine-tuned LLM result |
| TF-IDF Linear SVM | test | 0.6007 | 0.5994 | 0.5766 | Strong lexical CPU baseline |
| TF-IDF Logistic Regression | test | 0.6223 | 0.6218 | 0.6088 | Strongest traditional CPU baseline |
| GraphCodeBERT | validation | 0.6618 | 0.6513 | 0.5908 | Strongest validation result |
| GraphCodeBERT | test | 0.6589 | 0.6517 | 0.6017 | Strongest test result |

## GraphCodeBERT Baseline

The initial GraphCodeBERT attempts exposed three implementation issues that were
fixed before the final run:

- H200/cuDNN attention incompatibility, fixed by forcing eager attention and disabling cuDNN SDPA.
- Binary classification loss mismatch, fixed by forcing `single_label_classification`.
- Boolean labels entering cross entropy, fixed by casting labels to `long` in the trainer loss path.

The final GraphCodeBERT run completed successfully and produced validation and
test metrics. It is the strongest baseline in the current result matrix.

## Important Files

- `lora_validation_metrics.json` and `lora_test_metrics.json`: final LoRA metrics.
- `zero_shot_*_metrics.json`: zero-shot baseline metrics.
- `four_shot_*_metrics.json`: 4-shot baseline metrics.
- `graphcodebert_*_metrics.json`: GraphCodeBERT validation/test metrics.
- `tfidf_*_test_metrics.json` and `majority_test_metrics.json`: traditional CPU baseline test metrics.
- `report_ready_metrics.csv` and `report_ready_metrics.md`: compact report tables.
- `error_analysis_*_taxonomy.csv`: error-analysis summaries.
- `logs/graphcodebert-131579.err`: first failed GraphCodeBERT run log kept for provenance.
