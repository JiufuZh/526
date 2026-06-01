# Experiment Results Snapshot

This folder contains lightweight artifacts copied from the Tillicum run directory:

`/gpfs/projects/imt526a/group8/final`

Large model artifacts are intentionally not committed, including LoRA adapter weights,
checkpoints, cached datasets, and Hugging Face model files.

## Completed Results

| Experiment | Split | Accuracy | Macro-F1 | Defective-F1 | Notes |
|---|---:|---:|---:|---:|---|
| Qwen zero-shot | validation | 0.5201 | 0.5086 | 0.4332 | Frozen prompt baseline |
| Qwen zero-shot | test | 0.5263 | 0.5170 | 0.4825 | Frozen prompt baseline |
| Qwen 4-shot | validation | 0.5655 | 0.3612 | 0.0000 | Collapsed to non-defective predictions |
| Qwen 4-shot | test | 0.5406 | 0.3509 | 0.0000 | Collapsed to non-defective predictions |
| Qwen LoRA fine-tuned | validation | 0.5556 | 0.5509 | 0.5045 | Main fine-tuned LLM result |
| Qwen LoRA fine-tuned | test | 0.5523 | 0.5507 | 0.5236 | Main fine-tuned LLM result |

## GraphCodeBERT Attempt

The first GraphCodeBERT Slurm attempt ran as job `131579` and failed after about
52 seconds. The captured error log is in `results/logs/graphcodebert-131579.err`.
The tail of the error shows:

`RuntimeError: cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built.`

This means the GraphCodeBERT baseline still needs a rerun with a safer attention
backend or adjusted Torch/cuDNN settings before it can be used as a final baseline.

## Important Files

- `lora_validation_metrics.json` and `lora_test_metrics.json`: final LoRA metrics.
- `zero_shot_*_metrics.json`: zero-shot baseline metrics.
- `four_shot_*_metrics.json`: 4-shot baseline metrics.
- `report_ready_metrics.csv` and `report_ready_metrics.md`: compact report tables.
- `error_analysis_*_taxonomy.csv`: error-analysis summaries.
- `logs/graphcodebert-131579.err`: failed GraphCodeBERT run log.
