# Fine-Tuned LLM for Function-Level Code Defect Detection

This repository implements the Group 8 defect-detection project pipeline:

- CodeXGLUE / Devign function-level defect dataset loading
- Split verification, label-distribution checks, duplicate checks, and token-length profiling
- Majority, TF-IDF + Logistic Regression, and TF-IDF + Linear SVM baselines
- Zero-shot and 4-shot Qwen2.5-7B-Instruct evaluation
- Qwen2.5-7B-Instruct LoRA / QLoRA supervised fine-tuning
- GraphCodeBERT-style encoder baseline
- Validation/test evaluation with Macro-F1, accuracy, precision, recall, and confusion matrix
- Ablation support for max length, prompt format, and train-set fraction
- Qualitative false-positive / false-negative inspection

## Quick start

```bash
conda create -n defect-llm python=3.11 -y
conda activate defect-llm
pip install -r requirements.txt
pip install -e .
```

## Local CPU baselines

```bash
python scripts/prepare_data.py --config configs/default.yaml
python scripts/profile_tokens.py --config configs/default.yaml
python scripts/run_baselines.py --config configs/default.yaml
```

## H200 LoRA smoke test

```bash
python scripts/train_sft.py --config configs/smoke_lora_512.yaml
python scripts/evaluate_llm.py --config configs/smoke_lora_512.yaml --split validation --adapter outputs/smoke_lora_512/final_adapter
```

## Main SFT run

```bash
python scripts/train_sft.py --config configs/qlora_512.yaml
python scripts/evaluate_llm.py --config configs/qlora_512.yaml --split validation --adapter outputs/qwen25_7b_qlora_512/final_adapter
```

## Encoder baseline

```bash
python scripts/train_encoder_baseline.py --config configs/encoder_graphcodebert.yaml
```

## Error analysis

```bash
python scripts/analyze_errors.py \
  --dataset-cache data/processed/code_x_glue_cc_defect_detection \
  --predictions outputs/qwen25_7b_qlora_512/validation_predictions.csv \
  --split validation \
  --output-dir outputs/qwen25_7b_qlora_512/error_analysis
```

## SLURM examples

```bash
sbatch slurm/run_smoke_lora.slurm
sbatch slurm/run_main_qlora_512.slurm
sbatch slurm/run_eval_validation.slurm
```

## Expected project folders

```text
configs/       YAML experiment configs
scripts/       command-line entry points
src/           reusable package code
slurm/         Tillicum / Hyak job templates
data/          raw and processed local dataset cache
outputs/       metrics, predictions, adapters, and run logs
docs/          method-design document
```
