# Reproducible Run Order

1. `prepare_data.py`: downloads/stages CodeXGLUE defect detection and saves local processed splits.
2. `profile_tokens.py`: computes P50/P90/max length and 256/512/1024 coverage using the selected tokenizer.
3. `run_baselines.py`: trains majority, TF-IDF/LR, and TF-IDF/SVM baselines. Run once for validation and once with `--eval-split test` for the final held-out CPU-baseline results.
4. `train_encoder_baseline.py`: optional stronger code-pretrained encoder baseline.
5. `run_zero_fewshot.py`: evaluates frozen Qwen2.5-7B-Instruct with zero-shot and 4-shot prompts.
6. `train_sft.py`: trains LoRA/QLoRA adapter using supervised label-token loss.
7. `evaluate_llm.py`: evaluates base or adapter model by two-label log-prob scoring or generation.
8. `analyze_errors.py`: samples false positives and false negatives for qualitative inspection.
9. `make_report_tables.py`: merges metrics into report-ready CSV and Markdown tables.

Recommended final-test discipline: use the validation set for all development decisions, then run the held-out test split once using the selected best configuration.

## Completed CPU Baseline Test Results

| Baseline | Accuracy | Macro-F1 | Defective-F1 | Defective recall |
|---|---:|---:|---:|---:|
| Majority baseline | 0.5406 | 0.3509 | 0.0000 | 0.0000 |
| TF-IDF + Linear SVM | 0.6007 | 0.5994 | 0.5766 | 0.5920 |
| TF-IDF + Logistic Regression | 0.6223 | 0.6218 | 0.6088 | 0.6398 |

These three test metrics are saved under `results/majority_test_metrics.json`, `results/tfidf_linear_svm_test_metrics.json`, and `results/tfidf_logreg_test_metrics.json`.
