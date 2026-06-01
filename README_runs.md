# Reproducible Run Order

1. `prepare_data.py`: downloads/stages CodeXGLUE defect detection and saves local processed splits.
2. `profile_tokens.py`: computes P50/P90/max length and 256/512/1024 coverage using the selected tokenizer.
3. `run_baselines.py`: trains majority, TF-IDF/LR, and TF-IDF/SVM baselines.
4. `train_encoder_baseline.py`: optional stronger code-pretrained encoder baseline.
5. `run_zero_fewshot.py`: evaluates frozen Qwen2.5-7B-Instruct with zero-shot and 4-shot prompts.
6. `train_sft.py`: trains LoRA/QLoRA adapter using supervised label-token loss.
7. `evaluate_llm.py`: evaluates base or adapter model by two-label log-prob scoring or generation.
8. `analyze_errors.py`: samples false positives and false negatives for qualitative inspection.
9. `make_report_tables.py`: merges metrics into report-ready CSV and Markdown tables.

Recommended final-test discipline: use the validation set for all development decisions, then run the held-out test split once using the selected best configuration.
