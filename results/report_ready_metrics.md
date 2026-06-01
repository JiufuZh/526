| source_file | accuracy | macro_f1 | defective_f1 | defective_recall | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outputs/baselines/majority_validation_metrics.json | 0.5655 | 0.3612 | 0.0000 | 0.0000 | 1545 | 0 | 1187 | 0 |
| outputs/baselines/tfidf_linear_svm_validation_metrics.json | 0.6182 | 0.6144 | 0.5762 | 0.5973 | 980 | 565 | 478 | 709 |
| outputs/baselines/tfidf_logreg_validation_metrics.json | 0.6464 | 0.6441 | 0.6158 | 0.6521 | 992 | 553 | 413 | 774 |
| outputs/qwen25_7b_lora_bf16_512/test_metrics.json | 0.5523 | 0.5507 | 0.5236 | 0.5355 | 837 | 640 | 583 | 672 |
| outputs/qwen25_7b_lora_bf16_512/validation_metrics.json | 0.5556 | 0.5509 | 0.5045 | 0.5206 | 900 | 645 | 569 | 618 |
| outputs/smoke_lora_512/validation_metrics.json | 0.4788 | 0.4706 | 0.5365 | 0.6942 | 484 | 1061 | 363 | 824 |
