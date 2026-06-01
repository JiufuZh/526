from __future__ import annotations

import argparse

from defect_detection.config import load_yaml
from defect_detection.label_norm import normalize_label
from defect_detection.llm_utils import load_causal_lm, load_tokenizer
from defect_detection.prompts import build_prompt
from defect_detection.scoring import score_two_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--code-file", required=True)
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    code = open(args.code_file, "r", encoding="utf-8").read()
    tokenizer = load_tokenizer(cfg["model"].get("tokenizer_name", cfg["model"]["base_model"]))
    model = load_causal_lm(cfg, adapter_path=args.adapter)
    prompt = build_prompt(code, variant=cfg["model"].get("prompt_variant", "detailed"))
    pred = score_two_labels(model, tokenizer, [prompt], max_prompt_length=int(cfg["model"].get("max_length", 512)))[0]
    print("defective" if pred == 1 else "non-defective")


if __name__ == "__main__":
    main()
