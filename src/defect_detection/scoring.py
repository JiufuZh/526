from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .constants import ID_TO_TEXT


@torch.no_grad()
def score_two_labels(model, tokenizer, prompts: List[str], max_prompt_length: int = 512) -> List[int]:
    """Choose the class with lower negative log-likelihood over the label tokens."""
    model.eval()
    device = next(model.parameters()).device
    predictions: List[int] = []

    for prompt in prompts:
        scores: Dict[int, float] = {}
        for label_id, label_text in ID_TO_TEXT.items():
            answer_ids = tokenizer(label_text + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
            max_prompt_len = max(1, max_prompt_length - len(answer_ids))
            prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_prompt_len).input_ids
            input_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            shifted = input_ids[:, 1:]
            label_start = max(0, len(prompt_ids) - 1)
            target_logits = logits[:, label_start:, :]
            target_ids = shifted[:, label_start:]
            log_probs = F.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            scores[label_id] = float(-token_log_probs.mean().detach().cpu())
        predictions.append(min(scores, key=scores.get))
    return predictions
