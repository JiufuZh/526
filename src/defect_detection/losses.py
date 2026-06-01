from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer


class OptionalLossTrainer(Trainer):
    def __init__(self, *args, focal_loss_gamma=None, label_smoothing=0.0, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_gamma = focal_loss_gamma
        self.custom_label_smoothing = float(label_smoothing or 0.0)
        self.class_weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        class_labels = inputs.pop("class_labels", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
            label_smoothing=self.custom_label_smoothing if self.focal_loss_gamma is None else 0.0,
        ).view_as(shift_labels)
        valid = shift_labels.ne(-100)

        if self.focal_loss_gamma is not None:
            pt = torch.exp(-token_loss.detach())
            token_loss = ((1 - pt) ** float(self.focal_loss_gamma)) * token_loss

        sample_loss = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        if self.class_weights is not None and class_labels is not None:
            weights = self.class_weights.to(sample_loss.device)[class_labels.to(sample_loss.device)]
            loss = (sample_loss * weights).sum() / weights.sum().clamp_min(1e-12)
        else:
            loss = sample_loss.mean()
        return (loss, outputs) if return_outputs else loss
