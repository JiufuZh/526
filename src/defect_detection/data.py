from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from .constants import ID_TO_TEXT
from .prompts import build_prompt, normalize_code_whitespace


def function_hash(code: str) -> str:
    norm = normalize_code_whitespace(str(code))
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()


def load_or_download_dataset(cfg: Dict[str, Any]) -> DatasetDict:
    data_cfg = cfg["data"]
    cache_dir = Path(data_cfg.get("cache_dir", "data/processed/code_x_glue_cc_defect_detection"))
    if cache_dir.exists():
        return load_from_disk(str(cache_dir))

    ds = load_dataset(data_cfg["dataset_name"])
    text_col = data_cfg.get("text_column", "func")
    label_col = data_cfg.get("label_column", "target")

    def transform(ex):
        code = normalize_code_whitespace(ex[text_col]) if data_cfg.get("normalize_whitespace", True) else ex[text_col]
        ex[text_col] = code
        ex["label_text"] = ID_TO_TEXT[int(ex[label_col])]
        ex["function_sha1"] = function_hash(code)
        return ex

    ds = ds.map(transform)
    if data_cfg.get("deduplicate", False):
        ds = flag_exact_duplicates(ds, text_col=text_col)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    return ds


def flag_exact_duplicates(ds: DatasetDict, text_col: str = "func") -> DatasetDict:
    seen = set()

    def mark(ex):
        h = function_hash(ex[text_col])
        is_dup = h in seen
        seen.add(h)
        ex["is_exact_duplicate_seen_before"] = bool(is_dup)
        return ex

    out = DatasetDict()
    for split, split_ds in ds.items():
        out[split] = split_ds.map(mark)
    return out


def dataset_to_pandas(ds: Dataset, columns: Iterable[str] | None = None) -> pd.DataFrame:
    data = ds.to_pandas()
    if columns:
        keep = [c for c in columns if c in data.columns]
        return data[keep]
    return data


def balanced_subset(ds: Dataset, n_total: int, label_column: str = "target", seed: int = 42) -> Dataset:
    if n_total is None:
        return ds
    half = n_total // 2
    pos = ds.filter(lambda ex: int(ex[label_column]) == 1).shuffle(seed=seed).select(range(min(half, sum(ds[label_column]))))
    neg_all = ds.filter(lambda ex: int(ex[label_column]) == 0).shuffle(seed=seed)
    neg = neg_all.select(range(min(n_total - len(pos), len(neg_all))))
    return Dataset.from_pandas(pd.concat([pos.to_pandas(), neg.to_pandas()], ignore_index=True)).shuffle(seed=seed)


def fractional_subset(ds: Dataset, fraction: float, seed: int = 42) -> Dataset:
    if fraction is None or fraction >= 1.0:
        return ds
    n = max(1, int(len(ds) * fraction))
    return ds.shuffle(seed=seed).select(range(n))


class CausalDefectDataset(TorchDataset):
    def __init__(self, hf_dataset: Dataset, tokenizer, cfg: Dict[str, Any], split: str):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.text_col = cfg["data"].get("text_column", "func")
        self.label_col = cfg["data"].get("label_column", "target")
        self.max_length = int(cfg["model"].get("max_length", 512))
        self.prompt_variant = cfg["model"].get("prompt_variant", "detailed")
        self.split = split

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        ex = self.ds[int(idx)]
        prompt = build_prompt(str(ex[self.text_col]), variant=self.prompt_variant)
        class_label = int(ex[self.label_col])
        answer = ID_TO_TEXT[class_label]
        eos = self.tokenizer.eos_token or ""
        answer_text = answer + eos
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False).input_ids
        max_prompt_len = max(1, self.max_length - len(answer_ids))
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_len,
        ).input_ids
        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + answer_ids
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "class_labels": class_label}


def make_data_collator(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        class_labels = [int(item["class_labels"]) for item in batch if "class_labels" in item]
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if len(class_labels) == len(batch):
            out["class_labels"] = torch.tensor(class_labels, dtype=torch.long)
        return out

    return collate
