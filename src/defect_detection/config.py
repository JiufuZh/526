from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("base", None)
    if base_ref:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_path if (path.parent / base_path).exists() else base_path
        base_cfg = load_yaml(base_path)
        cfg = deep_update(base_cfg, cfg)
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> Path:
    output_root = Path(cfg.get("project", {}).get("output_root", "outputs"))
    run_name = cfg.get("training", {}).get("run_name") or cfg.get("encoder", {}).get("run_name") or "run"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
