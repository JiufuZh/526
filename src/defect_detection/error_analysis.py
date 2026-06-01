from __future__ import annotations

import re
from typing import Dict


def summarize_code_features(code: str) -> Dict[str, int]:
    code = code or ""
    return {
        "line_count": len(code.splitlines()),
        "branch_count": len(re.findall(r"\b(if|else|switch|case|for|while|do)\b", code)),
        "return_count": len(re.findall(r"\breturn\b", code)),
        "pointer_ops": code.count("*") + code.count("->"),
        "array_accesses": len(re.findall(r"\[[^\]]*\]", code)),
        "null_checks": len(re.findall(r"\bNULL\b|nullptr|==\s*0|!=\s*0", code)),
        "malloc_free_calls": len(re.findall(r"\b(malloc|calloc|realloc|free)\s*\(", code)),
        "string_api_calls": len(re.findall(r"\b(strcpy|strncpy|strcat|sprintf|snprintf|memcpy|memmove)\s*\(", code)),
    }
