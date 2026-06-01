from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    variant: str = "detailed"


def normalize_code_whitespace(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in code.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def build_prompt(function_code: str, variant: str = "detailed", fewshot_block: str | None = None) -> str:
    function_code = normalize_code_whitespace(function_code)
    if variant == "short":
        header = (
            "Classify the following C function. "
            "Return exactly one label: defective or non-defective.\n"
        )
    elif variant == "strict":
        header = (
            "You are a binary code-defect classifier. Output must be exactly one of these two strings: "
            "defective, non-defective. Do not explain.\n"
        )
    else:
        header = (
            "You are a software quality assistant for C code. "
            "Decide whether the function may contain a defect. "
            "Use only the provided function body. "
            "Return exactly one label: defective or non-defective.\n"
        )

    fewshot = f"\n### Labeled Examples\n{fewshot_block}\n" if fewshot_block else ""
    return (
        f"### Instruction\n{header}"
        f"{fewshot}"
        f"\n### C Function\n```c\n{function_code}\n```\n\n"
        f"### Answer\n"
    )


def build_fewshot_block(examples: list[dict], text_column: str = "func", label_column: str = "target") -> str:
    chunks: list[str] = []
    for i, ex in enumerate(examples, start=1):
        label = "defective" if int(ex[label_column]) == 1 else "non-defective"
        code = normalize_code_whitespace(str(ex[text_column]))
        chunks.append(f"Example {i}\nFunction:\n```c\n{code}\n```\nLabel: {label}")
    return "\n\n".join(chunks)
