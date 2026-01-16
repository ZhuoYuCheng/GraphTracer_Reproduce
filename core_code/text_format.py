from typing import Dict


def format_step(step: Dict) -> str:
    role = step.get("role", "")
    name = step.get("name", "")
    content = step.get("content", "")
    header = role
    if name:
        header = f"{role} | {name}"
    return f"[Step] {header}\n{content}".strip()


def serialize_history(steps, max_chars_per_step=None):
    blocks = []
    for idx, step in enumerate(steps):
        role = step.get("role", "")
        name = step.get("name", "")
        content = step.get("content", "")
        if max_chars_per_step is not None and len(content) > max_chars_per_step:
            content = content[:max_chars_per_step]
        header = f"Step {idx} | Role {role}"
        if name:
            header = f"{header} | Name {name}"
        block = f"[{header}]\n{content}".strip()
        blocks.append(block)

    spans = []
    parts = []
    cursor = 0
    for block in blocks:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(block)
        cursor += len(block)
        end = cursor
        spans.append((start, end))

    return "".join(parts), spans
