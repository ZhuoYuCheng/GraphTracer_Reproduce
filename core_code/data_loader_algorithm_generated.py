import json
import os
from typing import Dict, List


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def infer_mistake_index(mistake_step, history_len, subset, base_mode="auto"):
    ms = _safe_int(mistake_step)
    if ms is None:
        return None

    if base_mode == "0":
        return ms
    if base_mode == "1":
        if ms <= 0:
            return None
        return ms - 1

    # auto
    if ms < 0:
        return None
    if ms >= history_len:
        return history_len - 1
    return ms


def load_dataset(base_dir: str, subset: str, base_mode: str = "auto") -> List[Dict]:
    subset_dir = os.path.join(base_dir, subset)
    items = []
    for fname in sorted(os.listdir(subset_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(subset_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        history = data.get("history", [])
        history_norm = []
        for step in history:
            history_norm.append(
                {
                    "role": step.get("role", ""),
                    "name": step.get("name", ""),
                    "content": step.get("content", ""),
                }
            )

        mistake_idx = infer_mistake_index(
            data.get("mistake_step"), len(history_norm), subset, base_mode
        )

        items.append(
            {
                "id": data.get("question_ID", fname),
                "question": data.get("question", ""),
                "history": history_norm,
                "mistake_idx": mistake_idx,
                "mistake_agent": data.get("mistake_agent", ""),
                "subset": subset,
            }
        )
    return items
