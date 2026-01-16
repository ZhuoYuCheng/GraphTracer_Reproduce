from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer


class Qwen3Embedder:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 4) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            masked = last_hidden * mask
            denom = mask.sum(dim=1).clamp(min=1)
            emb = masked.sum(dim=1) / denom
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)
