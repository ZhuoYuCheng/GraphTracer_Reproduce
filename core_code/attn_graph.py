from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


def build_step_token_map(offsets: List[Tuple[int, int]], spans: List[Tuple[int, int]]):
    step_tokens = [[] for _ in spans]
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for step_idx, (s_start, s_end) in enumerate(spans):
            if start >= s_start and start < s_end:
                step_tokens[step_idx].append(idx)
                break
    return step_tokens


def sparsify_weights(
    weights: torch.Tensor,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    renorm: bool = True,
) -> torch.Tensor:
    t = weights.size(0)
    pruned = torch.zeros_like(weights)
    for j in range(t):
        candidates = []
        for i in range(j):
            w = weights[i, j].item()
            if threshold is not None and w < threshold:
                continue
            if w > 0:
                candidates.append((i, w))
        if not candidates:
            continue
        if top_k is not None and top_k > 0:
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
        for i, w in candidates:
            pruned[i, j] = w
        if renorm:
            s = pruned[:j, j].sum()
            if s > 0:
                pruned[:j, j] = pruned[:j, j] / s
    return pruned


def backtrace_scores(
    weights: torch.Tensor,
    depth: int,
    dist_power: float = 0.0,
    mode: str = "sum",
) -> torch.Tensor:
    t = weights.size(0)
    scores = torch.zeros(t)
    if t == 0:
        return scores
    t_f = t - 1
    scores[t_f] = 1.0
    for i in range(t_f - 1, -1, -1):
        if depth is None or depth <= 0:
            j_max = t_f
        else:
            j_max = min(t_f, i + depth)
        if j_max <= i:
            continue
        contrib = None
        for j in range(i + 1, j_max + 1):
            w = weights[i, j]
            if dist_power != 0:
                w = w * ((j - i) ** dist_power)
            val = w * scores[j]
            if mode == "max":
                contrib = val if contrib is None else max(contrib, val)
            else:
                contrib = val if contrib is None else contrib + val
        scores[i] = contrib if contrib is not None else 0.0
    return scores


class Qwen3AttentionExtractor:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = 2048,
        last_n_layers: int = 4,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        head_agg: str = "mean",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.last_n_layers = last_n_layers
        self.head_agg = head_agg
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
        )
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def attention_matrix(
        self, text: str
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_attentions=True, output_hidden_states=False)
        attn = outputs.attentions
        if attn is None:
            raise RuntimeError(
                "Attentions are None. Ensure attn_implementation='eager' and that the model supports output_attentions."
            )
        hidden = outputs.last_hidden_state.squeeze(0).cpu()
        total_layers = len(attn)
        start = max(0, total_layers - self.last_n_layers)
        selected = attn[start:]
        stacked = torch.stack(selected, dim=0)
        layer_mean = stacked.mean(dim=0)  # B x H x S x S
        if self.head_agg == "max":
            attn_mean = layer_mean.max(dim=1).values
        else:
            attn_mean = layer_mean.mean(dim=1)
        attn_mean = attn_mean.squeeze(0)
        return attn_mean.cpu(), offsets, hidden


def build_soft_dependency_graph(
    attn_matrix: torch.Tensor,
    step_tokens: List[List[int]],
    top_k: Optional[int],
    threshold: Optional[float],
    renorm: bool,
    normalize_by_src_len: bool = False,
    agg: str = "mean",
    token_agg: str = "mean",
    token_topk: int = 5,
) -> torch.Tensor:
    t = len(step_tokens)
    weights = torch.zeros((t, t))
    for j in range(t):
        tok_j = step_tokens[j]
        if not tok_j:
            continue
        attn_j = attn_matrix[tok_j]
        for i in range(j):
            tok_i = step_tokens[i]
            if not tok_i:
                continue
            mass = attn_j[:, tok_i].sum(dim=1)
            if token_agg == "max":
                w = mass.max().item()
            elif token_agg == "topk":
                k = min(token_topk, mass.numel())
                w = mass.topk(k).values.mean().item()
            else:
                w = mass.mean().item()
            if normalize_by_src_len:
                w = w / max(len(tok_i), 1)
            weights[i, j] = w
    return sparsify_weights(weights, top_k=top_k, threshold=threshold, renorm=renorm)
