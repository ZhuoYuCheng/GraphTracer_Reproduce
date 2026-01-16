import argparse
import random

import torch
import torch.nn as nn

from attn_graph import (
    Qwen3AttentionExtractor,
    backtrace_scores,
    build_soft_dependency_graph,
    build_step_token_map,
)
from data_loader import load_dataset as load_dataset_handcrafted, normalize_role
from data_loader_algorithm_generated import (
    load_dataset as load_dataset_algorithm_generated,
)
from text_format import serialize_history


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_agent_mode(subset: str) -> str:
    if subset == "Algorithm-Generated":
        return "name"
    return "role4"


def normalize_agent_key(key: str, agent_mode: str) -> str:
    if agent_mode == "name":
        return (key or "").strip().lower()
    return normalize_role(key).lower()


def get_agent_keys(history, agent_mode: str):
    if agent_mode == "name":
        return [step.get("name", "") for step in history]
    return [step.get("role", "") for step in history]


def agent_match(agent_key: str, step_key: str, agent_mode: str) -> bool:
    if not agent_key:
        return False
    if agent_mode == "name":
        return agent_key == normalize_agent_key(step_key, agent_mode)
    return agent_key in (step_key or "").lower()


def load_dataset_by_subset(base_dir: str, subset: str, base_mode: str):
    if subset == "Algorithm-Generated":
        return load_dataset_algorithm_generated(base_dir, subset, base_mode)
    return load_dataset_handcrafted(base_dir, subset, base_mode)


class StepScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class GraphScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.lin_self1 = nn.Linear(in_dim, hidden_dim)
        self.lin_in1 = nn.Linear(in_dim, hidden_dim)
        self.lin_out1 = nn.Linear(in_dim, hidden_dim)
        self.lin_self2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_in2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h = self.lin_self1(x) + w.t() @ self.lin_in1(x) + w @ self.lin_out1(x)
        h = torch.relu(h)
        h = self.lin_self2(h) + w.t() @ self.lin_in2(h) + w @ self.lin_out2(h)
        h = torch.relu(h)
        return self.out(h).squeeze(-1)


class EmbeddingScorer(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class SequenceScorer(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x))
        h = self.encoder(h.unsqueeze(0)).squeeze(0)
        return self.out(h).squeeze(-1)

def compute_step_embeddings(hidden: torch.Tensor, step_tokens):
    embs = []
    for toks in step_tokens:
        if toks:
            embs.append(hidden[toks].mean(dim=0))
        else:
            embs.append(torch.zeros(hidden.size(1)))
    return torch.stack(embs, dim=0)

def build_step_features(
    weights: torch.Tensor,
    scores: torch.Tensor,
    step_tokens,
    roles,
    sim_to_final: torch.Tensor,
    dist_to_final: torch.Tensor,
) -> torch.Tensor:
    t = weights.size(0)
    if t == 0:
        return torch.empty((0, 14))
    in_sum = weights.sum(dim=0)
    out_sum = weights.sum(dim=1)
    in_deg = (weights > 0).sum(dim=0).to(torch.float32)
    out_deg = (weights > 0).sum(dim=1).to(torch.float32)
    pos = torch.arange(t, dtype=torch.float32) / max(t - 1, 1)
    tok_lens = torch.tensor([len(x) for x in step_tokens], dtype=torch.float32)
    tok_lens = tok_lens / max(tok_lens.max().item(), 1.0)
    in_deg = in_deg / max(in_deg.max().item(), 1.0)
    out_deg = out_deg / max(out_deg.max().item(), 1.0)
    role_flags = []
    for role in roles:
        r = (role or "").lower()
        role_flags.append(
            [
                1.0 if "websurfer" in r else 0.0,
                1.0 if "orchestrator" in r else 0.0,
                1.0 if "assistant" in r else 0.0,
                1.0 if "filesurfer" in r else 0.0,
            ]
        )
    role_flags = torch.tensor(role_flags, dtype=torch.float32)

    feats = torch.stack([scores, in_sum, out_sum, pos, tok_lens, in_deg, out_deg], dim=1)
    feats = torch.cat([feats, role_flags], dim=1)
    to_final = weights[:, -1].unsqueeze(1)
    feats = torch.cat([feats, to_final], dim=1)
    feats = torch.cat([feats, sim_to_final.unsqueeze(1), dist_to_final.unsqueeze(1)], dim=1)
    return feats


def build_embed_features(
    step_embs: torch.Tensor,
    scores: torch.Tensor,
    agent_mode: str,
    embed_extra: str,
) -> torch.Tensor:
    if embed_extra == "auto":
        embed_extra = "scores_pos" if agent_mode == "name" else "none"

    if embed_extra == "none":
        return step_embs

    extras = []
    if "scores" in embed_extra:
        extras.append(scores.unsqueeze(1))
    if "pos" in embed_extra:
        t = step_embs.size(0)
        pos = torch.arange(t, dtype=torch.float32) / max(t - 1, 1)
        extras.append(pos.unsqueeze(1))
    if not extras:
        return step_embs
    return torch.cat([step_embs, *extras], dim=1)


def evaluate(
    records,
    extractor,
    top_k,
    threshold,
    renorm,
    depth,
    max_steps,
    max_chars_per_step,
    normalize_by_src_len,
    model,
    device,
    agent_loss_weight,
    model_type,
    agg,
    score_mix,
    pred_offset,
    step_pred_mode,
    score_mode,
    direct_weight,
    dist_power,
    token_agg,
    token_topk,
    backtrace_mode,
    embed_extra,
    agent_agg,
):
    correct_when = 0
    total_when = 0
    correct_who = 0
    total_who = 0

    for rec in records:
        agent_mode = get_agent_mode(rec.get("subset", "Hand-Crafted"))
        history = rec["history"]
        mistake_idx = rec["mistake_idx"]
        if max_steps is not None and len(history) > max_steps:
            offset = len(history) - max_steps
            history = history[offset:]
            if mistake_idx is not None:
                if mistake_idx < offset:
                    mistake_idx = None
                else:
                    mistake_idx -= offset

        text, spans = serialize_history(history, max_chars_per_step=max_chars_per_step)
        attn_matrix, offsets, hidden = extractor.attention_matrix(text)
        step_tokens = build_step_token_map(offsets, spans)
        step_embs = compute_step_embeddings(hidden, step_tokens)
        final_emb = step_embs[-1] if step_embs.numel() > 0 else torch.zeros(1)
        sim = nn.functional.cosine_similarity(
            step_embs, final_emb.unsqueeze(0), dim=1
        )
        dist = (step_embs - final_emb).pow(2).sum(dim=1).sqrt()
        dist = dist / max(dist.max().item(), 1.0)
        weights = build_soft_dependency_graph(
            attn_matrix,
            step_tokens,
            top_k=top_k,
            threshold=threshold,
            renorm=renorm,
            normalize_by_src_len=normalize_by_src_len,
            agg=agg,
            token_agg=token_agg,
            token_topk=token_topk,
        )
        back_scores = backtrace_scores(
            weights, depth, dist_power=dist_power, mode=backtrace_mode
        )
        direct_scores = weights[:, -1] if weights.numel() > 0 else back_scores
        if score_mode == "direct":
            scores = direct_scores
        elif score_mode == "hybrid":
            scores = (1 - direct_weight) * back_scores + direct_weight * direct_scores
        else:
            scores = back_scores
        if scores.numel() == 0:
            continue
        agent_keys = get_agent_keys(history, agent_mode)
        if model is None:
            pred_idx = int(torch.argmax(scores[:-1]).item()) if scores.numel() > 1 else 0
            pred_agent = normalize_agent_key(agent_keys[pred_idx], agent_mode)
        else:
            roles = [step.get("role", "") for step in history]
            if model_type == "gnn":
                feats = build_step_features(
                    weights, scores, step_tokens, roles, sim, dist
                ).to(device)
                logits = model(feats, weights.to(device))
            elif model_type in ("embed", "seq"):
                feats = build_embed_features(
                    step_embs, scores, agent_mode, embed_extra
                ).to(device)
                logits = model(feats.float())
            else:
                feats = build_step_features(
                    weights, scores, step_tokens, roles, sim, dist
                ).to(device)
                logits = model(feats)
            if score_mix != 0:
                combined = logits + score_mix * scores.to(device)
            else:
                combined = logits
            if step_pred_mode == "scores":
                pred_idx = int(torch.argmax(scores[:-1]).item()) if scores.numel() > 1 else 0
            elif step_pred_mode == "sim":
                pred_idx = int(torch.argmax(sim[:-1]).item()) if sim.numel() > 1 else 0
            elif step_pred_mode == "direct":
                pred_idx = int(torch.argmax(direct_scores[:-1]).item()) if direct_scores.numel() > 1 else 0
            else:
                pred_idx = int(torch.argmax(combined[:-1]).item()) if combined.numel() > 1 else 0
            pred_agent, _, _ = aggregate_agent_logits(
                combined, agent_keys, device, agent_mode, agent_agg
            )
            if step_pred_mode in ("agent_scores", "agent_sim", "agent_direct"):
                agent_key = pred_agent
                best_idx = None
                best_val = None
                if step_pred_mode == "agent_sim":
                    vec = sim
                elif step_pred_mode == "agent_direct":
                    vec = direct_scores
                else:
                    vec = scores
                for idx, key in enumerate(agent_keys):
                    if agent_match(agent_key, key, agent_mode):
                        val = vec[idx].item()
                        if best_val is None or val > best_val:
                            best_val = val
                            best_idx = idx
                if best_idx is not None:
                    pred_idx = best_idx

        pred_idx = max(0, min(pred_idx + pred_offset, len(history) - 1))

        if mistake_idx is not None:
            total_when += 1
            if pred_idx == mistake_idx:
                correct_when += 1

        pred_role = normalize_agent_key(pred_agent, agent_mode)
        gt_role = normalize_agent_key(rec["mistake_agent"], agent_mode)
        if gt_role:
            total_who += 1
            if pred_role.lower() == gt_role.lower():
                correct_who += 1

    when_acc = correct_when / max(total_when, 1)
    who_acc = correct_who / max(total_who, 1)
    print(f"when_acc {when_acc:.4f} ({correct_when}/{total_when})")
    print(f"who_acc {who_acc:.4f} ({correct_who}/{total_who})")


def aggregate_agent_logits(
    logits: torch.Tensor,
    agent_keys,
    device,
    agent_mode: str,
    agent_agg: str,
):
    if agent_mode == "name":
        key_to_idx = {}
        key_logits = []
        for idx, key in enumerate(agent_keys):
            norm = normalize_agent_key(key, agent_mode)
            if not norm:
                continue
            if norm not in key_to_idx:
                key_to_idx[norm] = len(key_logits)
                key_logits.append(logits[idx])
            else:
                j = key_to_idx[norm]
                if agent_agg == "max":
                    key_logits[j] = torch.max(key_logits[j], logits[idx])
                else:
                    key_logits[j] = torch.logsumexp(
                        torch.stack([key_logits[j], logits[idx]]), dim=0
                    )
        if not key_logits:
            role_keys = [f"step_{i}" for i in range(logits.size(0))]
            role_logits = logits
        else:
            role_keys = list(key_to_idx.keys())
            role_logits = torch.stack(key_logits)
        pred_idx = int(torch.argmax(role_logits).item()) if role_logits.numel() else 0
        pred_key = role_keys[pred_idx] if role_keys else ""
        return pred_key, role_logits, role_keys

    role_keys = ["websurfer", "orchestrator", "assistant", "filesurfer"]
    role_map = {k: i for i, k in enumerate(role_keys)}
    role_logits = torch.full((len(role_keys),), -1e9, device=device)
    for idx, role in enumerate(agent_keys):
        r = (role or "").lower()
        for key, rid in role_map.items():
            if key in r:
                role_logits[rid] = torch.logsumexp(
                    torch.stack([role_logits[rid], logits[idx]]), dim=0
                )
                break
    pred_idx = int(torch.argmax(role_logits).item())
    return role_keys[pred_idx], role_logits, role_keys


def train_step_scorer(
    records,
    extractor,
    top_k,
    threshold,
    renorm,
    depth,
    max_steps,
    max_chars_per_step,
    normalize_by_src_len,
    device,
    epochs,
    lr,
    agent_loss_weight,
    model_type,
    agg,
    score_mode,
    direct_weight,
    dist_power,
    token_agg,
    token_topk,
    backtrace_mode,
    embed_extra,
    agent_agg,
):
    if model_type == "gnn":
        model = GraphScorer(14).to(device)
    elif model_type in ("embed", "seq"):
        model = None
    else:
        model = StepScorer(14).to(device)
    optimizer = None

    samples = []
    for rec in records:
        agent_mode = get_agent_mode(rec.get("subset", "Hand-Crafted"))
        history = rec["history"]
        mistake_idx = rec["mistake_idx"]
        if max_steps is not None and len(history) > max_steps:
            offset = len(history) - max_steps
            history = history[offset:]
            if mistake_idx is not None:
                if mistake_idx < offset:
                    mistake_idx = None
                else:
                    mistake_idx -= offset

        if mistake_idx is None:
            continue

        text, spans = serialize_history(history, max_chars_per_step=max_chars_per_step)
        attn_matrix, offsets, hidden = extractor.attention_matrix(text)
        step_tokens = build_step_token_map(offsets, spans)
        step_embs = compute_step_embeddings(hidden, step_tokens)
        final_emb = step_embs[-1] if step_embs.numel() > 0 else torch.zeros(1)
        sim = nn.functional.cosine_similarity(
            step_embs, final_emb.unsqueeze(0), dim=1
        )
        dist = (step_embs - final_emb).pow(2).sum(dim=1).sqrt()
        dist = dist / max(dist.max().item(), 1.0)
        weights = build_soft_dependency_graph(
            attn_matrix,
            step_tokens,
            top_k=top_k,
            threshold=threshold,
            renorm=renorm,
            normalize_by_src_len=normalize_by_src_len,
            agg=agg,
            token_agg=token_agg,
            token_topk=token_topk,
        )
        back_scores = backtrace_scores(
            weights, depth, dist_power=dist_power, mode=backtrace_mode
        )
        direct_scores = weights[:, -1] if weights.numel() > 0 else back_scores
        if score_mode == "direct":
            scores = direct_scores
        elif score_mode == "hybrid":
            scores = (1 - direct_weight) * back_scores + direct_weight * direct_scores
        else:
            scores = back_scores
        roles = [step.get("role", "") for step in history]
        agent_keys = get_agent_keys(history, agent_mode)
        if model_type in ("embed", "seq"):
            feats = build_embed_features(step_embs, scores, agent_mode, embed_extra)
        else:
            feats = build_step_features(weights, scores, step_tokens, roles, sim, dist)
        if feats.numel() == 0:
            continue
        samples.append((feats, mistake_idx, roles, agent_keys, agent_mode, weights))
        if model_type == "embed" and model is None:
            model = EmbeddingScorer(feats.size(1)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if model_type == "seq" and model is None:
            model = SequenceScorer(feats.size(1)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if model is not None and optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not samples:
        print("no training samples with mistake labels; skip training")
        return model

    if model is None:
        print("no training samples with mistake labels; skip training")
        return model
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for feats, mistake_idx, roles, agent_keys, agent_mode, weights in samples:
            feats = feats.to(device)
            if model_type == "gnn":
                logits = model(feats, weights.to(device))
            elif model_type in ("embed", "seq"):
                logits = model(feats.float())
            else:
                logits = model(feats)
            loss = nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([mistake_idx], device=device)
            )
            if agent_loss_weight > 0:
                if agent_mode == "name":
                    _, role_logits, role_keys = aggregate_agent_logits(
                        logits, agent_keys, device, agent_mode, agent_agg
                    )
                    gt_key = normalize_agent_key(agent_keys[mistake_idx], agent_mode)
                    if gt_key in role_keys:
                        gt = role_keys.index(gt_key)
                        agent_loss = nn.functional.cross_entropy(
                            role_logits.unsqueeze(0),
                            torch.tensor([gt], device=device),
                        )
                        loss = loss + agent_loss_weight * agent_loss
                else:
                    _, role_logits, _ = aggregate_agent_logits(
                        logits, roles, device, agent_mode, agent_agg
                    )
                    gt_role = roles[mistake_idx]
                    gt = -1
                    for idx, key in enumerate(
                        ["websurfer", "orchestrator", "assistant", "filesurfer"]
                    ):
                        if key in (gt_role or "").lower():
                            gt = idx
                            break
                    if gt >= 0:
                        agent_loss = nn.functional.cross_entropy(
                            role_logits.unsqueeze(0),
                            torch.tensor([gt], device=device),
                        )
                        loss = loss + agent_loss_weight * agent_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        avg = total_loss / max(count, 1)
        print(f"train_epoch {epoch+1} loss {avg:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--subset", default="Hand-Crafted")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_cache_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_chars_per_step", type=int, default=600)
    parser.add_argument("--normalize_by_src_len", action="store_true")
    parser.add_argument("--agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--renorm", action="store_true")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--last_n_layers", type=int, default=4)
    parser.add_argument("--mistake_base_mode", default="auto", choices=["auto", "0", "1"])
    parser.add_argument("--train_epochs", type=int, default=0)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--agent_loss_weight", type=float, default=0.5)
    parser.add_argument(
        "--model_type", choices=["mlp", "gnn", "embed", "seq"], default="mlp"
    )
    parser.add_argument("--score_mix", type=float, default=0.0)
    parser.add_argument("--pred_offset", type=int, default=0)
    parser.add_argument(
        "--step_pred_mode",
        choices=["model", "scores", "sim", "direct", "agent_scores", "agent_sim", "agent_direct"],
        default="model",
    )
    parser.add_argument("--score_mode", choices=["backtrace", "direct", "hybrid"], default="backtrace")
    parser.add_argument("--direct_weight", type=float, default=0.5)
    parser.add_argument("--dist_power", type=float, default=0.0)
    parser.add_argument("--head_agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--token_agg", choices=["mean", "max", "topk"], default="mean")
    parser.add_argument("--token_topk", type=int, default=5)
    parser.add_argument("--backtrace_mode", choices=["sum", "max"], default="sum")
    parser.add_argument(
        "--embed_extra",
        choices=["auto", "none", "scores", "scores_pos"],
        default="auto",
    )
    parser.add_argument(
        "--agent_agg",
        choices=["logsumexp", "max"],
        default="logsumexp",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    subsets = [args.subset]
    if args.subset == "all":
        subsets = ["Hand-Crafted", "Algorithm-Generated"]

    all_records = []
    for subset in subsets:
        all_records.extend(
            load_dataset_by_subset(args.data_dir, subset, base_mode=args.mistake_base_mode)
        )

    random.shuffle(all_records)
    split = int(len(all_records) * args.split_ratio)
    train_records = all_records[:split]
    test_records = all_records[split:]

    if args.torch_dtype == "float16":
        dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = None

    extractor = Qwen3AttentionExtractor(
        model_name_or_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        last_n_layers=args.last_n_layers,
        cache_dir=args.model_cache_dir,
        torch_dtype=dtype,
        head_agg=args.head_agg,
    )

    model = None
    if args.train_epochs > 0:
        model = train_step_scorer(
            train_records,
            extractor,
            top_k=args.top_k,
            threshold=args.threshold,
            renorm=args.renorm,
            depth=args.depth,
            max_steps=args.max_steps,
            max_chars_per_step=args.max_chars_per_step,
            normalize_by_src_len=args.normalize_by_src_len,
            device=extractor.device,
            epochs=args.train_epochs,
            lr=args.train_lr,
            agent_loss_weight=args.agent_loss_weight,
            model_type=args.model_type,
            agg=args.agg,
            score_mode=args.score_mode,
            direct_weight=args.direct_weight,
            dist_power=args.dist_power,
            token_agg=args.token_agg,
            token_topk=args.token_topk,
            backtrace_mode=args.backtrace_mode,
            embed_extra=args.embed_extra,
            agent_agg=args.agent_agg,
        )

    evaluate(
        test_records,
        extractor,
        top_k=args.top_k,
        threshold=args.threshold,
        renorm=args.renorm,
        depth=args.depth,
        max_steps=args.max_steps,
        max_chars_per_step=args.max_chars_per_step,
        normalize_by_src_len=args.normalize_by_src_len,
        model=model,
        device=extractor.device,
        agent_loss_weight=args.agent_loss_weight,
        model_type=args.model_type,
        agg=args.agg,
        score_mix=args.score_mix,
        pred_offset=args.pred_offset,
        step_pred_mode=args.step_pred_mode,
        score_mode=args.score_mode,
        direct_weight=args.direct_weight,
        dist_power=args.dist_power,
        token_agg=args.token_agg,
        token_topk=args.token_topk,
        backtrace_mode=args.backtrace_mode,
        embed_extra=args.embed_extra,
        agent_agg=args.agent_agg,
    )


if __name__ == "__main__":
    main()
