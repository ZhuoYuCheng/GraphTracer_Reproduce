# GraphTracer_Reproduce

Reproduction-oriented code for attention-based dependency backtracing on the Who&When dataset, with a lightweight alternative to explicit code-graph parsing. The goal is to localize failure steps (When) and responsible agents (Who) using attention-derived soft dependency graphs and small scoring heads.

This repo keeps Hand-Crafted and Algorithm-Generated subsets isolated where they differ (notably agent identity: role-based vs name-based) so improvements on one subset do not affect the other.

## Repository Structure

- `core_code/`: main pipeline
  - `run_mvp.py`: training/evaluation entrypoint
  - `attn_graph.py`: attention extraction + soft dependency graph + backtrace
  - `data_loader.py`: Hand-Crafted loader (role-based)
  - `data_loader_algorithm_generated.py`: Algorithm-Generated loader (name-based)
  - `text_format.py`: step serialization
  - `README_Attention_Backtrace.md`: detailed method and params
- `Who&When/`: dataset directory
  - `Hand-Crafted/`: 58 JSON files
  - `Algorithm-Generated/`: 126 JSON files
- `model/`: local model checkpoint (e.g., Qwen3-8B)
- `summary.md`: filtered run log with best-performing configs

## Environment

- Python 3.10+ (tested with 3.12)
- PyTorch + CUDA
- `transformers`

Install deps (minimal):

```bash
pip install torch transformers
```

## Model Setup

This project uses Qwen3 (e.g., `Qwen3-8B`). You can download it once into:

```
/root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B
```

The code uses `--model_path` to point to the local folder. First run will load weights from disk; no remote download is required if the folder is populated.

## Dataset Format (Who&When)

Each JSON has keys like:

- `history`: list of steps with `role`, `name`, `content`
- `mistake_step`: step index (string)
- `mistake_agent`: responsible agent

Key difference:
- Hand-Crafted uses **role-based** agent labels (mapped to 4 roles).
- Algorithm-Generated uses **name-based** agent labels (82 unique expert names).

The code keeps these isolated:
- Hand-Crafted: role aggregation
- Algorithm-Generated: name aggregation

## Quickstart

Set the memory allocator to reduce fragmentation on GPU:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Hand-Crafted (best known)

```bash
python core_code/run_mvp.py \
  --data_dir /root/autodl-tmp/GraphTracer_Reproduce/Who\&When \
  --subset Hand-Crafted \
  --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
  --torch_dtype float16 \
  --max_length 512 \
  --max_steps 16 \
  --max_chars_per_step 400 \
  --normalize_by_src_len \
  --last_n_layers 2 \
  --top_k 5 \
  --depth 6 \
  --renorm \
  --train_epochs 10 \
  --train_lr 5e-4 \
  --split_ratio 0.5 \
  --agent_loss_weight 0.5 \
  --model_type gnn \
  --agg mean \
  --score_mix 5.0 \
  --step_pred_mode agent_sim
```

### Algorithm-Generated (meets target thresholds)

```bash
python core_code/run_mvp.py \
  --data_dir /root/autodl-tmp/GraphTracer_Reproduce/Who\&When \
  --subset Algorithm-Generated \
  --model_path /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B \
  --torch_dtype float16 \
  --max_length 512 \
  --max_steps 16 \
  --max_chars_per_step 400 \
  --normalize_by_src_len \
  --last_n_layers 2 \
  --top_k 5 \
  --depth 6 \
  --renorm \
  --train_epochs 10 \
  --train_lr 5e-4 \
  --split_ratio 0.5 \
  --agent_loss_weight 0.7 \
  --model_type seq \
  --agg mean \
  --score_mix 0 \
  --step_pred_mode model
```

## Notes

- `--embed_extra` defaults to `auto` (adds `scores+pos` for Algorithm-Generated only; no change for Hand-Crafted).
- If you change Algorithm-Generated logic, keep it isolated from Hand-Crafted to preserve baseline stability.
- Full parameter explanations: `core_code/README_Attention_Backtrace.md`
- Best runs and reasoning: `summary.md`

## Reproduce Paper-aligned Guidance

This repo follows the teacher’s guidance: use attention-weighted dependency backtrace instead of code-parsing graphs while retaining the “dependency tracing” idea.

## License

For model weights, follow the original Qwen3 license in `model/Qwen3-8B/`.
