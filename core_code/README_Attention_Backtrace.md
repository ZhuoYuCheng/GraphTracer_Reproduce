# Attention Backtrace (Qwen3)

Pipeline for Who&When using Qwen3 attention weights to build a soft dependency
graph and backtrace scores.

## Run

```bash
cd /root/autodl-tmp/GraphTracer_Reproduce/core_code
python run_mvp.py \
  --data_dir /root/autodl-tmp/GraphTracer_Reproduce/Who\&When \
  --subset Hand-Crafted \
  --model_path Qwen/Qwen3-8B \
  --model_cache_dir /root/.cache/huggingface \
  --torch_dtype float16 \
  --max_length 512 \
  --max_steps 12 \
  --max_chars_per_step 400 \
  --normalize_by_src_len \
  --top_k 3 \
  --depth 5 \
  --renorm \
  --train_epochs 3 \
  --train_lr 1e-3 \
  --split_ratio 0.2
```

Notes:
- `--model_path` can be a local Qwen3 checkpoint directory or a Hugging Face repo id (e.g. `Qwen/Qwen3-8B`). If downloading on first run, ensure at least 20-30GB free space on the data disk (total size is 50GB).
- `--model_cache_dir` controls where the model is cached; use a path with enough free space.
- Use `--subset all` to combine Hand-Crafted and Algorithm-Generated.
- `--mistake_base_mode` can be `auto`, `0`, or `1` to control mistake-step indexing.
- `--last_n_layers` controls which attention layers are averaged (default: last 4).
- If GPU OOM occurs, reduce `--max_length`, lower `--last_n_layers`, or set `--max_steps` to truncate long histories.
- `--max_chars_per_step` limits per-step text length to keep more steps within the token budget.
- Training is optional. Set `--train_epochs` > 0 to fit a lightweight step scorer on attention-derived features.
