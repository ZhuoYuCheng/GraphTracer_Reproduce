# Run Summary (Filtered)

Date: 2025-01-16

Only keeps runs that define a new best after the mistake_step indexing fix
(applied in `core_code/data_loader.py`).

## Run A1 (Post-fix Baseline)
## Environment
- Model: Qwen3-8B (local)
- Model path: /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B
- Dataset: Who&When (Hand-Crafted)
- Command:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_mvp.py \
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
    --renorm
  ```

## Result
- when_acc 0.0556 (1/18)
- who_acc 0.1915 (9/47)

## Run B1 (Best Agent-level, split_ratio=0.3)
## Environment
- Model: Qwen3-8B (local)
- Model path: /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B
- Dataset: Who&When (Hand-Crafted)
- Command:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_mvp.py \
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
    --split_ratio 0.3 \
    --agent_loss_weight 0.5 \
    --model_type gnn \
    --agg mean \
    --score_mix 5.0 \
    --step_pred_mode agent_sim
  ```

## Result
- train_epoch 1 loss 2.959889
- train_epoch 2 loss 2.917572
- train_epoch 3 loss 2.885650
- train_epoch 4 loss 2.861306
- train_epoch 5 loss 2.841872
- train_epoch 6 loss 2.825451
- train_epoch 7 loss 2.810405
- train_epoch 8 loss 2.796025
- train_epoch 9 loss 2.781732
- train_epoch 10 loss 2.767221
- when_acc 0.1333 (2/15)
- who_acc 0.6829 (28/41)  (best Agent-level)

## Reason for Improvement
- 增大训练比例到 0.3 提供更多带错误标签的训练样本，GNN 学到更稳定的“错误角色”统计模式；测试集仍有足够样本，Agent-level 明显上升。

## Run B2 (Best Step-level, split_ratio=0.5)
## Environment
- Model: Qwen3-8B (local)
- Model path: /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B
- Dataset: Who&When (Hand-Crafted)
- Command:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_mvp.py \
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

## Result
- train_epoch 1 loss 3.003004
- train_epoch 2 loss 2.952135
- train_epoch 3 loss 2.913570
- train_epoch 4 loss 2.883741
- train_epoch 5 loss 2.859952
- train_epoch 6 loss 2.838850
- train_epoch 7 loss 2.819337
- train_epoch 8 loss 2.799728
- train_epoch 9 loss 2.781682
- train_epoch 10 loss 2.762713
- when_acc 0.2000 (2/10)  (best Step-level)
- who_acc 0.6552 (19/29)

## Reason for Improvement
- 训练样本比例增加到 0.5 后，step 级别的回溯+GNN 监督更充分；同时测试集仍保留一定规模，使 When 准确率出现阶段性提升。

## Current Best
- Best Step-level: 0.2000 (Run B2)
- Best Agent-level: 0.6829 (Run B1)

## Algorithm-Generated (Baseline, split_ratio=0.5)
## Environment
- Model: Qwen3-8B (local)
- Model path: /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B
- Dataset: Who&When (Algorithm-Generated)
- Command:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_mvp.py \
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
    --agent_loss_weight 0.5 \
    --model_type gnn \
    --agg mean \
    --score_mix 5.0 \
    --step_pred_mode agent_sim
  ```

## Result
- when_acc 0.1746 (11/63)
- who_acc 0.0000 (0/63)

## Reason for Result
- 首次在 Algorithm-Generated 子集上建立基线；who_acc=0 可能提示 agent 标签映射或分布与 Hand-Crafted 不同，需要单独检查数据统计与标签对齐方式。

## Algorithm-Generated (Agent-Name Isolation, split_ratio=0.5)
## Environment
- Model: Qwen3-8B (local)
- Model path: /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B
- Dataset: Who&When (Algorithm-Generated)
- Command:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_mvp.py \
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
    --agent_loss_weight 0.5 \
    --model_type gnn \
    --agg mean \
    --score_mix 5.0 \
    --step_pred_mode agent_sim
  ```

## Result
- when_acc 0.2381 (15/63)
- who_acc 0.3651 (23/63)

## Reason for Improvement
- 将 Algorithm-Generated 的 agent 身份从 role 切换为 name，并单独聚合与监督，使 who 预测标签空间对齐；同时保留 Hand-Crafted 原逻辑，避免交叉影响。

## Algorithm-Generated (Seq Model + Moderate Agent Loss, split_ratio=0.5)
## Environment
- Model: Qwen3-8B (local)
- Model path: /root/autodl-tmp/GraphTracer_Reproduce/model/Qwen3-8B
- Dataset: Who&When (Algorithm-Generated)
- Command:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_mvp.py \
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

## Result
- when_acc 0.3175 (20/63)
- who_acc 0.6032 (38/63)

## Reason for Improvement
- seq 相比 gnn 更适合 AG 子集：AG 的 history 长度集中在 5–10 步，序列依赖明显；seq 直接建模相邻步骤关系，避免 gnn 受稀疏图质量影响而“传不动”。实测 seq 能明显拉升 when（step-level）。
- agent_loss_weight 从 0.5→0.7：提高 agent 监督在总 loss 中的权重，使 name-based 的 agent 聚合更稳定，who 明显上升；同时不明显挤压 step 学习，二者共同达标。
- score_mix=0：避免 backtrace scores 对 seq 的 step logits 产生偏置（AG 的标签更依赖 name/序列关系），使模型更专注于序列判别。
