#!/bin/bash
# =============================================================================
# RAFT: Main Experiments — RAFT vs PatchTST on ETTh1
# =============================================================================
# Reproduces Table 1 from the paper.
# Requires: GPU with >= 8GB VRAM (T4, RTX 2080, etc.)
# Estimated time: ~2-4 hours total on T4

set -e

echo "============================================================"
echo "RAFT Main Experiments: ETTh1, pred_len=96"
echo "============================================================"

# --- RAFT (seq_len=720) ---
echo ""
echo "[1/3] RAFT with seq_len=720 (our method)"
python run.py \
    --model RAFT \
    --model_id RAFT_720 \
    --data ETTh1 \
    --root_path ./data/ETT/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 720 \
    --label_len 48 \
    --pred_len 96 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --dropout 0.1 \
    --train_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj type1

# --- PatchTST (seq_len=720, corrected hyperparameters) ---
echo ""
echo "[2/3] PatchTST with seq_len=720 (corrected config)"
python run.py \
    --model PatchTST \
    --model_id PatchTST_720 \
    --data ETTh1 \
    --root_path ./data/ETT/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 720 \
    --label_len 48 \
    --pred_len 96 \
    --d_model 16 \
    --n_heads 4 \
    --e_layers 3 \
    --d_layers 1 \
    --d_ff 128 \
    --dropout 0.3 \
    --train_epochs 100 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --lradj TST

# --- TransformerLongContext (seq_len=3000) ---
echo ""
echo "[3/3] Vanilla Transformer with seq_len=3000 (long context baseline)"
python run_long_context.py \
    --model TransformerLongContext \
    --model_id TransformerLC_3000 \
    --data ETTh1 \
    --root_path ./data/ETT/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 3000 \
    --label_len 48 \
    --pred_len 96 \
    --d_model 128 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 512 \
    --dropout 0.15 \
    --train_epochs 30 \
    --batch_size 24 \
    --learning_rate 0.0001

echo ""
echo "============================================================"
echo "All main experiments complete."
echo "Results saved to: result_long_term_forecast.txt"
echo "                  result_long_context_forecast.txt"
echo "============================================================"
