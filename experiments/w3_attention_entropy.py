#!/usr/bin/env python3
"""
W3: Attention Entropy Analysis — Quantitative Mechanism Analysis
================================================================
Reviewer W3: "The analysis of why retrieval works is largely qualitative."

This script provides QUANTITATIVE evidence by measuring attention entropy
in PatchTST at different context lengths (336, 720, 3000).

WHAT IT MEASURES:
  - Attention entropy per head, per layer
  - Entropy across context lengths → shows if attention "dilutes" at longer context
  - Effective attention rank (how many patches genuinely attend)
  - Attention distance distribution (where does each patch look?)

WHAT THESE MEAN: See W3_CONCEPTS_EXPLAINED.md

USAGE:
  python w3_attention_entropy.py                     # All 3 seq_lens
  python w3_attention_entropy.py --seq_lens 336 720  # Just 2

Requires trained PatchTST checkpoints from corrected experiments.
If checkpoints don't exist, runs with random weights (still shows structural patterns).
"""

import os
import sys
import json
import time
import math
import numpy as np
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAFT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
sys.path.insert(0, RAFT_ROOT)
os.chdir(RAFT_ROOT)

import torch
import torch.nn as nn
from data_provider.data_factory import data_provider


# ════════════════════════════════════════════════════════════════════════════════
# PatchTST with attention output enabled
# ════════════════════════════════════════════════════════════════════════════════

def build_patchtst_with_attention(args):
    """
    Build PatchTST model with output_attention=True so we can extract
    attention weight matrices from each layer.
    """
    # Temporarily enable attention output
    args.output_attention = True

    from layers.Transformer_EncDec import Encoder, EncoderLayer
    from layers.SelfAttention_Family import FullAttention, AttentionLayer
    from layers.Embed import PatchEmbedding

    class PatchTSTWithAttention(nn.Module):
        def __init__(self, configs, patch_len=16, stride=8):
            super().__init__()
            self.seq_len = configs.seq_len
            self.pred_len = configs.pred_len
            padding = stride

            self.patch_embedding = PatchEmbedding(
                configs.d_model, patch_len, stride, padding, configs.dropout)

            self.encoder = Encoder(
                [EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=True),  # ← KEY: enabled
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)],
                norm_layer=nn.LayerNorm(configs.d_model),  # operates on last dim [B*C, L, d_model]
            )

        def get_attention_maps(self, x_enc):
            """
            Run forward pass and return attention maps from each layer.
            Input:  x_enc [B, seq_len, channels]
            Output: list of attention maps, one per layer
                    each map: [B*channels, n_heads, n_patches, n_patches]
            """
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

            x_enc = x_enc.permute(0, 2, 1)  # [B, C, seq_len]
            enc_out, n_vars = self.patch_embedding(x_enc)  # [B*C, n_patches, d_model]
            enc_out, attns = self.encoder(enc_out)  # attns: list of [B*C, H, L, L] or None

            return attns

    model = PatchTSTWithAttention(args)
    return model


# ════════════════════════════════════════════════════════════════════════════════
# Entropy & Analysis Functions
# ════════════════════════════════════════════════════════════════════════════════

def attention_entropy(attn_weights, eps=1e-10):
    """
    Compute Shannon entropy of attention distributions.

    Input:  attn_weights [B, H, L, L] — attention probabilities (sum to 1 over last dim)
    Output: entropy [B, H, L] — entropy for each query position

    Entropy = -sum(p * log(p))
    Max entropy = log(L)  (uniform attention)
    Min entropy = 0       (attend to single token)
    """
    # Clamp to avoid log(0)
    p = attn_weights.clamp(min=eps)
    entropy = -(p * p.log()).sum(dim=-1)  # [B, H, L]
    return entropy


def normalized_entropy(attn_weights, eps=1e-10):
    """
    Entropy / log(L) — normalized to [0, 1].
    1.0 = uniform attention (diffuse/confused)
    0.0 = fully focused
    """
    L = attn_weights.shape[-1]
    ent = attention_entropy(attn_weights, eps)
    return ent / math.log(L)


def effective_attention_rank(attn_weights, threshold=0.01):
    """
    Count how many keys each query attends to with weight > threshold.
    This measures "how spread out" the attention is.

    Output: [B, H, L] number of attended positions per query
    """
    return (attn_weights > threshold).float().sum(dim=-1)


def attention_distance(attn_weights):
    """
    Compute mean attention distance — how far away does each query look?

    For position i attending to position j, distance = |i - j|.
    Mean weighted distance tells us if patches attend locally or globally.

    Output: mean_distance [B, H, L]
    """
    B, H, L, _ = attn_weights.shape
    positions = torch.arange(L, device=attn_weights.device).float()
    # Distance matrix: |i - j|
    dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # [L, L]
    # Weighted mean distance
    mean_dist = (attn_weights * dist_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, H, L]
    return mean_dist


# ════════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════════

class BaseArgs:
    task_name = 'long_term_forecast'
    is_training = 0
    model = 'PatchTST'
    data = 'ETTh1'
    root_path = os.path.join(RAFT_ROOT, 'data', 'ETT')
    data_path = 'ETTh1.csv'
    features = 'M'
    target = 'OT'
    freq = 'h'
    checkpoints = os.path.join(RAFT_ROOT, 'checkpoints')
    label_len = 48
    pred_len = 96
    seasonal_patterns = 'Monthly'
    inverse = False
    enc_in = 7
    dec_in = 7
    c_out = 7
    d_model = 16
    n_heads = 4
    e_layers = 3
    d_layers = 1
    d_ff = 128
    patch_len = 16
    stride = 8
    num_workers = 0
    itr = 1
    train_epochs = 100
    batch_size = 32
    patience = 20
    learning_rate = 0.0001
    loss = 'MSE'
    lradj = 'TST'
    pct_start = 0.3
    use_amp = False
    dropout = 0.3
    fc_dropout = 0.3
    head_dropout = 0.0
    moving_avg = 25
    factor = 1
    distil = True
    embed = 'timeF'
    activation = 'gelu'
    output_attention = True
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0'
    augmentation_ratio = 0
    p_hidden_dims = [128, 256]
    p_hidden_layers = 2


# ════════════════════════════════════════════════════════════════════════════════
# Main Analysis
# ════════════════════════════════════════════════════════════════════════════════

def analyze_seq_len(seq_len, n_batches=10):
    """Run attention entropy analysis for a given seq_len. Returns summary dict."""

    args = BaseArgs()
    args.seq_len = seq_len
    n_patches = int((seq_len - args.patch_len) / args.stride + 2)

    print(f"\n  seq_len={seq_len} → {n_patches} patches")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, e_layers={args.e_layers}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = build_patchtst_with_attention(args).to(device)

    # Try loading checkpoint
    ckpt_path = os.path.join(
        args.checkpoints,
        f'PatchTST_{seq_len}_CORRECTED_{args.data}_sl{seq_len}_pl{args.pred_len}',
        'checkpoint.pth'
    )
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            # Try to load, ignore mismatched keys (architecture might differ slightly)
            model.load_state_dict(state, strict=False)
            print(f"  Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"  Could not load checkpoint ({e}), using random weights")
    else:
        print(f"  No checkpoint found, using random weights (structural analysis only)")

    model.eval()

    # Load test data
    test_data, test_loader = data_provider(args, 'test')
    print(f"  Test set: {len(test_data)} samples")

    # Collect attention stats
    layer_entropies = defaultdict(list)       # layer_idx → list of [H] mean entropies
    layer_norm_entropies = defaultdict(list)
    layer_eff_ranks = defaultdict(list)
    layer_distances = defaultdict(list)

    batch_count = 0
    with torch.no_grad():
        for _, batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            if batch_count >= n_batches:
                break

            batch_x = batch_x.float().to(device)
            attns = model.get_attention_maps(batch_x)

            for layer_idx, attn in enumerate(attns):
                if attn is None:
                    continue
                # attn: [B*C, H, L, L]
                ent = attention_entropy(attn)        # [B*C, H, L]
                nent = normalized_entropy(attn)       # [B*C, H, L]
                erank = effective_attention_rank(attn) # [B*C, H, L]
                dist = attention_distance(attn)       # [B*C, H, L]

                # Average over batch and query positions → per-head stats
                layer_entropies[layer_idx].append(ent.mean(dim=(0, 2)).cpu().numpy())
                layer_norm_entropies[layer_idx].append(nent.mean(dim=(0, 2)).cpu().numpy())
                layer_eff_ranks[layer_idx].append(erank.mean(dim=(0, 2)).cpu().numpy())
                layer_distances[layer_idx].append(dist.mean(dim=(0, 2)).cpu().numpy())

            batch_count += 1
            print(f"    batch {batch_count}/{n_batches}", end='\r')

    print(f"    Processed {batch_count} batches")

    # Aggregate
    summary = {
        'seq_len': seq_len,
        'n_patches': n_patches,
        'max_entropy': round(math.log(n_patches), 4),
        'layers': {},
    }

    for layer_idx in sorted(layer_entropies.keys()):
        ent = np.mean(layer_entropies[layer_idx], axis=0)       # [H]
        nent = np.mean(layer_norm_entropies[layer_idx], axis=0)
        erank = np.mean(layer_eff_ranks[layer_idx], axis=0)
        dist = np.mean(layer_distances[layer_idx], axis=0)

        layer_stats = {
            'entropy_per_head': [round(float(e), 4) for e in ent],
            'mean_entropy': round(float(ent.mean()), 4),
            'normalized_entropy_per_head': [round(float(e), 4) for e in nent],
            'mean_normalized_entropy': round(float(nent.mean()), 4),
            'effective_rank_per_head': [round(float(e), 1) for e in erank],
            'mean_effective_rank': round(float(erank.mean()), 1),
            'attention_distance_per_head': [round(float(d), 2) for d in dist],
            'mean_attention_distance': round(float(dist.mean()), 2),
        }
        summary['layers'][f'layer_{layer_idx}'] = layer_stats

        print(f"\n    Layer {layer_idx}:")
        print(f"      Entropy:     {ent.mean():.4f} (normalized: {nent.mean():.4f})")
        print(f"      Eff. rank:   {erank.mean():.1f} / {n_patches} patches")
        print(f"      Attn. dist:  {dist.mean():.2f} patches")

    # Overall summary
    all_nent = [summary['layers'][k]['mean_normalized_entropy']
                for k in summary['layers']]
    all_erank = [summary['layers'][k]['mean_effective_rank']
                 for k in summary['layers']]
    all_dist = [summary['layers'][k]['mean_attention_distance']
                for k in summary['layers']]

    summary['overall'] = {
        'mean_normalized_entropy': round(float(np.mean(all_nent)), 4),
        'mean_effective_rank': round(float(np.mean(all_erank)), 1),
        'mean_attention_distance': round(float(np.mean(all_dist)), 2),
    }

    del model
    torch.cuda.empty_cache()

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='W3 Attention Entropy Analysis')
    parser.add_argument('--seq_lens', type=int, nargs='+', default=[336, 720, 3000],
                        help='Sequence lengths to analyze')
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of test batches to analyze per seq_len')
    args = parser.parse_args()

    print('═' * 80)
    print('  W3: Attention Entropy Analysis')
    print('  "Quantitative mechanism analysis of why retrieval works"')
    print('═' * 80)
    print(f'  Time:     {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Device:   {"CUDA" if torch.cuda.is_available() else "CPU"}')
    print(f'  seq_lens: {args.seq_lens}')
    print('═' * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_summaries = {}
    t0 = time.time()

    for sl in args.seq_lens:
        print(f'\n{"─" * 80}')
        print(f'  Analyzing seq_len={sl}')
        print(f'{"─" * 80}')
        summary = analyze_seq_len(sl, n_batches=args.n_batches)
        all_summaries[str(sl)] = summary

    elapsed = time.time() - t0

    # ── Cross-context comparison ──────────────────────────────────────────────
    print('\n' + '═' * 80)
    print('  CROSS-CONTEXT COMPARISON')
    print('═' * 80)
    print(f"\n  {'seq_len':>8s}  {'patches':>8s}  {'norm_ent':>10s}  {'eff_rank':>10s}  {'attn_dist':>10s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")

    for sl_str in sorted(all_summaries.keys(), key=int):
        s = all_summaries[sl_str]
        o = s['overall']
        print(f"  {s['seq_len']:>8d}  {s['n_patches']:>8d}  "
              f"{o['mean_normalized_entropy']:>10.4f}  "
              f"{o['mean_effective_rank']:>10.1f}  "
              f"{o['mean_attention_distance']:>10.2f}")

    # Interpretation
    sl_keys = sorted(all_summaries.keys(), key=int)
    if len(sl_keys) >= 2:
        short = all_summaries[sl_keys[0]]
        long = all_summaries[sl_keys[-1]]
        ent_change = (long['overall']['mean_normalized_entropy'] -
                      short['overall']['mean_normalized_entropy'])
        rank_change = (long['overall']['mean_effective_rank'] -
                       short['overall']['mean_effective_rank'])

        print(f"\n  INTERPRETATION:")
        if ent_change > 0.05:
            print(f"    Normalized entropy INCREASES (+{ent_change:.4f}) with context length.")
            print(f"    → Attention becomes more diffuse/uniform at longer context.")
            print(f"    → Supports our claim: more patches = harder to focus on relevant ones.")
        elif ent_change < -0.05:
            print(f"    Normalized entropy DECREASES ({ent_change:.4f}) with context length.")
            print(f"    → Attention becomes MORE focused at longer context.")
        else:
            print(f"    Normalized entropy is stable (Δ={ent_change:.4f}).")
            print(f"    → Attention focus doesn't change much with context length.")

        if rank_change > 5:
            print(f"    Effective rank grows (+{rank_change:.1f}) — attention spreads over more patches.")
        print(f"\n    Context scaling: {short['seq_len']}→{long['seq_len']} "
              f"({short['n_patches']}→{long['n_patches']} patches)")

    # Save
    output = {
        'purpose': 'W3: Quantitative attention analysis',
        'question': 'Does attention become more diffuse at longer context lengths?',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': round(elapsed, 1),
        'results': all_summaries,
    }
    out_path = os.path.join(RESULTS_DIR, 'w3_attention_entropy.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved → {out_path}")
    print(f"  Total time: {round(elapsed, 1)}s")
    print('═' * 80)


if __name__ == '__main__':
    main()
