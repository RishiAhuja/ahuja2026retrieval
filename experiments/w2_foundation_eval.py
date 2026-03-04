#!/usr/bin/env python3
"""
W2: Foundation Model Zero-Shot Evaluation (Chronos + Moirai)
=============================================================
Reviewer asked: "Have you evaluated on Moirai or Chronos?"

KEY FIX: Chronos recommends pred_len <= 64. For H=96, we use autoregressive
rolling: predict 64 steps, append to context, predict remaining 32 steps.
We ONLY test H=96 (matching our main experiments) since longer horizons
are outside Chronos's design range.

MODELS:
  - Chronos-T5-Small (Amazon, T5-based, univariate)
  - Moirai-1.1-R-Small (Salesforce, any-variate, masked encoder)

Both are ZERO-SHOT — no training, just inference on ETTh1 test set.

INSTALL:
  pip install chronos-forecasting
  pip install uni2ts einops

USAGE:
  python w2_foundation_eval.py --model chronos    # ~15-30 min on T4
  python w2_foundation_eval.py --model moirai     # ~15-30 min on T4
  python w2_foundation_eval.py --model all         # Both
"""

import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAFT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
sys.path.insert(0, RAFT_ROOT)

import torch
# Pre-import torchvision early to prevent circular import when
# uni2ts triggers: lightning → torchmetrics → torchvision (partially init'd)
try:
    import torchvision  # noqa: F401
except Exception:
    pass


# ════════════════════════════════════════════════════════════════════════════════
# Data Loading (same splits as our experiments)
# ════════════════════════════════════════════════════════════════════════════════

def load_etth1_normalized():
    """Load ETTh1 with same normalization as our experiments."""
    from sklearn.preprocessing import StandardScaler

    csv_path = os.path.join(RAFT_ROOT, 'data', 'ETT', 'ETTh1.csv')
    df = pd.read_csv(csv_path)
    cols = list(df.columns[1:])  # ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    data = df[cols].values.astype(np.float32)

    # Same borders as Dataset_ETT_hour
    TRAIN_END = 12 * 30 * 24   # 8640
    VAL_END   = TRAIN_END + 4 * 30 * 24  # 11520
    TEST_END  = TRAIN_END + 8 * 30 * 24  # 17280

    scaler = StandardScaler()
    scaler.fit(data[:TRAIN_END])
    data_normed = scaler.transform(data).astype(np.float32)

    return data_normed, TRAIN_END, VAL_END, TEST_END, cols


def load_etth1_raw():
    """Load ETTh1 raw (for Moirai which handles normalization internally)."""
    csv_path = os.path.join(RAFT_ROOT, 'data', 'ETT', 'ETTh1.csv')
    df = pd.read_csv(csv_path)
    cols = list(df.columns[1:])
    data = df[cols].values.astype(np.float32)

    TRAIN_END = 8640
    VAL_END   = 11520
    TEST_END  = 17280

    return data, TRAIN_END, VAL_END, TEST_END, cols


def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ════════════════════════════════════════════════════════════════════════════════
# Chronos Evaluation
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_chronos(context_lengths=[336, 720], pred_len=96, max_samples=200):
    """
    Evaluate Chronos zero-shot on ETTh1 test set.

    Chronos is univariate: we run each of the 7 channels independently (same
    channel-independence approach as PatchTST) and average metrics.

    For pred_len=96: we do autoregressive rolling — predict 64, then 32.
    This avoids the quality degradation Chronos warns about for pred_len > 64.
    """
    try:
        from chronos import ChronosPipeline
    except (ImportError, Exception):
        print("  Chronos not found — attempting auto-install...")
        import subprocess
        # Install chronos + upgrade transformers (requires >= 4.38)
        for pkg in ['--upgrade transformers', 'chronos-forecasting']:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install'] + pkg.split(),
                capture_output=True, text=True
            )
            status = 'OK' if result.returncode == 0 else 'FAILED'
            print(f"    pip install {pkg}: {status}")
            if result.returncode != 0:
                print(f"    {result.stderr[-500:]}")
        print("  Re-importing...")
        try:
            from chronos import ChronosPipeline
        except Exception as e2:
            print(f"  ERROR after install: {e2}")
            print("  Manual fix: pip install --upgrade transformers chronos-forecasting")
            return None

    model_name = "amazon/chronos-t5-small"
    print(f"\n  Loading {model_name}...")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch.float32)

    data, _, VAL_END, TEST_END, col_names = load_etth1_normalized()
    n_channels = data.shape[1]

    # Autoregressive chunk size (Chronos sweet spot)
    CHUNK = 64

    all_results = []

    for ctx_len in context_lengths:
        print(f"\n  {'─'*60}")
        print(f"    Chronos │ context={ctx_len}  horizon={pred_len}")
        print(f"  {'─'*60}")

        # How many test windows
        n_possible = TEST_END - VAL_END - pred_len + 1
        stride = max(1, n_possible // max_samples)
        sample_starts = list(range(VAL_END, TEST_END - pred_len + 1, stride))[:max_samples]
        n_samples = len(sample_starts)
        print(f"    Samples: {n_samples} (stride={stride})")

        channel_mses, channel_maes = [], []
        t0 = time.time()

        for ch in range(n_channels):
            series = data[:, ch]
            preds_all, trues_all = [], []

            for si, t in enumerate(sample_starts):
                ctx_start = max(0, t - ctx_len)
                context_np = series[ctx_start:t]
                true_np = series[t:t + pred_len]
                if len(true_np) < pred_len:
                    continue

                # Autoregressive rolling prediction
                pred_pieces = []
                running_ctx = context_np.copy()
                remaining = pred_len
                while remaining > 0:
                    step = min(CHUNK, remaining)
                    ctx_tensor = torch.tensor(running_ctx, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        forecast = pipeline.predict(ctx_tensor, prediction_length=step, num_samples=20)
                    piece = forecast.median(dim=1).values.squeeze().cpu().numpy()
                    if piece.ndim == 0:
                        piece = np.array([piece.item()])
                    pred_pieces.append(piece[:step])
                    running_ctx = np.concatenate([running_ctx, piece[:step]])
                    remaining -= step

                pred_full = np.concatenate(pred_pieces)[:pred_len]
                preds_all.append(pred_full)
                trues_all.append(true_np)

                # Progress
                if (si + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    done_frac = ((ch * n_samples) + si + 1) / (n_channels * n_samples)
                    eta = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
                    print(f"      ch {ch+1}/{n_channels} sample {si+1}/{n_samples} │ "
                          f"elapsed {fmt_time(elapsed)} │ ETA {fmt_time(eta)}")

            if preds_all:
                P = np.stack(preds_all)
                T = np.stack(trues_all)
                channel_mses.append(float(np.mean((P - T) ** 2)))
                channel_maes.append(float(np.mean(np.abs(P - T))))

        elapsed = time.time() - t0
        mse = float(np.mean(channel_mses))
        mae = float(np.mean(channel_maes))

        print(f"\n    RESULT: MSE={mse:.6f}  MAE={mae:.6f}  Time={fmt_time(elapsed)}")

        all_results.append({
            'model': 'Chronos-T5-Small',
            'context_len': ctx_len,
            'pred_len': pred_len,
            'test_mse': mse,
            'test_mae': mae,
            'per_channel_mse': {col_names[i]: channel_mses[i] for i in range(len(channel_mses))},
            'n_samples': n_samples,
            'time_seconds': round(elapsed, 1),
            'method': f'autoregressive rolling (chunk={CHUNK})',
            'status': 'SUCCESS',
        })

    del pipeline
    torch.cuda.empty_cache()
    return all_results


# ════════════════════════════════════════════════════════════════════════════════
# Moirai Evaluation
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_moirai(context_lengths=[336, 720], pred_len=96, max_samples=200):
    """
    Evaluate Moirai zero-shot on ETTh1 test set.

    Uses the correct Moirai inference API: create_predictor() + GluonTS ListDataset.
    Runs channel-independently (target_dim=1) for simplicity and fair comparison
    with channel-independent baselines (PatchTST, Chronos).
    """
    try:
        try:
            import torchvision  # noqa: F401 — prevents circular import
        except Exception:
            pass
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from gluonts.dataset.common import ListDataset
    except ImportError as e:
        print(f"  Import error: {e}")
        print("  Install: pip install uni2ts einops gluonts")
        return None
    except AttributeError as e:
        print(f"  Moirai import failed (torchvision conflict): {e}")
        return None

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    model_name = "Salesforce/moirai-1.1-R-small"
    print(f"\n  Loading {model_name}...")

    # Raw data — Moirai handles its own normalization internally.
    # We normalise predictions+ground truth at the end for fair comparison.
    data, _, VAL_END, TEST_END, col_names = load_etth1_raw()
    n_channels = data.shape[1]

    # Train scaler for fair metric normalisation
    scaler = StandardScaler()
    scaler.fit(data[:8640])

    all_results = []

    for ctx_len in context_lengths:
        print(f"\n  {'─'*60}")
        print(f"    Moirai │ context={ctx_len}  horizon={pred_len}")
        print(f"  {'─'*60}")

        n_possible = TEST_END - VAL_END - pred_len + 1
        stride = max(1, n_possible // max_samples)
        sample_starts = list(range(VAL_END, TEST_END - pred_len + 1, stride))[:max_samples]
        n_samples = len(sample_starts)
        print(f"    Samples: {n_samples}  Channels: {n_channels}  Total series: {n_samples * n_channels}")

        try:
            t0 = time.time()

            # Correct API: MoiraiForecast wraps a MoiraiModule
            # from_pretrained lives on MoiraiModule, not MoiraiForecast
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(model_name),
                prediction_length=pred_len,
                context_length=ctx_len,
                patch_size='auto',
                num_samples=20,
                target_dim=1,              # univariate, channel-independent
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            predictor = model.create_predictor(batch_size=32)

            # Build GluonTS ListDataset.
            # Each entry = one (channel, window) pair.
            # We supply context_length + pred_len timesteps; GluonTS predictor
            # uses the last pred_len as "future" and everything before as context.
            fake_start = pd.Timestamp("2000-01-01")
            entries = []
            true_values = []  # [n_channels * n_samples, pred_len]

            for ch in range(n_channels):
                for t in sample_starts:
                    ctx_start = max(0, t - ctx_len)
                    # Context only — predictor forecasts the next pred_len steps
                    window = data[ctx_start: t, ch].astype(np.float32)
                    entries.append({"target": window, "start": fake_start})
                    true_values.append(data[t: t + pred_len, ch])

            dataset = ListDataset(entries, freq="H")

            print(f"    Running predictor on {len(entries)} series...")
            forecasts = list(predictor.predict(dataset))

            elapsed = time.time() - t0

            # Extract median forecasts
            preds = np.stack([np.median(f.samples, axis=0) for f in forecasts])  # [N, pred_len]
            trues = np.stack(true_values)  # [N, pred_len]

            # Normalise channel-by-channel for fair MSE comparison
            N = n_samples * n_channels
            preds_normed = np.zeros_like(preds)
            trues_normed = np.zeros_like(trues)
            channel_mses, channel_maes = [], []

            for ch in range(n_channels):
                idx = slice(ch * n_samples, (ch + 1) * n_samples)
                p_ch = preds[idx]   # [n_samples, pred_len]
                t_ch = trues[idx]

                # Normalise using train scaler's stats for this channel
                mean_ch = scaler.mean_[ch]
                std_ch = scaler.scale_[ch]
                p_norm = (p_ch - mean_ch) / std_ch
                t_norm = (t_ch - mean_ch) / std_ch

                preds_normed[idx] = p_norm
                trues_normed[idx] = t_norm
                channel_mses.append(float(np.mean((p_norm - t_norm) ** 2)))
                channel_maes.append(float(np.mean(np.abs(p_norm - t_norm))))

            mse = float(np.mean(channel_mses))
            mae = float(np.mean(channel_maes))

            print(f"\n    RESULT: MSE={mse:.6f}  MAE={mae:.6f}  Time={fmt_time(elapsed)}")

            all_results.append({
                'model': 'Moirai-1.1-R-Small',
                'context_len': ctx_len,
                'pred_len': pred_len,
                'test_mse': mse,
                'test_mae': mae,
                'per_channel_mse': {col_names[i]: channel_mses[i] for i in range(len(channel_mses))},
                'n_samples': n_samples,
                'time_seconds': round(elapsed, 1),
                'method': 'univariate via create_predictor() + GluonTS ListDataset',
                'note': 'Metrics on StandardScaler-normalised data for fair comparison',
                'status': 'SUCCESS',
            })

            del model, predictor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n    ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'model': 'Moirai-1.1-R-Small',
                'context_len': ctx_len,
                'pred_len': pred_len,
                'status': f'ERROR: {e}',
            })

    return all_results


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Foundation Model Zero-Shot Evaluation')
    parser.add_argument('--model', type=str, default='all',
                        choices=['chronos', 'moirai', 'all'],
                        help='Which foundation model(s) to evaluate')
    parser.add_argument('--max_samples', type=int, default=200,
                        help='Max test samples per config (default: 200)')
    args = parser.parse_args()

    print('═' * 80)
    print('  W2: Foundation Model Zero-Shot Evaluation on ETTh1')
    print('  Addresses reviewer question: "Have you evaluated on Moirai or Chronos?"')
    print('═' * 80)
    print(f'  Time:    {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Device:  {"CUDA: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'  Models:  {args.model}')
    print(f'  Pred_len: 96 (matching main experiments)')
    print(f'  Context:  [336, 720]')
    print('═' * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    grand_start = time.time()

    # ── Chronos ───────────────────────────────────────────────────────────────
    if args.model in ('chronos', 'all'):
        print('\n' + '─' * 80)
        print('  CHRONOS (Amazon) — T5-based, univariate, channel-independent')
        print('─' * 80)
        chronos_results = evaluate_chronos(
            context_lengths=[336, 720],
            pred_len=96,
            max_samples=args.max_samples,
        )
        if chronos_results:
            all_results.extend(chronos_results)

    # ── Moirai ────────────────────────────────────────────────────────────────
    if args.model in ('moirai', 'all'):
        print('\n' + '─' * 80)
        print('  MOIRAI (Salesforce) — Any-variate, masked encoder')
        print('─' * 80)
        moirai_results = evaluate_moirai(
            context_lengths=[336, 720],
            pred_len=96,
            max_samples=args.max_samples,
        )
        if moirai_results:
            all_results.extend(moirai_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    grand_time = time.time() - grand_start

    output = {
        'purpose': 'W2: Foundation model zero-shot evaluation',
        'dataset': 'ETTh1',
        'pred_len': 96,
        'note': 'Zero-shot (no training). Compared on StandardScaler-normalized data.',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_minutes': round(grand_time / 60, 1),
        'results': all_results,
    }

    out_path = os.path.join(RESULTS_DIR, 'w2_foundation_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '═' * 80)
    print('  FINAL COMPARISON (H=96)')
    print('═' * 80)
    print(f"\n  {'Model':<30s} {'Context':>8s}  {'MSE':>8s}  {'MAE':>8s}")
    print(f"  {'─'*30} {'─'*8}  {'─'*8}  {'─'*8}")

    # Our models (reference)
    print(f"  {'RAFT (ours)':<30s} {'720':>8s}  {'0.3790':>8s}  {'':>8s}")
    print(f"  {'PatchTST-corrected (ours)':<30s} {'720':>8s}  {'0.3949':>8s}  {'':>8s}")
    print(f"  {'PatchTST-corrected (ours)':<30s} {'3000':>8s}  {'0.4259':>8s}  {'':>8s}")

    # Foundation models
    for r in all_results:
        if r.get('status') == 'SUCCESS':
            print(f"  {r['model']:<30s} {r['context_len']:>8d}  {r['test_mse']:>8.4f}  {r['test_mae']:>8.4f}")
        else:
            print(f"  {r['model']:<30s} {r.get('context_len', '?'):>8}  {'ERROR':>8s}  {'':>8s}")

    print(f"\n  Total time: {fmt_time(grand_time)}")
    print(f"  Results: {out_path}")
    print('═' * 80)


if __name__ == '__main__':
    main()
