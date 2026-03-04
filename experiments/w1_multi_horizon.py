#!/usr/bin/env python3
"""
CAMERA-READY EXPERIMENT W1: Multi-Horizon Evaluation
=====================================================
Reviewer asked: "Does the inverse scaling law hold for H=336 and H=720?"

Currently ALL experiments use pred_len=96 (predict 96 steps ahead).
This script runs PatchTST, RAFT, and Vanilla Transformer at:
  - H=336 (medium horizon)
  - H=720 (long horizon)
with context lengths 720 and 3000 for each.

USES SAME CONFIG AS CURRENT PAPER (for consistency with existing results).
"""

import os
import sys
import json
import time
import gc
from datetime import datetime

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)

import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_context_forecasting import Exp_LongContext_Forecast


# ============================================================
# CONFIGURATION TEMPLATES
# ============================================================

class BaseArgs:
    """Common settings shared by all experiments"""
    task_name = 'long_term_forecast'
    is_training = 1
    data = 'ETTh1'
    root_path = './data/ETT/'
    data_path = 'ETTh1.csv'
    features = 'M'
    target = 'OT'
    freq = 'h'
    checkpoints = './checkpoints/'
    label_len = 48
    seasonal_patterns = 'Monthly'
    inverse = False
    enc_in = 7
    dec_in = 7
    c_out = 7
    num_workers = 0
    itr = 1
    learning_rate = 0.0001
    loss = 'MSE'
    use_amp = False
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0'
    augmentation_ratio = 0
    p_hidden_dims = [128, 256]
    p_hidden_layers = 2
    moving_avg = 25
    factor = 1
    distil = True
    embed = 'timeF'
    activation = 'gelu'
    output_attention = False
    channel_independence = 1
    decomp_method = 'moving_avg'
    use_norm = 1
    down_sampling_layers = 0
    down_sampling_window = 1
    down_sampling_method = None
    seg_len = 48
    expand = 2
    d_conv = 4
    top_k = 5
    num_kernels = 6


def make_patchtst_args(seq_len, pred_len, batch_size):
    """PatchTST config matching current paper (Table A1)"""
    args = BaseArgs()
    args.model = 'PatchTST'
    args.model_id = f'PatchTST_{seq_len}_H{pred_len}'
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.d_model = 128
    args.n_heads = 8
    args.e_layers = 3
    args.d_layers = 1
    args.d_ff = 256
    args.dropout = 0.1
    args.patch_len = 16
    args.stride = 8
    args.train_epochs = 10
    args.batch_size = batch_size
    args.patience = 3
    args.lradj = 'type1'
    args.des = f'PatchTST_sl{seq_len}_H{pred_len}'
    return args


def make_raft_args(seq_len, pred_len):
    """RAFT config matching current paper"""
    args = BaseArgs()
    args.model = 'RAFT'
    args.model_id = f'RAFT_{seq_len}_H{pred_len}'
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.dropout = 0.1
    args.train_epochs = 10
    args.batch_size = 32
    args.patience = 10
    args.lradj = 'type1'
    args.n_period = 3
    args.topm = 20
    args.des = f'RAFT_sl{seq_len}_H{pred_len}'
    return args


def make_vanilla_args(seq_len, pred_len, batch_size):
    """Vanilla Transformer config matching current paper"""
    args = BaseArgs()
    args.model = 'TransformerLongContext'
    args.model_id = f'VanillaTF_{seq_len}_H{pred_len}'
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.dropout = 0.1
    args.train_epochs = 10
    args.batch_size = batch_size
    args.patience = 3
    args.lradj = 'type1'
    args.des = f'VanillaTF_sl{seq_len}_H{pred_len}'
    return args


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_single_experiment(args, experiment_name):
    """Run a single training + testing experiment"""
    print(f"\n{'='*80}")
    print(f"  {experiment_name}")
    print(f"  Model={args.model}  seq_len={args.seq_len}  pred_len={args.pred_len}")
    print(f"{'='*80}\n")

    # Fix paths to be absolute
    args.root_path = os.path.join(raft_root, 'data/ETT/')
    args.checkpoints = os.path.join(raft_root, 'checkpoints/')

    start_time = time.time()

    try:
        # Choose experiment class
        if args.model == 'RAFT':
            exp = Exp_Long_Term_Forecast(args)
        elif args.model in ['TransformerLongContext', 'TimeCAG']:
            exp = Exp_LongContext_Forecast(args)
        else:
            exp = Exp_Long_Term_Forecast(args)

        setting = f'{args.model_id}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}'

        # Train
        exp.train(setting)
        training_time = time.time() - start_time

        # Test
        mse, mae = exp.test(setting, test=1)
        total_time = time.time() - start_time

        result = {
            'experiment': experiment_name,
            'model': args.model,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'test_mse': float(mse),
            'test_mae': float(mae),
            'training_time_minutes': round(training_time / 60, 2),
            'total_time_minutes': round(total_time / 60, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'SUCCESS'
        }

        print(f"\n  RESULT: MSE={mse:.6f}  MAE={mae:.6f}  Time={training_time/60:.1f}min")
        return result

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'experiment': experiment_name,
            'model': args.model,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'error': str(e),
            'status': 'FAILED'
        }
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def run_all_experiments():
    """Run all multi-horizon experiments"""
    print("\n" + "=" * 80)
    print("  CAMERA-READY W1: Multi-Horizon Evaluation")
    print("  Testing: Does the inverse scaling law hold at H=336 and H=720?")
    print("=" * 80)

    results_dir = os.path.join(raft_root, 'experiments_camera_ready', 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # --------------------------------------------------------
    # EXPERIMENT GROUP 1: PatchTST at H=336
    # --------------------------------------------------------
    experiments = [
        # (name, args_factory)
        ("PatchTST seq=720 H=336",
         lambda: make_patchtst_args(seq_len=720, pred_len=336, batch_size=16)),
        ("PatchTST seq=3000 H=336",
         lambda: make_patchtst_args(seq_len=3000, pred_len=336, batch_size=8)),

        # EXPERIMENT GROUP 2: PatchTST at H=720
        ("PatchTST seq=720 H=720",
         lambda: make_patchtst_args(seq_len=720, pred_len=720, batch_size=16)),
        ("PatchTST seq=3000 H=720",
         lambda: make_patchtst_args(seq_len=3000, pred_len=720, batch_size=8)),

        # EXPERIMENT GROUP 3: RAFT at H=336 and H=720
        ("RAFT seq=720 H=336",
         lambda: make_raft_args(seq_len=720, pred_len=336)),
        ("RAFT seq=720 H=720",
         lambda: make_raft_args(seq_len=720, pred_len=720)),

        # EXPERIMENT GROUP 4: Vanilla Transformer (optional - for completeness)
        ("Vanilla TF seq=720 H=336",
         lambda: make_vanilla_args(seq_len=720, pred_len=336, batch_size=32)),
        ("Vanilla TF seq=3000 H=336",
         lambda: make_vanilla_args(seq_len=3000, pred_len=336, batch_size=8)),
        ("Vanilla TF seq=720 H=720",
         lambda: make_vanilla_args(seq_len=720, pred_len=720, batch_size=32)),
        ("Vanilla TF seq=3000 H=720",
         lambda: make_vanilla_args(seq_len=3000, pred_len=720, batch_size=8)),
    ]

    for i, (name, args_fn) in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"  Experiment {i}/{len(experiments)}: {name}")
        print(f"{'#'*80}")

        args = args_fn()
        result = run_single_experiment(args, name)
        all_results.append(result)

        # Save progress after each experiment
        with open(os.path.join(results_dir, 'w1_multi_horizon_results.json'), 'w') as f:
            json.dump({
                'purpose': 'Camera-ready W1: Multi-horizon evaluation',
                'question': 'Does inverse scaling law hold at H=336 and H=720?',
                'completed': i,
                'total': len(experiments),
                'results': all_results
            }, f, indent=2)

        print(f"\n  Progress saved ({i}/{len(experiments)} complete)")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("  MULTI-HORIZON RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Context':<10} {'Horizon':<10} {'MSE':<12} {'Status'}")
    print("-" * 70)

    for r in all_results:
        if r['status'] == 'SUCCESS':
            print(f"{r['model']:<25} {r['seq_len']:<10} {r['pred_len']:<10} {r['test_mse']:<12.6f} OK")
        else:
            print(f"{r['model']:<25} {r['seq_len']:<10} {r['pred_len']:<10} {'FAILED':<12} {r.get('error','')[:30]}")

    # Compute degradation for each horizon
    print("\n\nDEGRADATION ANALYSIS:")
    print("-" * 50)
    for horizon in [336, 720]:
        p720 = next((r for r in all_results if r['model'] == 'PatchTST'
                      and r['seq_len'] == 720 and r['pred_len'] == horizon
                      and r['status'] == 'SUCCESS'), None)
        p3000 = next((r for r in all_results if r['model'] == 'PatchTST'
                       and r['seq_len'] == 3000 and r['pred_len'] == horizon
                       and r['status'] == 'SUCCESS'), None)

        if p720 and p3000:
            degradation = (p3000['test_mse'] - p720['test_mse']) / p720['test_mse'] * 100
            print(f"  H={horizon}: PatchTST 720→3000 = {p720['test_mse']:.4f} → {p3000['test_mse']:.4f} ({degradation:+.1f}%)")

    print(f"\n  Full results saved to: {results_dir}/w1_multi_horizon_results.json")
    print("=" * 80)


# ============================================================
# SELECTIVE RUNNER (for running individual experiments)
# ============================================================

def run_patchtst_only():
    """Run only PatchTST experiments (4 configs)"""
    results_dir = os.path.join(raft_root, 'experiments_camera_ready', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results = []

    for seq_len, batch_size in [(720, 16), (3000, 8)]:
        for pred_len in [336, 720]:
            name = f"PatchTST seq={seq_len} H={pred_len}"
            args = make_patchtst_args(seq_len, pred_len, batch_size)
            result = run_single_experiment(args, name)
            results.append(result)

    with open(os.path.join(results_dir, 'w1_patchtst_multi_horizon.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_raft_only():
    """Run only RAFT experiments (2 configs)"""
    results_dir = os.path.join(raft_root, 'experiments_camera_ready', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results = []

    for pred_len in [336, 720]:
        name = f"RAFT seq=720 H={pred_len}"
        args = make_raft_args(720, pred_len)
        result = run_single_experiment(args, name)
        results.append(result)

    with open(os.path.join(results_dir, 'w1_raft_multi_horizon.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'patchtst', 'raft'],
                        help='Which experiments to run')
    cli_args = parser.parse_args()

    if cli_args.mode == 'all':
        run_all_experiments()
    elif cli_args.mode == 'patchtst':
        run_patchtst_only()
    elif cli_args.mode == 'raft':
        run_raft_only()
