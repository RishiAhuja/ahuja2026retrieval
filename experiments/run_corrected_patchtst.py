#!/usr/bin/env python3
"""
Corrected PatchTST Experiments — T4 Optimized
==============================================

Tests PatchTST with CORRECT hyperparameters from the original ICLR 2023 paper.
Uses proper OneCycleLR per-step scheduler (matching the original PatchTST repo exactly).

PURPOSE:
  Determine if PatchTST truly degrades with longer context,
  or if the degradation was caused by wrong hyperparameters.

PHASES:
  Phase 1  (sl=336):  Validation — reproduce original paper results     (~2-3 hrs on T4)
  Phase 2A (sl=720):  Fair comparison with RAFT-720 (MSE=0.379)         (~4-6 hrs on T4)
  Phase 2B (sl=3000): Critical degradation test                          (~8-12 hrs on T4)

USAGE:
  python run_corrected_patchtst.py --phase all      # Run all 3 phases sequentially
  python run_corrected_patchtst.py --phase 1        # Phase 1 only
  python run_corrected_patchtst.py --phase 2a       # Phase 2A only
  python run_corrected_patchtst.py --phase 2b       # Phase 2B only

HYPERPARAMETER CORRECTIONS (vs our original wrong config):
  Parameter      WRONG (ours)     CORRECT (original paper)
  ──────────     ────────────     ────────────────────────
  d_model        128              16
  n_heads        8                4
  d_ff           256              128
  train_epochs   10               100
  batch_size     16 (or 8)        128 (32 for sl=3000 on T4)
  lradj          'type1'          OneCycleLR per-step
  dropout        0.1              0.3
  patience       3                20 (unused, no early stopping)
"""

import os
import sys
import json
import time
import math
import copy
import argparse
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch import optim

# ════════════════════════════════════════════════════════════════════════════════
# Path Setup
# ════════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAFT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

sys.path.insert(0, RAFT_ROOT)
os.chdir(RAFT_ROOT)

from data_provider.data_factory import data_provider
from models.PatchTST import Model as PatchTSTModel
from utils.metrics import metric


# ════════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ════════════════════════════════════════════════════════════════════════════════

def get_gpu_info():
    """Get GPU memory stats. Returns (allocated_gb, total_gb) or (0, 0) for CPU."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return round(alloc, 2), round(total, 2)
    return 0, 0


def fmt_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 0:
        return "???"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def print_box(text, width=90, char='═'):
    """Print text in a box."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_separator(width=90):
    print(f"{'─' * width}")


# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

class BaseArgs:
    """
    CORRECTED PatchTST configuration matching the original ICLR 2023 paper.
    Reference: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/etth1.sh
    """
    # Task
    task_name = 'long_term_forecast'
    is_training = 1
    model = 'PatchTST'

    # Data
    data = 'ETTh1'
    root_path = os.path.join(RAFT_ROOT, 'data', 'ETT')
    data_path = 'ETTh1.csv'
    features = 'M'
    target = 'OT'
    freq = 'h'
    checkpoints = os.path.join(RAFT_ROOT, 'checkpoints')

    # Forecasting
    label_len = 48
    pred_len = 96
    seasonal_patterns = 'Monthly'
    inverse = False

    # Architecture ── CORRECTED
    enc_in = 7
    dec_in = 7
    c_out = 7
    d_model = 16       # was 128
    n_heads = 4        # was 8
    e_layers = 3
    d_layers = 1
    d_ff = 128         # was 256
    patch_len = 16
    stride = 8

    # Training ── CORRECTED
    num_workers = 0
    itr = 1
    train_epochs = 100  # was 10
    patience = 20       # was 3 (unused—no early stopping in training loop)
    learning_rate = 0.0001
    loss = 'MSE'
    lradj = 'TST'
    pct_start = 0.3
    use_amp = False

    # Regularization ── CORRECTED
    dropout = 0.3       # was 0.1
    fc_dropout = 0.3
    head_dropout = 0.0

    # Other
    moving_avg = 25
    factor = 1
    distil = True
    embed = 'timeF'
    activation = 'gelu'
    output_attention = False

    # GPU
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0'

    # Required by data_loader but unused by PatchTST
    augmentation_ratio = 0
    p_hidden_dims = [128, 256]
    p_hidden_layers = 2


def make_args(seq_len, batch_size, use_amp=False, model_id='PatchTST', des='exp'):
    """Create an Args object for a given seq_len / batch_size."""
    args = BaseArgs()
    args.seq_len = seq_len
    args.batch_size = batch_size
    args.use_amp = use_amp
    args.model_id = model_id
    args.des = des
    return args


PHASE_CONFIGS = {
    '1': {
        'name': 'Phase 1 — Validation (sl=336)',
        'goal': 'Reproduce original PatchTST paper results (expected MSE ≈ 0.370–0.385)',
        'seq_len': 336,
        'batch_size': 128,
        'use_amp': False,
        'model_id': 'PatchTST_336_CORRECTED',
        'result_key': 'phase1_sl336',
    },
    '2a': {
        'name': 'Phase 2A — Fair comparison (sl=720)',
        'goal': 'Compare corrected PatchTST-720 against RAFT-720 (MSE=0.379)',
        'seq_len': 720,
        'batch_size': 128,
        'use_amp': False,
        'model_id': 'PatchTST_720_CORRECTED',
        'result_key': 'phase2a_sl720',
    },
    '2b': {
        'name': 'Phase 2B — Critical degradation test (sl=3000)',
        'goal': 'Test whether PatchTST still degrades at extreme context length',
        'seq_len': 3000,
        'batch_size': 32,   # Reduced for T4 (15 GB)
        'use_amp': True,     # FP16 for additional memory savings
        'model_id': 'PatchTST_3000_CORRECTED',
        'result_key': 'phase2b_sl3000',
    },
}


# ════════════════════════════════════════════════════════════════════════════════
# Validation Loop
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, data_loader, criterion, args, device):
    """Run evaluation on a data loader. Returns mean loss."""
    model.eval()
    losses = []
    for _, batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        if args.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
        losses.append(criterion(outputs.detach().cpu(), batch_y.detach().cpu()).item())

    model.train()
    return float(np.mean(losses))


# ════════════════════════════════════════════════════════════════════════════════
# Testing (final metrics)
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_model(model, data_loader, args, device):
    """Run test and compute MSE, MAE using the project's metric function."""
    model.eval()
    preds, trues = [], []

    for _, batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        if args.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if args.features == 'MS' else 0
        out_np = outputs[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
        true_np = batch_y[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
        preds.append(out_np)
        trues.append(true_np)

    preds = np.concatenate(preds, axis=0).reshape(-1, args.pred_len, args.enc_in if args.features == 'M' else 1)
    trues = np.concatenate(trues, axis=0).reshape(-1, args.pred_len, args.enc_in if args.features == 'M' else 1)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    return float(mse), float(mae), float(rmse)


# ════════════════════════════════════════════════════════════════════════════════
# Main Training + Evaluation Pipeline
# ════════════════════════════════════════════════════════════════════════════════

def run_phase(phase_key):
    """Run a single experiment phase with rich monitoring."""

    cfg = PHASE_CONFIGS[phase_key]
    args = make_args(
        seq_len=cfg['seq_len'],
        batch_size=cfg['batch_size'],
        use_amp=cfg['use_amp'],
        model_id=cfg['model_id'],
        des=cfg['result_key'],
    )
    result_key = cfg['result_key']

    # ── Banner ────────────────────────────────────────────────────────────────
    print_box(cfg['name'])
    print(f"  Goal:      {cfg['goal']}")
    print(f"  Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    num_patches = int((args.seq_len - args.patch_len) / args.stride + 2)
    print(f"\n  CONFIG")
    print(f"  ├─ seq_len:     {args.seq_len}  ({num_patches} patches)")
    print(f"  ├─ pred_len:    {args.pred_len}")
    print(f"  ├─ d_model:     {args.d_model}  (original paper: 16)")
    print(f"  ├─ n_heads:     {args.n_heads}")
    print(f"  ├─ e_layers:    {args.e_layers}")
    print(f"  ├─ d_ff:        {args.d_ff}")
    print(f"  ├─ dropout:     {args.dropout}")
    print(f"  ├─ batch_size:  {args.batch_size}")
    print(f"  ├─ epochs:      {args.train_epochs}")
    print(f"  ├─ lr:          {args.learning_rate}  (OneCycleLR per step)")
    print(f"  └─ use_amp:     {args.use_amp}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n  Loading data...")
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    train_steps = len(train_loader)
    print(f"  ├─ Train: {len(train_data)} samples → {train_steps} batches/epoch")
    print(f"  ├─ Val:   {len(val_data)} samples")
    print(f"  └─ Test:  {len(test_data)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n  Building model...")
    model = PatchTSTModel(args).float().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    total_steps = train_steps * args.train_epochs

    # Proper OneCycleLR per-step (matches original PatchTST repo exactly)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=train_steps,
        epochs=args.train_epochs,
        pct_start=args.pct_start,
    )
    criterion = nn.MSELoss()

    if args.use_amp:
        scaler = torch.amp.GradScaler('cuda')

    # ── Checkpoint path ───────────────────────────────────────────────────────
    setting = f"{args.model_id}_{args.data}_sl{args.seq_len}_pl{args.pred_len}"
    ckpt_dir = os.path.join(args.checkpoints, setting)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val = float('inf')
    best_state = None
    best_epoch = -1
    log = []
    total_start = time.time()

    alloc, total_mem = get_gpu_info()
    print(f"\n  GPU memory: {alloc:.1f} / {total_mem:.1f} GB")

    print_box(f"TRAINING — {cfg['name']}", char='─')
    print(f"  {args.train_epochs} epochs × {train_steps} steps = {total_steps} total steps")
    print(f"  OneCycleLR: warmup {int(args.pct_start*100)}% → cosine decay")
    print()

    for epoch in range(args.train_epochs):
        ep = epoch + 1
        epoch_start = time.time()
        model.train()
        epoch_losses = []

        for step, (idx, bx, by, bxm, bym) in enumerate(train_loader):
            optimizer.zero_grad()
            bx = bx.float().to(device)
            by = by.float().to(device)
            bxm = bxm.float().to(device)
            bym = bym.float().to(device)

            dec_inp = torch.zeros_like(by[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([by[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            if args.use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(bx, bxm, dec_inp, bym)
                    f_dim = -1 if args.features == 'MS' else 0
                    loss = criterion(out[:, -args.pred_len:, f_dim:],
                                     by[:, -args.pred_len:, f_dim:])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(bx, bxm, dec_inp, bym)
                f_dim = -1 if args.features == 'MS' else 0
                loss = criterion(out[:, -args.pred_len:, f_dim:],
                                 by[:, -args.pred_len:, f_dim:])
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_losses.append(loss.item())

            # Intra-epoch progress (5 updates per epoch)
            report_interval = max(train_steps // 5, 1)
            if (step + 1) % report_interval == 0 or (step + 1) == train_steps:
                cur_lr = optimizer.param_groups[0]['lr']
                recent = np.mean(epoch_losses[-report_interval:])
                alloc, _ = get_gpu_info()
                global_step = epoch * train_steps + step + 1
                pct = global_step / total_steps * 100
                print(f"    step {step+1:4d}/{train_steps} │ loss {recent:.4f} │ "
                      f"lr {cur_lr:.2e} │ gpu {alloc:.1f}GB │ "
                      f"global {global_step}/{total_steps} ({pct:.0f}%)")

        # ── Epoch summary ─────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        train_loss = float(np.mean(epoch_losses))
        val_loss = evaluate(model, val_loader, criterion, args, device)
        cur_lr = optimizer.param_groups[0]['lr']

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            improved = "  ★ BEST"

        elapsed = time.time() - total_start
        eta = (elapsed / ep) * (args.train_epochs - ep)
        alloc, total_mem = get_gpu_info()

        print(f"\n  Epoch {ep:3d}/{args.train_epochs} │ "
              f"train {train_loss:.4f} │ val {val_loss:.4f} │ "
              f"lr {cur_lr:.2e} │ {epoch_time:.0f}s{improved}")
        print(f"  {'':10s}       │ "
              f"best  {best_val:.4f} (ep {best_epoch:3d}) │ "
              f"gpu {alloc:.1f}/{total_mem:.1f}GB │ "
              f"ETA {fmt_time(eta)} │ elapsed {fmt_time(elapsed)}")
        print()

        log.append({
            'epoch': ep,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': cur_lr,
            'epoch_time_s': round(epoch_time, 1),
            'best_val': best_val,
            'best_epoch': best_epoch,
            'gpu_alloc_gb': alloc,
        })

        # Progressive save every 10 epochs
        if ep % 10 == 0 or ep == args.train_epochs:
            _save_json(os.path.join(RESULTS_DIR, f'{result_key}_training_log.json'), log)

    training_time = time.time() - total_start

    # ── Save best checkpoint ──────────────────────────────────────────────────
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    torch.save(best_state, ckpt_path)
    print_box(f"TRAINING COMPLETE — {cfg['name']}", char='─')
    print(f"  Best val loss: {best_val:.6f} (epoch {best_epoch})")
    print(f"  Training time: {fmt_time(training_time)}")
    print(f"  Checkpoint:    {ckpt_path}")

    # ── Test with best model ──────────────────────────────────────────────────
    model.load_state_dict(best_state)
    print(f"\n  Running test evaluation...")
    mse, mae, rmse = test_model(model, test_loader, args, device)

    print_box(f"TEST RESULTS — {cfg['name']}", char='─')
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    # Phase-specific analysis
    if phase_key == '1':
        if 0.350 <= mse <= 0.405:
            print(f"\n  ✓ VALIDATION PASSED — MSE {mse:.4f} in expected range [0.350, 0.405]")
        else:
            print(f"\n  ✗ MSE {mse:.4f} outside expected range — investigate further")
    elif phase_key == '2a':
        diff_pct = (mse - 0.379) / 0.379 * 100
        print(f"\n  vs RAFT-720 (0.379): {'better' if mse < 0.379 else 'worse'} by {abs(diff_pct):.1f}%")
    elif phase_key == '2b':
        # Try loading Phase 2A result
        p2a_file = os.path.join(RESULTS_DIR, 'phase2a_sl720_results.json')
        if os.path.exists(p2a_file):
            p2a_mse = json.load(open(p2a_file))['test_mse']
            degradation = (mse - p2a_mse) / p2a_mse * 100
            print(f"\n  DEGRADATION ANALYSIS (720 → 3000):")
            print(f"    PatchTST-720  (corrected): {p2a_mse:.4f}")
            print(f"    PatchTST-3000 (corrected): {mse:.4f}")
            print(f"    Change: {degradation:+.1f}%")
            if degradation > 10:
                print(f"    → Degradation CONFIRMED even with correct config")
            elif degradation < -5:
                print(f"    → Longer context HELPS — original finding was wrong")
            else:
                print(f"    → Roughly comparable — ambiguous result")
        print(f"\n  vs RAFT-720 (0.379): {(mse - 0.379)/0.379*100:+.1f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'phase': cfg['name'],
        'result_key': result_key,
        'model': 'PatchTST',
        'config': 'CORRECTED (ICLR 2023 original hyperparameters)',
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'test_mse': mse,
        'test_mae': mae,
        'test_rmse': rmse,
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'training_time_minutes': round(training_time / 60, 1),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'configuration': {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'e_layers': args.e_layers,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'train_epochs': args.train_epochs,
            'learning_rate': args.learning_rate,
            'scheduler': 'OneCycleLR (per-step)',
            'pct_start': args.pct_start,
            'use_amp': args.use_amp,
        },
    }
    out_file = os.path.join(RESULTS_DIR, f'{result_key}_results.json')
    _save_json(out_file, results)
    print(f"\n  Results saved → {out_file}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()

    return mse, mae


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Corrected PatchTST Experiments (T4 optimized)')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', '1', '2a', '2b'],
                        help='Which phase(s) to run')
    cli = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Startup banner ────────────────────────────────────────────────────────
    print('═' * 90)
    print('  CORRECTED PATCHTST EXPERIMENTS')
    print('  Using original ICLR 2023 paper hyperparameters + OneCycleLR')
    print('  Optimised for Tesla T4 (15 GB)')
    print('═' * 90)
    print(f'  Time:     {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if torch.cuda.is_available():
        print(f'  GPU:      {torch.cuda.get_device_name(0)}')
        _, total = get_gpu_info()
        print(f'  VRAM:     {total:.1f} GB')
    else:
        print('  GPU:      None (CPU mode — will be slow)')
    print(f'  Phase:    {cli.phase}')
    print(f'  Results:  {RESULTS_DIR}/')
    print('═' * 90)

    phases_to_run = ['1', '2a', '2b'] if cli.phase == 'all' else [cli.phase]
    all_results = {}
    grand_start = time.time()

    for p in phases_to_run:
        try:
            mse, mae = run_phase(p)
            all_results[p] = {'mse': mse, 'mae': mae, 'status': 'SUCCESS'}
        except torch.cuda.OutOfMemoryError:
            print(f"\n  ✗ CUDA OOM in phase {p}!")
            print(f"    Try reducing batch_size in PHASE_CONFIGS['{p}']")
            all_results[p] = {'status': 'OOM'}
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n  ✗ ERROR in phase {p}: {e}")
            import traceback
            traceback.print_exc()
            all_results[p] = {'status': f'ERROR: {e}'}

    # ── Final summary ─────────────────────────────────────────────────────────
    grand_time = time.time() - grand_start
    print_box('FINAL SUMMARY')
    print(f"  Total wall time: {fmt_time(grand_time)}")
    print()
    print(f"  {'Phase':<35s} {'MSE':>8s}  {'MAE':>8s}  {'Status'}")
    print(f"  {'─'*35} {'─'*8}  {'─'*8}  {'─'*12}")

    for p in phases_to_run:
        r = all_results[p]
        name = PHASE_CONFIGS[p]['name']
        if r['status'] == 'SUCCESS':
            print(f"  {name:<35s} {r['mse']:8.4f}  {r['mae']:8.4f}  ✓")
        else:
            print(f"  {name:<35s} {'—':>8s}  {'—':>8s}  {r['status']}")

    print(f"\n  {'RAFT-720 (baseline)':<35s} {'0.3790':>8s}  {'':>8s}")
    print(f"\n  Results directory: {RESULTS_DIR}/")
    print()


if __name__ == '__main__':
    main()
