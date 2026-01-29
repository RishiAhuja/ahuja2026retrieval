#!/usr/bin/env python3
"""
Updated Graph Generation Script
Uses REAL training logs and predictions instead of synthetic data
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11


def load_training_logs():
    """Load all training logs"""
    log_dir = './experiments/results/training_logs'
    
    logs = {}
    log_files = {
        'RAFT': 'raft_losses.json',
        'Time-CAG v1 (3000)': 'time_cag_v1_3000.json',
        'Time-CAG v2 (1440)': 'time_cag_v2_1440.json',
        'Time-CAG v3 (720)': 'time_cag_v3_720.json',
    }
    
    for name, filename in log_files.items():
        filepath = os.path.join(log_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                logs[name] = json.load(f)
            print(f"✅ Loaded: {name}")
        else:
            # Try alternate naming
            alt_filename = filename.replace('_', '-').replace('.json', '_losses.json')
            alt_filepath = os.path.join(log_dir, alt_filename)
            if os.path.exists(alt_filepath):
                with open(alt_filepath, 'r') as f:
                    logs[name] = json.load(f)
                print(f"✅ Loaded: {name} (from {alt_filename})")
            else:
                print(f"⚠️  Missing: {filepath}")
    
    return logs


def load_predictions():
    """Load prediction data"""
    pred_file = './experiments/results/predictions/all_predictions.json'
    
    if os.path.exists(pred_file):
        with open(pred_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded predictions from: {pred_file}")
        return data['predictions']
    else:
        print(f"⚠️  Predictions file not found: {pred_file}")
        return None


def plot_loss_curves(logs, save_path='./experiments/results/graphs'):
    """
    Graph 2: Training/Validation Loss Curves (REAL DATA)
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'RAFT': '#2ecc71',
        'Time-CAG v1 (3000)': '#e74c3c',
        'Time-CAG v2 (1440)': '#f39c12',
        'Time-CAG v3 (720)': '#3498db',
    }
    
    # Left: Training Loss
    for name, data in logs.items():
        ax1.plot(data['epochs'], data['train_losses'], 
                marker='o', label=name, color=colors.get(name, 'gray'),
                linewidth=2, markersize=6, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Training Loss (MSE)', fontsize=13, fontweight='bold')
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Validation Loss
    for name, data in logs.items():
        ax2.plot(data['epochs'], data['val_losses'],
                marker='s', label=name, color=colors.get(name, 'gray'),
                linewidth=2, markersize=6, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Validation Loss (MSE)', fontsize=13, fontweight='bold')
    ax2.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(save_path, 'graph2_loss_curves_REAL.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graph 2 saved: {output_file}")


def plot_predictions(predictions, save_path='./experiments/results/graphs'):
    """
    Graph 4: Prediction Visualization (REAL DATA)
    """
    os.makedirs(save_path, exist_ok=True)
    
    if not predictions:
        print("⚠️  No predictions available, skipping Graph 4")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    models_to_plot = [
        ('RAFT_720', 'RAFT (seq_len=720)', '#2ecc71'),
        ('TimeCAG_3000', 'Time-CAG v1 (seq_len=3000)', '#e74c3c'),
        ('TimeCAG_720', 'Time-CAG v3 (seq_len=720)', '#3498db'),
    ]
    
    for idx, (key, title, color) in enumerate(models_to_plot):
        if key not in predictions:
            print(f"⚠️  Missing predictions for {key}")
            continue
        
        data = predictions[key]
        ground_truth = data['ground_truth']
        preds = data['predictions']
        
        time_steps = list(range(len(ground_truth)))
        
        ax = axes[idx]
        ax.plot(time_steps, ground_truth, label='Ground Truth', 
               color='black', linewidth=2.5, alpha=0.7, linestyle='-')
        ax.plot(time_steps, preds, label='Predictions',
               color=color, linewidth=2, alpha=0.8, linestyle='--')
        
        # Calculate MSE
        mse = np.mean((np.array(ground_truth) - np.array(preds))**2)
        
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\nMSE: {mse:.4f}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    # Remove unused subplot
    if len(models_to_plot) < 4:
        fig.delaxes(axes[3])
    
    plt.tight_layout()
    output_file = os.path.join(save_path, 'graph4_predictions_REAL.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graph 4 saved: {output_file}")


def main():
    """Generate all graphs with REAL data"""
    print("=" * 80)
    print("GENERATING GRAPHS WITH REAL DATA")
    print("=" * 80)
    print()
    
    # Load data
    print("📊 Loading training logs...")
    logs = load_training_logs()
    
    print("\n📊 Loading predictions...")
    predictions = load_predictions()
    
    print("\n🎨 Generating graphs...")
    
    # Generate updated graphs
    if logs:
        plot_loss_curves(logs)
    else:
        print("⚠️  No training logs found, skipping Graph 2")
    
    if predictions:
        plot_predictions(predictions)
    else:
        print("⚠️  No predictions found, skipping Graph 4")
    
    print("\n" + "=" * 80)
    print("✅ DONE! Check ./experiments/results/graphs/")
    print("=" * 80)


if __name__ == '__main__':
    main()
