"""Training Logger -- captures training/validation losses for visualization."""

import os
import json
from datetime import datetime


class TrainingLogger:
    """Logs training and validation losses during model training."""

    def __init__(self, log_dir='./experiments/results/training_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.experiment_name = None

    def start_experiment(self, experiment_name):
        """Initialize logging for a new experiment."""
        self.experiment_name = experiment_name
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        print(f"Training Logger: Tracking {experiment_name}")

    def log_epoch(self, epoch, train_loss, val_loss):
        """Log losses for a single epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))

    def save(self):
        """Save logged data to JSON file."""
        if not self.experiment_name:
            print("Warning: No experiment name set, skipping save")
            return

        data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'min_val_loss': min(self.val_losses) if self.val_losses else None,
        }

        filename = f"{self.experiment_name.replace(' ', '_').lower()}_losses.json"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Training logs saved to: {filepath}")
        return filepath


def load_training_logs(experiment_name, log_dir='./experiments/results/training_logs'):
    """Load training logs for a given experiment."""
    filename = f"{experiment_name.replace(' ', '_').lower()}_losses.json"
    filepath = os.path.join(log_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training log not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data
