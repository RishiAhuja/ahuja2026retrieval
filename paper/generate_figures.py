"""Generate compressed figures for the ICLR 2026 TSALM Workshop paper."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

os.makedirs('figures', exist_ok=True)

TARGET_HEIGHT = 1.0   # ← change this ONE value to control both figures


def _save(fig, path, width, height):
    """
    Lay out, then clamp the figure to exactly (width × height) before saving.
    Skipping bbox_inches='tight' is what makes the height stick.
    """
    fig.set_size_inches(width, height)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.22)
    plt.savefig(path, format='pdf', pad_inches=0.02)
    plt.close()
    print(f"Saved {path}")


def plot_iclr_mse_comparison():
    models = [
        'RAFT (720)',
        'PatchTST (720)',
        'Exp7 (720)',
        'Exp3 (720)',
        'Exp2 (1440)',
        'PatchTST (3000)',
        'Exp1 (3000)',
    ]
    mse_values = [0.379, 0.385, 0.416, 0.441, 0.484, 0.647, 1.323]
    colors = [
        '#005A9C',
        '#7f7f7f',
        '#D6B656',
        '#CC79A7',
        '#E69F00',
        '#7f7f7f',
        '#D55E00',
    ]

    fig, ax = plt.subplots(figsize=(10, TARGET_HEIGHT))

    bars = ax.bar(models, mse_values, color=colors, alpha=0.9,
                  edgecolor='black', linewidth=0.6, width=0.65)

    for bar, val in zip(bars, mse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontweight='normal', fontsize=6.5, color='black')

    ax.axhline(y=0.379, color='#005A9C', linestyle='--',
               linewidth=1, alpha=0.6, label='RAFT Baseline')

    ax.set_ylabel('Test MSE', labelpad=4, fontsize=8, weight='bold')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.95, edgecolor='gray')
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.grid(axis='y', linestyle=':', alpha=0.4, color='gray', zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.52)

    _save(fig, 'figures/mse_comparison_iclr.pdf', width=10, height=TARGET_HEIGHT)


def plot_context_vs_mse():
    context_lengths = [720, 1440, 3000]
    timecag_mse = [0.441, 0.484, 1.323]
    patchtst_x = [720, 3000]
    patchtst_y = [0.385, 0.647]
    raft_val = 0.379

    c_raft = '#005A9C'
    c_patch = '#7f7f7f'
    c_vanilla = '#D55E00'

    fig, ax = plt.subplots(figsize=(8, TARGET_HEIGHT))

    ax.plot(context_lengths, timecag_mse, marker='o', linestyle='-',
            color=c_vanilla, linewidth=1.5, markersize=5,
            label='Vanilla Transformer')
    ax.plot(patchtst_x, patchtst_y, marker='D', linestyle='-.',
            color=c_patch, linewidth=1.5, markersize=5,
            label='PatchTST')
    ax.axhline(y=raft_val, color=c_raft, linestyle='--', linewidth=1.5,
               label='RAFT @ 720', zorder=10)
    ax.scatter([720], [raft_val], color=c_raft, marker='*', s=80, zorder=11)

    ax.set_xlabel('Context Window Size (L)', fontsize=8, fontweight='bold')
    ax.set_ylabel('Test MSE', fontsize=8, fontweight='bold')
    ax.set_xticks(context_lengths)
    ax.tick_params(axis='both', labelsize=7)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='white',
              fontsize=7)
    ax.annotate('Error rises with context',
                xy=(1440, 0.484), xytext=(1600, 0.72),
                arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6),
                fontsize=7, style='italic', alpha=0.8)

    _save(fig, 'figures/context_vs_mse_iclr.pdf', width=8, height=TARGET_HEIGHT)


if __name__ == '__main__':
    plot_iclr_mse_comparison()
    plot_context_vs_mse()
    print("Done.")