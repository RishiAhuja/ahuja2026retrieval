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


def plot_iclr_mse_comparison():
    models = [
        'RAFT\n(Ret-720)',
        'PatchTST\n(720)',
        'Exp7\nBest\n(720)',
        'Exp3\nSame\nCtx',
        'Exp2\nMed\n(1440)',
        'PatchTST\n(3000)',
        'Exp1\nLong\n(3000)',
    ]
    mse_values = [0.379, 0.385, 0.416, 0.441, 0.484, 0.647, 1.323]
    colors = [
        '#005A9C',  # RAFT — strong blue
        '#7f7f7f',  # PatchTST 720 — slate grey
        '#D6B656',  # Exp7 — muted gold
        '#CC79A7',  # Exp3 — muted pink
        '#E69F00',  # Exp2 — amber
        '#7f7f7f',  # PatchTST 3000 — slate grey
        '#D55E00',  # Exp1 — vermilion
    ]

    # 50% height reduction: original was (10, 4.5) → now (10, 2.25)
    fig, ax = plt.subplots(figsize=(10, 2.25))

    bars = ax.bar(models, mse_values, color=colors, alpha=0.9,
                  edgecolor='black', linewidth=0.8, width=0.65)

    for bar, val in zip(bars, mse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.015,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontweight='normal', fontsize=8, color='black')

    ax.axhline(y=0.379, color='#005A9C', linestyle='--',
               linewidth=1, alpha=0.6, label='RAFT Baseline')

    ax.set_ylabel('Test MSE (Lower is Better)', labelpad=6, weight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95, edgecolor='gray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.grid(axis='y', linestyle=':', alpha=0.4, color='gray', zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.52)

    plt.tight_layout(pad=0.4)
    plt.savefig('figures/mse_comparison_iclr.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/mse_comparison_iclr.pdf")


def plot_context_vs_mse():
    context_lengths = [720, 1440, 3000]
    timecag_mse = [0.441, 0.484, 1.323]
    patchtst_x = [720, 3000]
    patchtst_y = [0.385, 0.647]
    raft_val = 0.379

    c_raft = '#005A9C'
    c_patch = '#7f7f7f'
    c_vanilla = '#D55E00'

    # 50% height reduction: original was (8, 5) → now (8, 2.5)
    fig, ax = plt.subplots(figsize=(8, 2.5))

    ax.plot(context_lengths, timecag_mse, marker='o', linestyle='-',
            color=c_vanilla, linewidth=2, markersize=7,
            label='Vanilla Transformer (Long Ctx)')
    ax.plot(patchtst_x, patchtst_y, marker='D', linestyle='-.',
            color=c_patch, linewidth=2, markersize=6,
            label='PatchTST (SOTA)')
    ax.axhline(y=raft_val, color=c_raft, linestyle='--', linewidth=2,
               label='RAFT (Retrieval @ 720)', zorder=10)
    ax.scatter([720], [raft_val], color=c_raft, marker='*', s=120, zorder=11)

    ax.set_xlabel('Context Window Size (L)', fontweight='bold')
    ax.set_ylabel('Test MSE (Lower is Better)', fontweight='bold')
    ax.set_xticks(context_lengths)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='white',
              fontsize=8)
    ax.annotate('Error rises with context',
                xy=(1440, 0.484), xytext=(1600, 0.85),
                arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6),
                fontsize=8, style='italic', alpha=0.8)

    plt.tight_layout(pad=0.4)
    plt.savefig('figures/context_vs_mse_iclr.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/context_vs_mse_iclr.pdf")


if __name__ == '__main__':
    plot_iclr_mse_comparison()
    plot_context_vs_mse()
    print("Done.")
