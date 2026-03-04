# RAFT: Retrieval-Augmented Forecasting for Time Series

**Retrieval Mechanisms Surpass Long-Context Scaling in Time Series Forecasting**

*Rishi Ahuja, Kumar Prateek, Simranjit Singh, Vijay Kumar*
Department of Information Technology, NIT Jalandhar

**Accepted as Poster at ICLR 2026 TSALM Workshop**

---

## Overview

RAFT introduces retrieval-augmented generation (RAG) to time series forecasting. Instead of scaling the context window (which causes attention entropy dilution), RAFT retrieves the most relevant historical subsequences using periodicity-aware DTW matching and augments the input at inference time.

**Key Results (ETTh1, pred_len=96):**

| Model | Context | MSE | MAE |
|-------|---------|-----|-----|
| **RAFT (ours)** | 720 | **0.379** | **0.399** |
| PatchTST | 720 | 0.385 | 0.405 |
| PatchTST | 3000 | 0.426 | 0.434 |
| TransformerLongContext | 3000 | 0.647 | 0.556 |
| Chronos (zero-shot) | 720 | 0.483 | 0.471 |
| Moirai (zero-shot) | 720 | 0.553 | 0.516 |

## Repository Structure

```
RAFT/
├── models/                     # Model implementations
│   ├── RAFT.py                 # RAFT: Retrieval-Augmented Forecasting Transformer
│   ├── PatchTST.py             # PatchTST baseline (Nie et al., 2023)
│   └── TransformerLongContext.py # Vanilla Transformer with long context
├── layers/                     # Shared layers
│   ├── Retrieval.py            # Periodicity-aware DTW retrieval module
│   ├── SelfAttention_Family.py # Full attention implementation
│   ├── Embed.py                # Positional and temporal embeddings
│   └── Transformer_EncDec.py   # Encoder/decoder architecture
├── data_provider/              # Data loading and processing
├── exp/                        # Experiment runners
│   ├── exp_long_term_forecasting.py   # RAFT & PatchTST experiments
│   └── exp_long_context_forecasting.py # Long-context baseline
├── utils/                      # Utilities (metrics, DTW, augmentation, etc.)
├── experiments/                # Camera-ready experiment scripts
│   ├── w1_multi_horizon.py     # Multi-horizon evaluation (H=96,336,720)
│   ├── w2_foundation_eval.py   # Chronos & Moirai zero-shot evaluation
│   ├── w3_attention_entropy.py # Attention entropy analysis
│   └── run_corrected_patchtst.py # Corrected PatchTST hyperparameters
├── scripts/                    # Shell scripts for reproduction
├── data/ETT/                   # ETT datasets
├── paper/                      # Paper source (LaTeX)
├── run.py                      # Entry point: RAFT & PatchTST
└── run_long_context.py         # Entry point: TransformerLongContext
```

## Setup

```bash
# Clone the repository
git clone https://github.com/RishiAhuja/caft.git
cd caft

# Install dependencies
pip install -r requirements.txt

# Download datasets (if not included)
bash scripts/download_data.sh
```

## Reproducing Main Results

### Quick Start — All Main Experiments

```bash
bash scripts/run_main_experiments.sh
```

### Individual Experiments

**RAFT (our method):**
```bash
python run.py \
    --model RAFT --data ETTh1 --seq_len 720 --pred_len 96 \
    --d_model 512 --n_heads 8 --e_layers 2 --d_ff 2048 \
    --train_epochs 10 --batch_size 32 --learning_rate 0.0001
```

**PatchTST (corrected hyperparameters):**
```bash
python run.py \
    --model PatchTST --data ETTh1 --seq_len 720 --pred_len 96 \
    --d_model 16 --n_heads 4 --e_layers 3 --d_ff 128 \
    --dropout 0.3 --train_epochs 100 --batch_size 128 --lradj TST
```

**Vanilla Transformer (long context baseline):**
```bash
python run_long_context.py \
    --model TransformerLongContext --data ETTh1 --seq_len 3000 --pred_len 96 \
    --d_model 128 --n_heads 8 --e_layers 2 --d_ff 512 \
    --dropout 0.15 --train_epochs 30 --batch_size 24
```

## Camera-Ready Experiments

These scripts reproduce the additional analyses added during the camera-ready revision:

```bash
# Multi-horizon evaluation (Table A2)
python experiments/w1_multi_horizon.py

# Foundation model comparison (Table A3) — requires chronos-forecasting, uni2ts
python experiments/w2_foundation_eval.py --model all

# Attention entropy analysis (Section 3.3)
python experiments/w3_attention_entropy.py

# Corrected PatchTST with proper hyperparameters (Table A4)
python experiments/run_corrected_patchtst.py --phase all
```

## Citation

```bibtex
@inproceedings{
  ahuja2026retrieval,
  title={Retrieval Mechanisms Surpass Long-Context Scaling in Time Series Forecasting},
  author={Rishi Ahuja and Kumar Prateek and Simranjit Singh and Dr Vijay Kumar},
  booktitle={1st ICLR Workshop on Time Series in the Age of Large Models},
  year={2026},
  url={https://openreview.net/forum?id=Qj96MlCmZw}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
