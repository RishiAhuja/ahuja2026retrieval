## ETHH2

RAFT baseline 720
{
  "model": "RAFT",
  "seq_len": 720,
  "pred_len": 96,
  "label_len": 48,
  
  "architecture": {
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "dropout": 0.1,
    "activation": "gelu"
  },
  
  "raft_specific": {
    "n_period": 3,      # Retrieval periods
    "topm": 20          # Top-M retrieved windows
  },
  
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "optimizer": "Adam",
    "loss": "MSE"
  },
  
  "results": {
    "test_mse": 0.2817,  # 🥇 BEST on ETTh2
    "test_mae": 0.3483,
    "training_time_min": 0.47
  }
}


PatchTST (720) - Short Context

{
  "model": "PatchTST",
  "seq_len": 720,
  "pred_len": 96,
  "label_len": 48,
  
  "architecture": {
    "d_model": 128,
    "n_heads": 16,
    "e_layers": 3,
    "d_layers": 1,
    "d_ff": 256,
    "dropout": 0.1,
    "activation": "gelu"
  },
  
  "patchtst_specific": {
    "patch_len": 16,
    "stride": 8
  },
  
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "optimizer": "Adam",
    "loss": "MSE"
  },
  
  "results": {
    "test_mse": 0.3073,
    "test_mae": 0.3667,
    "training_time_min": 3.84
  }
}

Exp 3: PatchTST (3000) - Long Context ⚠️ DEGRADATION

{
  "model": "PatchTST",
  "seq_len": 3000,      # 4× longer context
  "pred_len": 96,
  "label_len": 48,
  
  "architecture": {
    "d_model": 128,
    "n_heads": 16,
    "e_layers": 3,
    "d_layers": 1,
    "d_ff": 256,
    "dropout": 0.1,
    "activation": "gelu"
  },
  
  "patchtst_specific": {
    "patch_len": 16,
    "stride": 8
  },
  
  "training": {
    "batch_size": 16,    # Reduced for memory
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "optimizer": "Adam",
    "loss": "MSE"
  },
  
  "results": {
    "test_mse": 0.5327,  # ⚠️ WORST - 73% degradation
    "test_mae": 0.5256,
    "training_time_min": 30.86,
    "degradation_vs_720": "+73.36%"
  }
}

Exp 4: Time-CAG Baseline (720)

{
  "model": "TransformerLongContext",
  "seq_len": 720,
  "pred_len": 96,
  "label_len": 48,
  
  "architecture": {
    "d_model": 32,       # Best from ETTh1 ablations
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 512,         # d_model × 16
    "dropout": 0.1,
    "activation": "gelu"
  },
  
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "optimizer": "Adam",
    "loss": "MSE"
  },
  
  "results": {
    "test_mse": 0.3282,
    "test_mae": 0.3878,
    "training_time_min": 2.13
  }
}

## Exchange

RAFT 720
{
  "model": "RAFT",
  "seq_len": 720,
  "pred_len": 96,
  "label_len": 48,
  
  "data_config": {
    "data": "custom",           # Exchange uses custom loader
    "features": 8,              # 8 currency pairs
    "freq": "d",                # Daily frequency
    "enc_in": 8,
    "dec_in": 8,
    "c_out": 8
  },
  
  "architecture": {
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "dropout": 0.1,
    "activation": "gelu"
  },
  
  "raft_specific": {
    "n_period": 3,
    "topm": 20
  },
  
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "optimizer": "Adam",
    "loss": "MSE"
  },
  
  "results": {
    "test_mse": 0.0907,  # 🥇 BEST on Exchange
    "test_mae": 0.2079,
    "training_time_min": 0.40
  }
}

PatchTST (720) - Short Context

{
  "model": "PatchTST",
  "seq_len": 720,
  "pred_len": 96,
  "label_len": 48,
  
  "data_config": {
    "data": "custom",
    "features": 8,
    "freq": "d",
    "enc_in": 8,
    "dec_in": 8,
    "c_out": 8
  },
  
  "architecture": {
    "d_model": 128,
    "n_heads": 16,
    "e_layers": 3,
    "d_layers": 1,
    "d_ff": 256,
    "dropout": 0.1,
    "activation": "gelu"
  },
  
  "patchtst_specific": {
    "patch_len": 16,
    "stride": 8
  },
  
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "optimizer": "Adam",
    "loss": "MSE"
  },
  
  "results": {
    "test_mse": 0.0928,
    "test_mae": 0.2176,
    "training_time_min": 2.42
  }
}

PatchTST (3000) - Long Context ⚠️ CATASTROPHIC

{
  "model": "PatchTST",
  "seq_len": 3000,      # 4× longer context
  "pred_len": 96,
  "label_len": 48,
  
  "data_config": {
    "data": "custom",
    "features": 8,
    "freq": "d",
    "enc_in": 8,
    "dec_in": 8,
    "c_out": 8
  },
  
  "architecture": {
    "d_model": 128,
    "n_heads": 16,
    "e_layers": 3,
    "d_layers": 1,
    "d_ff": 256,
    "dropout": 0.1,
    "activation": "gelu"
  },
  
  "patchtst_specific": {
    "patch_len": 16,
    "stride": 8
  },
  
  "training": {
    "batch_size": 8,     # Reduced for memory
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "optimizer": "Adam",
    "loss": "MSE"
  },
  
  "results": {
    "test_mse": 0.3496,  # ⚠️ WORST - 277% degradation!
    "test_mae": 0.4515,
    "training_time_min": 14.09,
    "degradation_vs_720": "+276.77%"  # CATASTROPHIC
  }
}

Exp 8: Time-CAG Baseline (720)

{
  "model": "TransformerLongContext",
  "seq_len": 720,
  "pred_len": 96,
  "label_len": 48,
  
  "data_config": {
    "data": "custom",
    "features": 8,
    "freq": "d",
    "enc_in": 8,
    "dec_in": 8,
    "c_out": 8
  },
  
  "architecture": {
    "d_model": 128,      # Different from ETTh2 (128 vs 32)
    "n_heads": 8,
    "e_layers": 3,       # Different from ETTh2 (3 vs 2)
    "d_layers": 1,
    "d_ff": 512,
    "dropout": 0.1,
    "activation": "gelu"
  },
  
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "optimizer": "Adam",
    "loss": "MSE"
  },
  
  "results": {
    "test_mse": 0.1976,
    "test_mae": 0.3491,
    "training_time_min": 4.29
  }
}