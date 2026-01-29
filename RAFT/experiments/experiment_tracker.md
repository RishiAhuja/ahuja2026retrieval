# Experiment Tracking for Workshop Paper

## Required Experiments

### Phase 1: Core Model Comparisons (For Graph 1 & 2)
- [x] **Experiment 0**: RAFT Baseline - **COMPLETED** (MSE: 0.379)
- [ ] **Experiment 1**: Time-CAG v1 (Raw, seq_len=3000, d_model=256)
- [ ] **Experiment 2**: Time-CAG v2 (Patched, seq_len=3000, patch_size=12)
- [ ] **Experiment 3**: Time-CAG v3 (Less is More, d_model=32, dropout=0.3)
- [ ] **Experiment 4**: Time-CAG v4 (Sweet Spot, d_model=128, seq_len=1440)

### Phase 2: Context Length Analysis (For Graph 3)
- [ ] **Experiment 5**: Time-CAG @ seq_len=720 (Same as RAFT)
- [ ] **Experiment 6**: Time-CAG @ seq_len=1440 (Mid-range)
- [ ] **Experiment 7**: Time-CAG @ seq_len=3000 (Full context)

### Phase 3: Ablation Study (For Graph 6)
Grid: d_model × e_layers
- [ ] **Experiment 8**: d_model=32, e_layers=1
- [ ] **Experiment 9**: d_model=32, e_layers=2
- [ ] **Experiment 10**: d_model=32, e_layers=3
- [ ] **Experiment 11**: d_model=32, e_layers=4
- [ ] **Experiment 12**: d_model=64, e_layers=1
- [ ] **Experiment 13**: d_model=64, e_layers=2
- [ ] **Experiment 14**: d_model=64, e_layers=3
- [ ] **Experiment 15**: d_model=64, e_layers=4
- [ ] **Experiment 16**: d_model=128, e_layers=1
- [ ] **Experiment 17**: d_model=128, e_layers=2
- [ ] **Experiment 18**: d_model=128, e_layers=3
- [ ] **Experiment 19**: d_model=128, e_layers=4
- [ ] **Experiment 20**: d_model=256, e_layers=1
- [ ] **Experiment 21**: d_model=256, e_layers=2
- [ ] **Experiment 22**: d_model=256, e_layers=3
- [ ] **Experiment 23**: d_model=256, e_layers=4

## Results Summary

| Exp | Model | seq_len | d_model | e_layers | dropout | Train MSE | Val MSE | Test MSE | Time (min) | Status |
|-----|-------|---------|---------|----------|---------|-----------|---------|----------|------------|--------|
| 0   | RAFT  | 720     | -       | -        | -       | -         | -       | 0.379    | ~15        | ✅ Done |
| 1   | CAG-v1| 3000    | 256     | 3        | 0.1     | -         | -       | -        | -          | ⏳ Next |
| 2   | CAG-v2| 3000    | 256     | 3        | 0.1     | -         | -       | -        | -          | 📋 Todo |
| 3   | CAG-v3| 3000    | 32      | 2        | 0.3     | -         | -       | -        | -          | 📋 Todo |
| 4   | CAG-v4| 1440    | 128     | 2        | 0.15    | -         | -       | -        | -          | 📋 Todo |

---

**Note**: All Time-CAG experiments use:
- Dataset: ETTh1
- pred_len: 96
- batch_size: Adaptive based on seq_len
- num_workers: 0 (Mac compatibility)
- Device: MPS (Mac GPU)
