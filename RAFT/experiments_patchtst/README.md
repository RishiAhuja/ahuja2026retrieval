# PatchTST Long Context Experiments

## Objective
Test whether **state-of-the-art PatchTST** also degrades with longer context, or if its patching mechanism solves the long-context problem.

## Hypothesis to Test
**H1**: PatchTST degrades with longer context (like vanilla Transformer)
- **Result**: RAFT's retrieval is superior to both vanilla AND modern transformers
- **Impact**: 🔥🔥🔥 Major finding - challenges SOTA

**H2**: PatchTST handles long context well
- **Result**: Patching helps, but retrieval still offers efficiency/accuracy tradeoffs
- **Impact**: 🔥🔥 Still publishable - different angle

## Experiments Needed

### Core Experiments (MUST RUN)
| Exp # | Model | seq_len | Comparison Point | Why? |
|-------|-------|---------|------------------|------|
| **1** | PatchTST | 720 | RAFT baseline (0.379) | Fair comparison at same context |
| **2** | PatchTST | 3000 | Vanilla Transformer worst case (1.323) | Test if PatchTST also degrades |

### Extended Experiments (NICE TO HAVE)
| Exp # | Model | seq_len | Why? |
|-------|-------|---------|------|
| **3** | PatchTST | 1440 | Mid-range degradation check |
| **4** | PatchTST | 336 | Standard benchmark length |
| **5** | PatchTST | 96 | Shortest context (speed test) |

## Expected Outcomes & Paper Impact

### Scenario A: PatchTST degrades (MSE increases 720→3000)
**Finding**: Even modern architectures suffer from long-context curse
```
PatchTST (720):  ~0.45 MSE
PatchTST (3000): ~0.80 MSE  ← Degradation!
RAFT (720):      0.379 MSE  ← Still best
```
**Paper claim**: "Retrieval > Patching for long sequences"

### Scenario B: PatchTST stable (MSE ~constant)
**Finding**: Patching helps, but retrieval is more efficient
```
PatchTST (720):  ~0.42 MSE
PatchTST (3000): ~0.44 MSE  ← Stable
RAFT (720):      0.379 MSE  ← Slightly better + 4× less memory
```
**Paper claim**: "Retrieval offers accuracy + efficiency gains"

### Scenario C: PatchTST beats RAFT
**Finding**: Need to reframe paper
```
PatchTST (3000): ~0.35 MSE  ← Beats RAFT
RAFT (720):      0.379 MSE
```
**Paper claim**: "Combining retrieval + patching (future work)"

## Priority Order

### Phase 1: Critical Validation (1-2 days)
1. ✅ **Exp 1**: PatchTST seq_len=720 
2. ✅ **Exp 2**: PatchTST seq_len=3000
   
**Decision point**: If degradation seen → proceed to Phase 2
**Decision point**: If stable/better → reframe paper angle

### Phase 2: Full Comparison (if Phase 1 shows degradation)
3. ✅ **Exp 3**: PatchTST seq_len=1440
4. ✅ **Exp 4**: PatchTST seq_len=336 (optional)

## Implementation Plan

### Step 1: Get PatchTST Code
```bash
# Option A: Use Time-Series-Library (recommended)
git clone https://github.com/thuml/Time-Series-Library.git
cp -r Time-Series-Library/models/PatchTST.py ./models/

# Option B: Use original PatchTST repo
git clone https://github.com/yuqinie98/PatchTST.git
```

### Step 2: Integrate PatchTST
- Add to `models/__init__.py`
- Create experiment configs
- Test one run to verify setup

### Step 3: Run Experiments
- Start with Exp 1 & 2 (critical)
- Generate training logs (real losses)
- Compare against existing results

### Step 4: Update Graphs
- Add PatchTST to comparison graphs
- Generate new visualizations
- Update paper narrative

## Timeline

- **Day 1 Morning**: Setup PatchTST integration
- **Day 1 Afternoon**: Run Exp 1 (seq_len=720)
- **Day 2 Morning**: Run Exp 2 (seq_len=3000)
- **Day 2 Afternoon**: Analyze results, decide on full experiments
- **Day 3**: Run remaining experiments if needed
- **Day 4**: Generate graphs, update paper

## Success Metrics

**Minimum viable**: 2 experiments (720, 3000)
**Full comparison**: 4 experiments (720, 1440, 3000, 336)
**Extended study**: 5+ experiments with ablations

## Files Generated

Each experiment will produce:
1. Training logs: `experiments_patchtst/results/training_logs/patchtst_sl{X}_losses.json`
2. Results: `experiments_patchtst/results/patchtst_sl{X}_results.json`
3. Checkpoints: `checkpoints/patchtst_sl{X}/`

## Next Steps

1. [ ] Clone PatchTST implementation
2. [ ] Integrate into RAFT codebase
3. [ ] Create exp_patchtst_sl720.py
4. [ ] Create exp_patchtst_sl3000.py
5. [ ] Run Phase 1 experiments
6. [ ] Analyze and decide next steps
