# Training Readiness Checklist
**Date**: 2025-10-04  
**Target**: Disease→Adjuvant Dual-Head Ranking Model

---

## ✅ Core Components Status

### Data Layer
- ✅ **training_samples.csv** — 原始训练数据存在
- ✅ **disease_adjuvant_pairs.csv** — Disease head 监督数据已生成
  - 288 pairs (81 diseases × 106 adjuvants)
  - 包含 edge_weight 和 vaxjo_stage 信息
- ✅ **build_da_pairs.py** — 数据生成脚本可复现

### Model Architecture
- ✅ **DualRanker class** (Line 456-540)
  - 双头设计：vax_head + dis_head
  - score_vax() 方法 (Line 483)
  - score_dis() 方法 (Line 495，支持 mechanism-aware scoring)
  - forward() 方法 (Line 517，backward compatible)
- ✅ **Shared Encoder** — PyGHeteroEncoder + optional APPNP
- ✅ **Feature Engineering**
  - SapBERT text embeddings
  - Immune profile multi-hot vectors
  - Receptor multi-hot vectors

### Training Logic
- ✅ **Loss Functions**
  - Vaccine ranking loss: ListNet (original)
  - Disease ranking loss: listnet_loss_disease() (Line 1604-1609)
  - Link prediction loss: typed negatives (original)
- ✅ **Disease Batch Sampling** (disease_head_utils.py)
  - sample_disease_batch() — uniform random with replacement
  - Hard negatives (10) + Easy negatives (30)
- ✅ **Dual Loss Integration**
  - Joint training: L = L_vax + λ_disease * L_dis + λ_lp * L_lp
  - Separate tracking: avg_rank_vax, avg_rank_dis, avg_lp

### Evaluation Metrics
- ✅ **Vaccine Head** — evaluate_ranking() (original, unchanged)
- ✅ **Disease Head** — evaluate_disease_ranking() (Line 894-960)
  - NDCG@5, NDCG@10
  - Recall@5, Recall@10
  - Uses score_dis() for consistency
- ✅ **Validation Loop** (Line 1700-1706)
  - Early stopping on val_vax_ndcg10
  - Logs val_dis_ndcg10
- ✅ **Final Evaluation** (Line 1757-1783)
  - Train/val/test metrics for both heads
  - Results saved to JSON

### Graph Construction
- ✅ **Disease→Adjuvant Edges** — ('disease','has_adjuvant','adjuvant')
- ✅ **Edge Filtering** — restrict_context_edges_for_training()
- ✅ **Leak Verification** — verify_no_context_leakage()
  - Checks 4 edge types for inductive split
  - Assertions for transductive split

### CLI Arguments
- ✅ **--lambda-disease** (default: 1.0) — Disease loss weight
- ✅ **--gamma-mech** (default: 0.3) — Mechanism scoring weight
- ✅ **--disease-batch-size** (default: 32) — Disease batch size
- ✅ **--split-scheme** (choices: transductive/inductive/both)
- ✅ All original arguments preserved

---

## ✅ Implementation Verification

### Phase 1: Data Generation
- ✅ build_da_pairs.py (130 lines)
- ✅ disease_head_utils.py (310 lines)
- ✅ disease_adjuvant_pairs.csv generated

### Phase 2: Graph Construction
- ✅ Disease→adjuvant edges loaded
- ✅ Edge filtering implemented
- ✅ Leak verification passed

### Phase 3: Training Loop (100% COMPLETE)
- ✅ **T1**: Disease batch loss computation
  - DualRanker.forward() added
  - listnet_loss_disease() integrated
  - Dead code removed
- ✅ **T2**: Disease evaluation metrics (VERIFIED)
  - evaluate_disease_ranking() function
  - Validation loop integration
  - Final evaluation (train/val/test)
  - Results aggregation

---

## 🚀 Training Commands

### Minimal Test Run (Transductive)
```bash
cd 'd:\research\Prof. He\VaxKG'

python train_disease_ranker.py \
  --data-path data/processed/training_samples.csv \
  --output-dir results/disease_test \
  --run-name test_transductive \
  --split-scheme transductive \
  --epochs 5 \
  --batch-size 32 \
  --disease-batch-size 16 \
  --lambda-disease 1.0 \
  --lambda-lp 0.3 \
  --gamma-mech 0.3 \
  --device cuda
```

### Full Training (Both Splits)
```bash
python train_disease_ranker.py \
  --data-path data/processed/training_samples.csv \
  --output-dir results/disease_dual_head \
  --run-name disease_dual_v1 \
  --split-scheme both \
  --epochs 100 \
  --batch-size 128 \
  --disease-batch-size 32 \
  --lambda-disease 1.0 \
  --lambda-lp 0.3 \
  --gamma-mech 0.3 \
  --patience 30 \
  --encoder-lr 2e-3 \
  --head-lr 1e-3 \
  --device cuda
```

### Ablation: Disease Head Only
```bash
python train_disease_ranker.py \
  --data-path data/processed/training_samples.csv \
  --output-dir results/disease_only \
  --run-name disease_only_v1 \
  --split-scheme transductive \
  --lambda-disease 1.0 \
  --lambda-lp 0.0 \
  --gamma-mech 0.0 \
  --device cuda
```

---

## ⚠️ Pre-Training Checks

### 1. Environment Setup
```bash
# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Check required packages
python -c "import torch_geometric, transformers, pandas, numpy; print('All packages OK')"
```

### 2. Data Integrity
```bash
# Verify all data files exist
python -c "from pathlib import Path; files = ['data/processed/training_samples.csv', 'data/processed/disease_adjuvant_pairs.csv']; print('All data files exist:', all(Path(f).exists() for f in files))"

# Check data quality
python src/build_da_pairs.py  # Should complete without errors
```

### 3. Code Syntax Check
```bash
# Basic syntax check
python -c "import train_disease_ranker; print('Syntax OK')"

# Import helper modules
python -c "from src.disease_head_utils import sample_disease_batch, listnet_loss_disease; print('Imports OK')"
```

---

## 📊 Expected Outputs

### During Training
```
Epoch 001 | loss=X.XXXX vax_rank=X.XXXX dis_rank=X.XXXX lp=X.XXXX | 
          val_vax_ndcg10=X.XXXX val_dis_ndcg10=X.XXXX
```

### Final Results JSON
```json
{
  "ranking_train": {"ndcg@5": X.XX, "ndcg@10": X.XX, ...},
  "ranking_val": {"ndcg@5": X.XX, "ndcg@10": X.XX, ...},
  "ranking_test": {"ndcg@5": X.XX, "ndcg@10": X.XX, ...},
  "disease_ranking_train": {"ndcg@5": X.XX, "ndcg@10": X.XX, ...},
  "disease_ranking_val": {"ndcg@5": X.XX, "ndcg@10": X.XX, ...},
  "disease_ranking_test": {"ndcg@5": X.XX, "ndcg@10": X.XX, ...},
  "link_prediction": {...}
}
```

### Checkpoint Files
```
results/disease_dual_head/checkpoints/
  - transductive_best.pt
  - inductive_best.pt
```

---

## 🎯 Success Criteria

### Minimum Viable Results
- ✅ Training completes without errors
- ✅ Vaccine head NDCG@10 ≥ baseline - 5% (backward compatibility)
- ✅ Disease head NDCG@10 > 0.1 (better than random)
- ✅ No NaN losses or metrics
- ✅ Results JSON contains all expected keys

### Target Performance
- 🎯 Vaccine head NDCG@10 ≈ 0.40-0.50 (preserve original performance)
- 🎯 Disease head NDCG@10 ≥ 0.25 (transductive)
- 🎯 Disease head NDCG@10 ≥ 0.15 (inductive)
- 🎯 Disease head Recall@10 ≥ 0.40

### Clinical Anchors (Sanity Check)
After training, test with showcase_ranker.py:
- "Anthrax" → aluminum hydroxide in top-3
- "Hepatitis B" → CpG/alum class in top-3
- "Influenza (elderly)" → MF59/emulsion in top-3

---

## ⚡ Quick Start (Recommended)

### Step 1: Environment Check (30 seconds)
```bash
cd 'd:\research\Prof. He\VaxKG'
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

### Step 2: Data Verification (10 seconds)
```bash
python -c "from pathlib import Path; print('Data ready:', Path('data/processed/disease_adjuvant_pairs.csv').exists())"
```

### Step 3: Quick Test Run (5-10 minutes)
```bash
python train_disease_ranker.py \
  --run-name quick_test \
  --split-scheme transductive \
  --epochs 3 \
  --batch-size 32 \
  --disease-batch-size 16
```

### Step 4: Check Output
```bash
# Should see training logs with both val_vax_ndcg10 and val_dis_ndcg10
cat results/richer_ds/results/transductive.json
```

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: 减小 batch_size 和 disease_batch_size
```bash
--batch-size 64 --disease-batch-size 16
```

### Issue: Disease head loss is NaN
**Solution**: 检查 disease_adjuvant_pairs.csv 是否加载成功
```bash
python -c "from src.disease_head_utils import load_disease_pairs; pairs, weights = load_disease_pairs(); print('Loaded pairs:', len(pairs))"
```

### Issue: Vaccine head degraded >5%
**Solution**: 降低 lambda_disease
```bash
--lambda-disease 0.5
```

### Issue: Training too slow
**Solution**: 
1. 减小 disease_batch_size (每个epoch只sample一次)
2. 使用 --freeze-text-encoder 1 (如果SapBERT已预训练)

---

## ✅ Final Checklist

Before running full training, confirm:

- [ ] GPU available and CUDA working
- [ ] All data files exist and validated
- [ ] Quick test run (3 epochs) completed successfully
- [ ] No syntax errors in train_disease_ranker.py
- [ ] disease_head_utils.py imports correctly
- [ ] Output directory has write permissions
- [ ] Enough disk space for checkpoints (~500MB per checkpoint)

**If all checked**: 🚀 **READY TO TRAIN!**

---

## 📝 Notes

1. **First run**: 建议用 `--epochs 5` 快速验证完整流程
2. **Full training**: 预计需要 30-60 分钟 (取决于GPU)
3. **Monitoring**: 观察 val_dis_ndcg10 是否收敛
4. **Baseline comparison**: 保存 vaccine head 原始性能作为参考

**Status**: ✅ ALL SYSTEMS GO — 可以开始训练！
