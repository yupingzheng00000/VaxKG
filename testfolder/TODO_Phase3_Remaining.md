# Phase 3 Remaining TODOs

## ✅ Completed (已完成)

1. ✅ Replace ListNetRankingHead with DualRanker
2. ✅ Update train_one_split() signature to accept disease_positives_lookup and disease_edge_weights
3. ✅ Add disease batch sampling in training loop (inline after vaccine batches)
4. ✅ Add argparse parameters: --lambda-disease, --gamma-mech, --disease-batch-size
5. ✅ Update logging to show separate vax_rank and dis_rank losses
6. ✅ Pass disease data from main() to train_one_split()

## 🔲 TODO: Critical (需要实现的核心功能)

### T1: Implement disease batch loss computation
**Location:** `train_disease_ranker.py`, line ~1577-1580

**Current code (placeholder):**
```python
# TODO: Implement disease batch loss computation
# For now, placeholder to avoid breaking the code
rank_loss_dis = torch.tensor(0.0, device=device)
```

**What to implement:**
```python
# Compute disease→adjuvant ranking loss
from src.disease_head_utils import listnet_loss_disease

disease_loss = 0.0
for query in disease_batch:
    disease_idx = query['disease']
    pos_adjs = query['pos']  # List of positive adjuvant indices
    neg_adjs = query['neg']  # List of negative adjuvant indices
    
    # Get embeddings
    h_dis = embeddings['disease'][disease_idx].unsqueeze(0)  # [1, H]
    h_pos = embeddings['adjuvant'][pos_adjs]  # [P, H]
    h_neg = embeddings['adjuvant'][neg_adjs]  # [N, H]
    h_all = torch.cat([h_pos, h_neg], dim=0)  # [P+N, H]
    
    # Score with disease head
    scores = ranking_head.score_dis(
        h_dis.expand(len(h_all), -1),  # [P+N, H]
        h_all,  # [P+N, H]
        mech_vec=None  # TODO: add mechanism vectors if using mechanism-aware scoring
    )  # [P+N]
    
    # Create labels (positives = 1.0, negatives = 0.0)
    labels = torch.cat([
        torch.ones(len(pos_adjs), device=device),
        torch.zeros(len(neg_adjs), device=device)
    ])
    
    # Compute ListNet loss
    disease_loss += listnet_loss_disease(scores, labels)

rank_loss_dis = disease_loss / len(disease_batch)
```

**Dependencies:**
- `listnet_loss_disease()` already exists in `src/disease_head_utils.py`
- `ranking_head.score_dis()` already exists in DualRanker class

---

### T2: Implement disease evaluation metrics
**Location:** `train_disease_ranker.py`, line ~1641-1644

**Current code (placeholder):**
```python
# Disease head validation (NEW)
val_dis_metrics = {}
if val_diseases and args.lambda_disease > 0:
    # TODO: Implement disease evaluation
    val_dis_metrics = {"ndcg@10": 0.0}
```

**What to implement:**
```python
# Disease head validation
val_dis_metrics = {}
if val_diseases and args.lambda_disease > 0 and disease_positives_lookup:
    val_dis_metrics = evaluate_disease_ranking(
        embeddings,
        val_diseases,
        disease_positives_lookup,
        candidate_ids,
        k_values=(5, 10)
    )
```

**Need to create new function:**
```python
def evaluate_disease_ranking(
    embeddings: Dict[str, Tensor],
    disease_indices: Sequence[int],
    disease_positives_lookup: Mapping[int, Sequence[int]],
    all_adjuvant_indices: Sequence[int],
    k_values: Tuple[int, ...] = (5, 10),
) -> Dict[str, float]:
    """
    Evaluate disease→adjuvant ranking performance.
    
    Returns metrics: ndcg@5, ndcg@10, recall@5, recall@10, map@5, map@10
    """
    # Similar to evaluate_ranking() but for disease queries
    # Use ranking_head.score_dis() instead of score_vax()
    pass
```

**Location to add:** After `evaluate_ranking()` function (~line 850)

---

### T3: Add final disease metrics to results
**Location:** `train_disease_ranker.py`, line ~1688-1701

**Current code:**
```python
results["ranking_train"] = train_metrics
if val_metrics:
    results["ranking_val"] = val_metrics
if test_metrics:
    results["ranking_test"] = test_metrics
if link_metrics:
    results["link_prediction"] = link_metrics
return results
```

**What to add:**
```python
# Add disease head metrics (NEW)
if train_diseases and disease_positives_lookup:
    disease_train_metrics = evaluate_disease_ranking(
        embeddings, train_diseases, disease_positives_lookup, candidate_ids, (5, 10)
    )
    results["disease_ranking_train"] = disease_train_metrics

if val_diseases and disease_positives_lookup:
    disease_val_metrics = evaluate_disease_ranking(
        embeddings, val_diseases, disease_positives_lookup, candidate_ids, (5, 10)
    )
    results["disease_ranking_val"] = disease_val_metrics

if test_diseases and disease_positives_lookup:
    disease_test_metrics = evaluate_disease_ranking(
        embeddings, test_diseases, disease_positives_lookup, candidate_ids, (5, 10)
    )
    results["disease_ranking_test"] = disease_test_metrics

results["ranking_train"] = train_metrics
# ... rest of existing code
```

---

## 🔲 TODO: Optional Enhancements (可选优化)

### T4: Mechanism-aware scoring
**Current:** `mech_in_dim = None` (mechanism-aware scoring disabled)

**To enable:**
1. Compute mechanism vectors during graph construction (in `build_graph()`)
2. Store as graph['adjuvant'].mech attribute
3. Set `mech_in_dim = receptor_dim + profile_dim` when creating DualRanker
4. Pass `mech_vec=graph['adjuvant'].mech` to `ranking_head.score_dis()`

**Rationale:** Can be deferred to ablation study (first verify baseline dual-head works)

---

### T5: Hard negatives for disease batch sampling
**Current:** `sample_negatives_mixed()` in disease_head_utils.py already supports hard negatives

**Status:** Already implemented, but need to ensure VO class lookup is passed correctly

**Check:** Verify `build_vo_class_lookup()` is called and vo_class_lookup is available

---

### T6: mAP@K metric
**Current:** Not implemented

**To add:** Extend `evaluate_disease_ranking()` to compute mAP@5 and mAP@10

**Implementation:** Use `mean_average_precision_at_k()` helper from disease.instructions.md Section 7

---

## 🔲 TODO: Testing & Validation (测试验证)

### T7: Backward compatibility test
**Goal:** Verify vaccine head NDCG@10 does NOT degrade by >5%

**Test:**
```bash
# Train original train_ranker.py (baseline)
python train_ranker.py --epochs 50 --output-dir artifacts_baseline

# Train new train_disease_ranker.py (dual-head)
python train_disease_ranker.py --epochs 50 --lambda-disease 1.0 --output-dir artifacts_dual

# Compare vaccine NDCG@10
python -c "
import json
baseline = json.load(open('artifacts_baseline/results/transductive.json'))
dual = json.load(open('artifacts_dual/results/transductive.json'))

baseline_ndcg = baseline['ranking_val']['ndcg@10']
dual_ndcg = dual['ranking_val']['ndcg@10']

degradation = baseline_ndcg - dual_ndcg
print(f'Baseline: {baseline_ndcg:.4f}')
print(f'Dual-head: {dual_ndcg:.4f}')
print(f'Degradation: {degradation:.4f} ({degradation/baseline_ndcg*100:.1f}%)')

assert degradation <= 0.05, 'Vaccine head degraded by >5%!'
print('✅ Backward compatibility test PASSED')
"
```

---

### T8: Disease head smoke test
**Goal:** Verify disease head produces non-zero NDCG@10 on validation set

**Test:**
```bash
python train_disease_ranker.py \
  --epochs 20 \
  --lambda-disease 1.0 \
  --disease-batch-size 16 \
  --output-dir artifacts_smoke_test

python -c "
import json
metrics = json.load(open('artifacts_smoke_test/results/transductive.json'))

dis_ndcg = metrics['disease_ranking_val']['ndcg@10']
print(f'Disease NDCG@10: {dis_ndcg:.4f}')

assert dis_ndcg > 0, 'Disease head produces zero NDCG!'
print('✅ Disease head smoke test PASSED')
"
```

---

## Implementation Priority (实施优先级)

**Phase 3a (Critical, must complete before training):**
1. T1: Disease batch loss computation ⚠️ **HIGHEST PRIORITY**
2. T2: Disease evaluation metrics ⚠️ **HIGHEST PRIORITY**
3. T3: Add disease metrics to results

**Phase 3b (Validation, run after 3a):**
4. T7: Backward compatibility test
5. T8: Disease head smoke test

**Phase 3c (Optional, can defer to ablation study):**
6. T4: Mechanism-aware scoring
7. T5: Hard negatives verification
8. T6: mAP@K metric

---

## Next Steps

1. **Implement T1 (disease batch loss)** — This is the blocker for training
2. **Implement T2 (disease evaluation)** — Needed for validation metrics
3. **Implement T3 (final metrics)** — Needed for results output
4. **Run T7 + T8 (tests)** — Verify everything works

After completing T1-T3, the training loop should be fully functional and ready to run.
