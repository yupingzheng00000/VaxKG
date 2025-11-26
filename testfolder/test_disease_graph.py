"""Quick test for disease head graph construction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from disease_head_utils import load_disease_positives, sample_disease_batch

# Load data
df = pd.read_csv("data/processed/training_samples.csv")

# Build mappings
disease_names = sorted(set(df["disease_name"].fillna(df["pathogen_name"]).dropna()))
adjuvant_ids = sorted(set(df["adjuvant_vo_id"].dropna()))

mappings = {
    "disease": {name: idx for idx, name in enumerate(disease_names)},
    "adjuvant": {aid: idx for idx, aid in enumerate(adjuvant_ids)},
}

# Test 1: Load disease positives
print("=" * 60)
print("TEST 1: Load disease→adjuvant positives")
print("=" * 60)

dis_pos, dis_weights = load_disease_positives(
    Path("data/processed/disease_adjuvant_pairs.csv"), mappings
)

print(f"✓ Disease positives loaded successfully")
print(f"  {len(dis_pos)} diseases with adjuvant positives")
print(f"  {sum(len(v) for v in dis_pos.values())} total disease→adjuvant pairs")
print(f"  Sample diseases: {list(dis_pos.keys())[:5]}")

# Test 2: Sample disease batch
print("\n" + "=" * 60)
print("TEST 2: Sample disease batch")
print("=" * 60)

all_disease_indices = list(dis_pos.keys())
all_adj_indices = list(range(len(adjuvant_ids)))

batch = sample_disease_batch(
    all_disease_indices,
    dis_pos,
    all_adj_indices,
    batch_size=5,
    num_negatives=10,
    seed=42,
)

print(f"✓ Disease batch sampling works")
print(f"  Sampled {len(batch)} disease queries")

for i, item in enumerate(batch[:3]):
    d_idx = item["disease_idx"]
    n_pos = len(item["pos_indices"])
    n_neg = len(item["neg_indices"])
    print(f"  Query {i+1}: disease_idx={d_idx}, {n_pos} positives, {n_neg} negatives")

# Test 3: Edge weight lookup
print("\n" + "=" * 60)
print("TEST 3: Edge weight lookup")
print("=" * 60)

sample_edges = list(dis_weights.items())[:5]
print(f"✓ Edge weights loaded")
print(f"  Total edges with weights: {len(dis_weights)}")
print(f"  Sample edges: {sample_edges}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
