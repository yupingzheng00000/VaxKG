#!/usr/bin/env python3
"""
Test disease→adjuvant edge filtering and leak verification.

Tests:
1. Build full graph with disease→adjuvant edges
2. Simulate inductive disease split (hold out 10 diseases)
3. Verify restrict_context_edges_for_training() removes all 4 edge types touching held-out diseases
4. Verify verify_no_context_leakage() passes (no leaks)
5. Verify verify_no_context_leakage() raises error when leak introduced
"""

import sys
from pathlib import Path

# Add parent directory to path to import from train_disease_ranker
sys.path.insert(0, str(Path(__file__).parent))

import torch
import pandas as pd
import random
from train_disease_ranker import build_graph, restrict_context_edges_for_training, verify_no_context_leakage

def main():
    print("=" * 80)
    print("TEST SUITE: Disease→Adjuvant Edge Filtering & Leak Verification")
    print("=" * 80)
    
    # Load data
    training_samples_csv = Path("data/processed/training_samples.csv")
    disease_adjuvant_csv = Path("data/processed/disease_adjuvant_pairs.csv")
    
    if not training_samples_csv.exists():
        print(f"❌ training_samples.csv not found at {training_samples_csv}")
        return
    
    if not disease_adjuvant_csv.exists():
        print(f"❌ disease_adjuvant_pairs.csv not found at {disease_adjuvant_csv}")
        return
    
    df = pd.read_csv(training_samples_csv)
    
    # Build full graph
    print("\n[Step 1] Building full graph with disease→adjuvant edges...")
    graph, mappings, vaccine_positives, vaccine_negatives, all_adjuvant_indices, disease_positives_lookup, disease_edge_weights = build_graph(
        df, 
        feature_dim=128,  # Dummy feature_dim for testing
        text_encoder=None,  # Skip SapBERT for speed
        text_encoder_config={"disease_adjuvant_csv": str(disease_adjuvant_csv)}
    )
    
    print(f"✅ Graph built:")
    print(f"   - Vaccines: {graph['vaccine'].num_nodes}")
    print(f"   - Diseases: {graph['disease'].num_nodes}")
    print(f"   - Adjuvants: {graph['adjuvant'].num_nodes}")
    print(f"   - Platforms: {graph['platform'].num_nodes}")
    
    # Check disease→adjuvant edges exist
    forward_da = graph["disease", "has_adjuvant", "adjuvant"].edge_index
    reverse_da = graph["adjuvant", "rev_has_adjuvant", "disease"].edge_index
    print(f"   - ('disease', 'has_adjuvant', 'adjuvant') edges: {forward_da.shape[1]}")
    print(f"   - ('adjuvant', 'rev_has_adjuvant', 'disease') edges: {reverse_da.shape[1]}")
    
    # Hold out 10 diseases for inductive split
    print("\n[Step 2] Simulating inductive disease split (hold out 10 diseases)...")
    all_disease_indices = list(range(graph['disease'].num_nodes))
    random.seed(42)
    held_out_diseases = random.sample(all_disease_indices, min(10, len(all_disease_indices)))
    train_disease_indices = [idx for idx in all_disease_indices if idx not in held_out_diseases]
    
    print(f"   - Train diseases: {len(train_disease_indices)}")
    print(f"   - Held-out diseases: {held_out_diseases}")
    
    # Count edges touching held-out diseases BEFORE filtering
    held_out_tensor = torch.tensor(held_out_diseases, dtype=torch.long)
    
    before_counts = {}
    before_counts["vaccine→disease"] = torch.isin(
        graph["vaccine", "for_disease", "disease"].edge_index[1], held_out_tensor
    ).sum().item()
    before_counts["disease→vaccine"] = torch.isin(
        graph["disease", "rev_for_disease", "vaccine"].edge_index[0], held_out_tensor
    ).sum().item()
    before_counts["disease→adjuvant"] = torch.isin(
        graph["disease", "has_adjuvant", "adjuvant"].edge_index[0], held_out_tensor
    ).sum().item()
    before_counts["adjuvant→disease"] = torch.isin(
        graph["adjuvant", "rev_has_adjuvant", "disease"].edge_index[1], held_out_tensor
    ).sum().item()
    
    print(f"\n   Edges touching held-out diseases BEFORE filtering:")
    for edge_type, count in before_counts.items():
        print(f"     - {edge_type}: {count}")
    
    # Filter training graph (keep all vaccines, remove held-out diseases)
    print("\n[Step 3] Filtering training graph (restrict_context_edges_for_training)...")
    train_graph = restrict_context_edges_for_training(
        graph,
        train_vaccine_indices=list(range(graph['vaccine'].num_nodes)),  # Keep all vaccines
        train_disease_indices=train_disease_indices
    )
    
    # Count edges touching held-out diseases AFTER filtering
    after_counts = {}
    after_counts["vaccine→disease"] = torch.isin(
        train_graph["vaccine", "for_disease", "disease"].edge_index[1], held_out_tensor
    ).sum().item()
    after_counts["disease→vaccine"] = torch.isin(
        train_graph["disease", "rev_for_disease", "vaccine"].edge_index[0], held_out_tensor
    ).sum().item()
    after_counts["disease→adjuvant"] = torch.isin(
        train_graph["disease", "has_adjuvant", "adjuvant"].edge_index[0], held_out_tensor
    ).sum().item()
    after_counts["adjuvant→disease"] = torch.isin(
        train_graph["adjuvant", "rev_has_adjuvant", "disease"].edge_index[1], held_out_tensor
    ).sum().item()
    
    print(f"   Edges touching held-out diseases AFTER filtering:")
    for edge_type, count in after_counts.items():
        print(f"     - {edge_type}: {count}")
    
    # Verify all counts are 0
    if all(count == 0 for count in after_counts.values()):
        print(f"\n✅ TEST 1 PASSED: All 4 edge types correctly filtered (0 held-out disease references)")
    else:
        print(f"\n❌ TEST 1 FAILED: Held-out diseases still referenced in training graph:")
        for edge_type, count in after_counts.items():
            if count > 0:
                print(f"     - {edge_type}: {count} leak(s)")
        return
    
    # Test leak verification (should pass)
    print("\n[Step 4] Testing verify_no_context_leakage() with clean graph...")
    try:
        verify_no_context_leakage(
            train_graph,
            heldout_vaccines=[],  # No held-out vaccines
            heldout_diseases=held_out_diseases
        )
        print(f"✅ TEST 2 PASSED: verify_no_context_leakage() passed with clean graph")
    except AssertionError as e:
        print(f"❌ TEST 2 FAILED: verify_no_context_leakage() raised error on clean graph:")
        print(f"   {e}")
        return
    
    # Test leak verification (should fail when we introduce a leak)
    print("\n[Step 5] Testing verify_no_context_leakage() with intentional leak...")
    leaked_graph = train_graph.clone()
    
    # Introduce a leak: add one disease→adjuvant edge for a held-out disease
    leaked_disease_idx = held_out_diseases[0]
    leaked_adjuvant_idx = 0
    leaked_edge = torch.tensor([[leaked_disease_idx], [leaked_adjuvant_idx]], dtype=torch.long)
    
    leaked_graph["disease", "has_adjuvant", "adjuvant"].edge_index = torch.cat([
        leaked_graph["disease", "has_adjuvant", "adjuvant"].edge_index,
        leaked_edge
    ], dim=1)
    
    try:
        verify_no_context_leakage(
            leaked_graph,
            heldout_vaccines=[],
            heldout_diseases=held_out_diseases
        )
        print(f"❌ TEST 3 FAILED: verify_no_context_leakage() did NOT catch intentional leak")
        return
    except AssertionError as e:
        print(f"✅ TEST 3 PASSED: verify_no_context_leakage() correctly caught leak:")
        print(f"   {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: All 3 tests passed ✅")
    print("=" * 80)
    print("✅ restrict_context_edges_for_training() correctly filters 4 edge types")
    print("✅ verify_no_context_leakage() passes with clean graph")
    print("✅ verify_no_context_leakage() catches intentional leaks")
    print("\n👉 Phase 2 (filter & leak verification) COMPLETE. Ready for Phase 3 (training loop).")

if __name__ == "__main__":
    main()
