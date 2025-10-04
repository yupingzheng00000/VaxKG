"""
Build disease→adjuvant positive pairs from vaccine data.

Projects vaccine→adjuvant relationships through disease to create
disease→adjuvant supervision edges for the disease head.

Usage:
    python src/build_da_pairs.py

Output:
    data/processed/disease_adjuvant_pairs.csv
    Columns: disease_key, adjuvant_vo_id, edge_weight, vaxjo_stage
"""

import pandas as pd
from pathlib import Path
from collections import Counter

# Input/output paths
IN = Path("data/processed/training_samples.csv")
OUT = Path("data/processed/disease_adjuvant_pairs.csv")


def build_disease_key(df: pd.DataFrame) -> pd.Series:
    """Build disease key (prefer disease_name; fallback pathogen_name)."""
    disease_key = df["disease_name"].fillna(df["pathogen_name"]).astype(str).str.strip()
    return disease_key


def extract_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Extract disease→adjuvant positive pairs with edge weights."""
    # Build disease key
    dkey = build_disease_key(df)
    
    # Create (disease_key, adjuvant_vo_id) pairs
    pairs = pd.DataFrame({
        "disease_key": dkey,
        "adjuvant_vo_id": df["adjuvant_vo_id"].astype(str).str.strip()
    })
    
    # Drop missing values (vaccines without disease/pathogen info)
    pairs = pairs.dropna().drop_duplicates()
    
    # Additional check: filter out 'nan' string (shouldn't happen but be defensive)
    pairs = pairs[pairs["disease_key"] != "nan"]
    
    # Count frequency (how many vaccines support each pair)
    freq = pairs.value_counts().reset_index(name="edge_weight")
    freq.columns = ["disease_key", "adjuvant_vo_id", "edge_weight"]
    
    return freq


def attach_stage_info(freq: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach vaxjo_stage information (most frequent stage per adjuvant)."""
    # Extract adjuvant stage info from training_samples
    stage_df = df[["adjuvant_vo_id", "vaxjo_stage"]].dropna().copy()
    stage_df["adjuvant_vo_id"] = stage_df["adjuvant_vo_id"].astype(str)
    
    # Get most frequent stage per adjuvant (prefer Licensed > Clinical > Research)
    stage_priority = {"Licensed": 3, "Clinical Trial": 2, "Research": 1}
    stage_df["priority"] = stage_df["vaxjo_stage"].map(stage_priority).fillna(0)
    stage_df = stage_df.sort_values("priority", ascending=False)
    stage_df = stage_df.drop_duplicates("adjuvant_vo_id", keep="first")
    stage_df = stage_df[["adjuvant_vo_id", "vaxjo_stage"]]
    
    # Merge with frequency table
    out = freq.merge(stage_df, on="adjuvant_vo_id", how="left")
    
    return out


def validate_output(out: pd.DataFrame) -> None:
    """Validate output DataFrame."""
    # Check for duplicates
    duplicates = out.duplicated(subset=["disease_key", "adjuvant_vo_id"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate pairs!"
    
    # Check edge weights
    assert (out["edge_weight"] >= 1).all(), "All edge_weight must be >= 1"
    
    # Check no missing keys
    assert out["disease_key"].notna().all(), "Missing disease_key values"
    assert out["adjuvant_vo_id"].notna().all(), "Missing adjuvant_vo_id values"
    
    print(f"✓ Validation passed: {len(out)} unique disease→adjuvant pairs")


def main():
    """Main execution."""
    print(f"Loading training data from {IN}...")
    df = pd.read_csv(IN)
    print(f"  Loaded {len(df)} rows")
    
    print("\nExtracting disease→adjuvant pairs...")
    freq = extract_pairs(df)
    print(f"  Found {len(freq)} unique pairs")
    
    print("\nAttaching stage information...")
    out = attach_stage_info(freq, df)
    
    print("\nValidating output...")
    validate_output(out)
    
    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total pairs: {len(out)}")
    print(f"Unique diseases: {out['disease_key'].nunique()}")
    print(f"Unique adjuvants: {out['adjuvant_vo_id'].nunique()}")
    print(f"\nEdge weight distribution:")
    print(out["edge_weight"].describe())
    print(f"\nStage distribution:")
    print(out["vaxjo_stage"].value_counts())
    print(f"\nTop 10 diseases by adjuvant count:")
    print(out["disease_key"].value_counts().head(10))
    
    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\n✓ Wrote {len(out)} pairs to {OUT}")


if __name__ == "__main__":
    main()
