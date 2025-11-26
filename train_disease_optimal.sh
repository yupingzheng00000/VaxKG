#!/bin/bash
################################################################################
# VaxKG Disease→Adjuvant Dual-Head Training Script (Optimal Configuration)
# 
# Features:
# - Dual-head architecture (Vaccine + Disease)
# - SapBERT embeddings (best text encoder)
# - Both transductive and inductive splits
# - Mechanism-aware scoring (γ=0.3)
# - Full training (100 epochs with early stopping)
# 
# Hardware requirements:
# - GPU with 8GB+ VRAM (recommended)
# - 16GB+ RAM
# - ~2GB disk space for outputs
#
# Estimated time: 30-60 minutes on GPU, 2-4 hours on CPU
################################################################################

set -e  # Exit on error

echo "=================================="
echo "VaxKG Optimal Training Pipeline"
echo "=================================="
echo ""

# Configuration
DATA_DIR="data"
PROCESSED_DIR="data/processed"
OUTPUT_DIR="results/disease_sapbert_optimal"
CHECKPOINT="cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"

# Training hyperparameters (optimal)
EPOCHS=100
BATCH_SIZE=128
DISEASE_BATCH_SIZE=32
LAMBDA_DISEASE=1.0
LAMBDA_LP=0.3
GAMMA_MECH=0.3
ENCODER_LR=2e-3
HEAD_LR=1e-3
PATIENCE=30

# Device auto-detection
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "✓ CUDA detected - using GPU acceleration"
else
    DEVICE="cpu"
    echo "⚠ No CUDA - using CPU (training will be slower)"
fi

echo ""
echo "Configuration:"
echo "  - Model: Dual-head (Vaccine + Disease)"
echo "  - Text Encoder: SapBERT (mean-token pooling)"
echo "  - Splits: Both transductive and inductive"
echo "  - Epochs: $EPOCHS (early stopping patience: $PATIENCE)"
echo "  - Batch sizes: vaccine=$BATCH_SIZE, disease=$DISEASE_BATCH_SIZE"
echo "  - Loss weights: λ_disease=$LAMBDA_DISEASE, λ_lp=$LAMBDA_LP, γ_mech=$GAMMA_MECH"
echo "  - Device: $DEVICE"
echo "  - Output: $OUTPUT_DIR"
echo ""

# Step 0: Environment check
echo "Step 0/4: Checking environment..."
echo "--------------------------------"

# Check Python
if ! command -v python &> /dev/null; then
    echo "✗ Python not found. Please install Python 3.8+"
    exit 1
fi
echo "✓ Python: $(python --version)"

# Check required packages
MISSING_PACKAGES=()

if ! python -c "import torch" 2>/dev/null; then
    MISSING_PACKAGES+=("torch")
fi

if ! python -c "import torch_geometric" 2>/dev/null; then
    MISSING_PACKAGES+=("torch-geometric")
fi

if ! python -c "import transformers" 2>/dev/null; then
    MISSING_PACKAGES+=("transformers")
fi

if ! python -c "import pandas" 2>/dev/null; then
    MISSING_PACKAGES+=("pandas")
fi

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "✗ Missing packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Install with:"
    echo "  pip install torch torch-geometric transformers pandas numpy scikit-learn"
    exit 1
fi

echo "✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "✓ PyTorch Geometric: $(python -c 'import torch_geometric; print(torch_geometric.__version__)')"
echo "✓ Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "✓ All dependencies installed"
echo ""

# Step 1: Data preparation
echo "Step 1/4: Preparing training data..."
echo "-------------------------------------"

if [ ! -f "$PROCESSED_DIR/training_samples.csv" ]; then
    echo "Running prepare_training_data.py..."
    python prepare_training_data.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$PROCESSED_DIR"
    echo "✓ Training samples generated"
else
    echo "✓ Training samples already exist"
fi

# Step 2: Generate disease→adjuvant pairs
echo ""
echo "Step 2/4: Generating disease→adjuvant pairs..."
echo "-----------------------------------------------"

if [ ! -f "$PROCESSED_DIR/disease_adjuvant_pairs.csv" ]; then
    echo "Running build_da_pairs.py..."
    python src/build_da_pairs.py
    echo "✓ Disease-adjuvant pairs generated"
else
    echo "✓ Disease-adjuvant pairs already exist"
fi

# Display data statistics
echo ""
echo "Data statistics:"
PAIR_COUNT=$(python -c "import pandas as pd; print(len(pd.read_csv('$PROCESSED_DIR/disease_adjuvant_pairs.csv')))")
DISEASE_COUNT=$(python -c "import pandas as pd; print(pd.read_csv('$PROCESSED_DIR/disease_adjuvant_pairs.csv')['disease_key'].nunique())")
ADJUVANT_COUNT=$(python -c "import pandas as pd; print(pd.read_csv('$PROCESSED_DIR/disease_adjuvant_pairs.csv')['adjuvant_vo_id'].nunique())")
echo "  - Disease-adjuvant pairs: $PAIR_COUNT"
echo "  - Unique diseases: $DISEASE_COUNT"
echo "  - Unique adjuvants: $ADJUVANT_COUNT"
echo ""

# Step 3: Train the dual-head model
echo "Step 3/4: Training dual-head model with SapBERT..."
echo "--------------------------------------------------"
echo "This will take 30-60 minutes on GPU (2-4 hours on CPU)"
echo "Training progress will be displayed below..."
echo ""

python train_disease_ranker.py \
    --data-path "$PROCESSED_DIR/training_samples.csv" \
    --output-dir "$OUTPUT_DIR" \
    --split-scheme both \
    --text-encoder-checkpoint "$CHECKPOINT" \
    --text-encoder-pooling mean \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --disease-batch-size "$DISEASE_BATCH_SIZE" \
    --lambda-disease "$LAMBDA_DISEASE" \
    --lambda-lp "$LAMBDA_LP" \
    --gamma-mech "$GAMMA_MECH" \
    --encoder-lr "$ENCODER_LR" \
    --head-lr "$HEAD_LR" \
    --patience "$PATIENCE" \
    --device "$DEVICE" \
    --seed 42

echo ""
echo "✓ Training completed!"
echo ""

# Step 4: Display results
echo "Step 4/4: Training results summary"
echo "----------------------------------"

# Check if results exist
if [ -f "$OUTPUT_DIR/results/transductive.json" ]; then
    echo ""
    echo "=== Transductive Split ==="
    python -c "
import json
with open('$OUTPUT_DIR/results/transductive.json') as f:
    results = json.load(f)
    
print('\nVaccine→Adjuvant (original):')
if 'ranking_val' in results:
    for k, v in results['ranking_val'].items():
        print(f'  {k}: {v:.4f}')

print('\nDisease→Adjuvant (NEW):')
if 'disease_ranking_val' in results:
    for k, v in results['disease_ranking_val'].items():
        print(f'  {k}: {v:.4f}')
"
fi

if [ -f "$OUTPUT_DIR/results/inductive.json" ]; then
    echo ""
    echo "=== Inductive Split ==="
    python -c "
import json
with open('$OUTPUT_DIR/results/inductive.json') as f:
    results = json.load(f)
    
print('\nVaccine→Adjuvant (original):')
if 'ranking_val' in results:
    for k, v in results['ranking_val'].items():
        print(f'  {k}: {v:.4f}')

print('\nDisease→Adjuvant (NEW):')
if 'disease_ranking_val' in results:
    for k, v in results['disease_ranking_val'].items():
        print(f'  {k}: {v:.4f}')
"
fi

echo ""
echo "=================================="
echo "Training Pipeline Complete! ✓"
echo "=================================="
echo ""
echo "Outputs saved to: $OUTPUT_DIR/"
echo ""
echo "Checkpoints:"
echo "  - $OUTPUT_DIR/checkpoints/transductive_best.pt"
echo "  - $OUTPUT_DIR/checkpoints/inductive_best.pt"
echo ""
echo "Results:"
echo "  - $OUTPUT_DIR/results/transductive.json"
echo "  - $OUTPUT_DIR/results/inductive.json"
echo ""
echo "Next steps:"
echo "  1. Try disease query (NEW!):"
echo ""
echo "     python showcase_ranker.py \\"
echo "         --checkpoint $OUTPUT_DIR/checkpoints/transductive_best.pt \\"
echo "         --disease-name \"Hepatitis B\" \\"
echo "         --top-k 10"
echo ""
echo "  2. Or try vaccine query (original):"
echo ""
echo "     python showcase_ranker.py \\"
echo "         --checkpoint $OUTPUT_DIR/checkpoints/transductive_best.pt \\"
echo "         --vaccine-name \"Hepatitis B Vaccine\" \\"
echo "         --top-k 10"
echo ""
echo "  2. Check detailed results:"
echo "     cat $OUTPUT_DIR/results/transductive.json | python -m json.tool"
echo ""
echo "  3. Explore split manifests:"
echo "     cat $OUTPUT_DIR/splits/transductive/train.jsonl | head -5"
echo ""
