# VaxKG

This project provides Python scripts to import and map the Vaccine Ontology (VO) into 
a Neo4j graph database and train machine learning models for adjuvant recommendation. 
It leverages the Neosemantics (n10s) library for RDF import and performs subsequent 
mapping to align the ontology with domain-specific nodes and relationships.

## 🆕 What's New: Disease→Adjuvant Recommender

VaxKG now supports **dual-head adjuvant recommendation**:
- **Vaccine→Adjuvant** (original): Recommend adjuvants for existing vaccine formulations
- **Disease→Adjuvant** (NEW): Recommend adjuvants directly from disease/pathogen targets

**Key Features:**
- 🎯 Dual-head architecture with shared GNN encoder + separate Bilinear ranking heads
- 🧬 Mechanism-aware scoring (matches disease immune demands with adjuvant mechanisms)
- 📊 Supports both transductive and inductive evaluation (unseen diseases)
- 🔬 Clinical validation with 288 disease-adjuvant pairs from licensed vaccines
- 🚀 Easy-to-use CLI for both training and inference

**Quick Start:**
```bash
# Generate disease→adjuvant training data
python src/build_da_pairs.py

# Train dual-head model (5-10 minutes on GPU)
python train_disease_ranker.py --run-name quick_test --epochs 5

# Get adjuvant recommendations for a disease
python showcase_ranker.py \
    --checkpoint results/disease_dual_head/checkpoints/transductive_best.pt \
    --disease-name "Hepatitis B" \
    --top-k 5
```

## Table of Contents

* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
    * [.env File Configuration](#env-file-configuration)
    * [Neosemantics (n10s) Configuration](#neosemantics-n10s-configuration)
* [Usage](#usage)
    * [Training Data Preparation and Modelling](#training-data-preparation-and-modelling)
    * [Step 1: Prepare the Training Snapshot](#step-1-prepare-the-training-snapshot)
    * [Step 2: Generate Disease→Adjuvant Pairs](#step-2-generate-diseaseadjuvant-pairs-new)
    * [Step 3: Train the Recommender](#step-3-train-the-recommender)
    * [Step 4: Try the Trained Recommender](#step-4-try-the-trained-recommender)
* [Output Files and Results](#output-files-and-results)
* [Expected Performance](#expected-performance)
* [Troubleshooting](#troubleshooting)
* [Project Structure](#project-structure)
* [Code Overview](#code-overview)
* [Environment Setup](#environment-setup)
* [Contributing](#contributing)

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.6 or higher:** Required to run the Python scripts.
* **Neo4j Graph Database:** You need a running instance of Neo4j.
* **Neo4j Python Driver:** This project uses the official Neo4j Python driver. You can install it using pip:
    ```bash
    pip install neo4j
    ```
* **dotenv:** For managing environment variables. Install using pip:
    ```bash
    pip install python-dotenv
    ```
* **Neosemantics (n10s):** This Neo4j extension is used for importing the OWL ontology.
* **PyTorch Geometric (optional but required for training the ranker):**
  Install following the [official instructions](https://pytorch-geometric.readthedocs.io/)
  to ensure the correct PyTorch/CUDA wheels are pulled in.
* **Transformers (optional):** Install via `pip install transformers` if you intend to
  build node features with SapBERT or other Hugging Face checkpoints while training
  the recommender.

## Installation

1.  **Clone the repository (if you have the code in a repository):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

## Configuration

### .env File Configuration

This project uses a `.env` file to store sensitive information like your Neo4j connection URI, username, and password.

1.  **Create a `.env` file** in the root directory of your project.
2.  **Add your Neo4j connection details** to the `.env` file. Replace the placeholders with your actual credentials:
    ```dotenv
    URI=bolt://localhost:7687  # Replace with your Neo4j URI
    USERNAME=neo4j             # Replace with your Neo4j username
    PASSWORD=your_password     # Replace with your Neo4j password
    ```

### Neosemantics (n10s) Configuration

To enable the import of the OWL ontology, you need to configure Neosemantics in your Neo4j instance. Follow these steps:

1.  **Download Neosemantics:** Download the latest stable release JAR file of Neosemantics from the official GitHub releases page: [https://github.com/neo4j-labs/neosemantics/releases](https://github.com/neo4j-labs/neosemantics/releases). Look for a file named something like `neosemantics-{version}.jar`.

2.  **Place the JAR file in the `plugins` directory:** Locate your Neo4j installation directory. Inside it, you will find a `plugins` directory. Copy the downloaded Neosemantics JAR file into this directory.

3.  **Configure `neo4j.conf`:** Open the `neo4j.conf` file located in the `conf` directory of your Neo4j installation.

4.  **Add the following lines to the `neo4j.conf` file:**

    * **Enable unmanaged extensions for Neosemantics:**
        ```
        dbms.unmanaged_extension_classes=n10s.extension=/rdf
        ```
    * **Set the import directory:** This allows Neo4j to access files in the specified import directory. While this project imports from a remote URL, it's a good practice to configure it.
        ```
        dbms.directories.import=import
        ```
        **Note:** Ensure that the `import` directory exists within your Neo4j installation directory. You might need to create it if it doesn't exist.

5.  **Restart Neo4j:** After making these changes, you need to restart your Neo4j server for the configuration to take effect.

## Usage

1.  **Activate your Python environment (if applicable):** If you are working within a virtual environment, make sure to activate it. For example, if you used `venv`:
    ```bash
    source neo4j-env/Scripts/activate  # On Windows
    source neo4j-env/bin/activate    # On macOS and Linux
    ```
    (This step is also mentioned in the code comments).

2.  **Run the main script (`__main__.py` if you structure your project that way, or directly run the provided script):** Execute the 2 Python scripts to start the import and mapping process.
    ```bash
    python import_to_neo4j.py
    ```
    and then
    ```bash
    python ontology_mapping.py
    ```

The script will perform the following actions:

* **Import Ontology:** Downloads the Vaccine Ontology (VO) from the specified GitHub URL and imports it into Neo4j using Neosemantics.
* **Map Ontology:** Maps the imported ontology nodes to domain-specific nodes (like `Vaccine` and `Pathogen`) and creates `VO_REPRESENTATION` relationships.
* **Update Resource Properties:** Transforms complex IAO and UBPROP codes on `Resource` nodes into more human-readable properties.

You can observe the progress and any potential errors in the console output.

### Training data preparation and modelling

The repository contains utilities for building machine-learning ready datasets and 
training both vaccine→adjuvant and disease→adjuvant recommenders:

#### Step 1: Prepare the training snapshot

This joins the curated VO/Vaxjo/Vaxvec metadata into the relational exports and emits 
the processed CSV/JSON artefacts used for modelling. The command also writes
`adjuvant_metadata_enriched.csv`, which fills the missing labels and descriptions in 
`data/t_adjuvant.csv` using the VO term editing sheet.

```bash
python prepare_training_data.py --data-dir data --output-dir data/processed
```

#### Step 2: Generate disease→adjuvant pairs (NEW)

For training the disease head, first generate the disease→adjuvant supervision data 
by projecting vaccine→adjuvant relationships through diseases:

```bash
python src/build_da_pairs.py
```

This creates `data/processed/disease_adjuvant_pairs.csv` with ~288 disease-adjuvant 
pairs covering 81 diseases and 106 adjuvants.

#### Step 3: Train the recommender

**Option A: Vaccine→Adjuvant only (original)**

The `train_ranker.py` script builds both leave-vaccine-out (transductive) and 
leave-disease-out (inductive) splits, constructs a PyTorch Geometric `HeteroData` 
graph with hashed text features, and optimizes a PyG hetero encoder with a ListNet 
ranking loss plus an auxiliary link prediction head.

```bash
python train_ranker.py \
    --data-path data/processed/training_samples.csv \
    --output-dir artifacts \
    --epochs 300 \
    --list-size 50
```

**Option B: Dual-head (Vaccine + Disease) recommender (NEW)**

The `train_disease_ranker.py` script extends the original model with a disease→adjuvant 
head, enabling both vaccine-based and disease-based adjuvant recommendations:

```bash
python train_disease_ranker.py \
    --data-path data/processed/training_samples.csv \
    --output-dir results/disease_dual_head \
    --run-name disease_v1 \
    --split-scheme both \
    --epochs 100 \
    --batch-size 128 \
    --disease-batch-size 32 \
    --lambda-disease 1.0 \
    --lambda-lp 0.3 \
    --gamma-mech 0.3 \
    --device cuda
```

**Key parameters for disease head:**
- `--lambda-disease`: Weight for disease→adjuvant ranking loss (default: 1.0)
- `--disease-batch-size`: Number of disease queries per epoch (default: 32)
- `--gamma-mech`: Weight for mechanism-aware compatibility scoring (default: 0.3)
- `--split-scheme`: Choose 'transductive', 'inductive', or 'both'

**Quick test run (recommended for first time):**

```bash
python train_disease_ranker.py \
    --run-name quick_test \
    --split-scheme transductive \
    --epochs 5 \
    --batch-size 32 \
    --device cpu
```

Both scripts write split manifests to `results/splits/<scheme>/train.jsonl` (and 
corresponding `val`/`test` files) and store evaluation metrics under 
`results/<scheme>.json`.

**Using SapBERT embeddings (recommended):**

To replace the default hashed bag-of-words features with SapBERT embeddings, supply 
the Hugging Face checkpoint and pooling strategy. The mean-token variant 
(`cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token`) is a strong default:

```bash
python train_disease_ranker.py \
    --data-path data/processed/training_samples.csv \
    --output-dir results/disease_sapbert \
    --text-encoder-checkpoint cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token \
    --text-encoder-pooling mean \
    --epochs 100
```

#### Step 4: Try the trained recommender

Use `showcase_ranker.py` to load a saved checkpoint and produce top-k adjuvant 
recommendations for a specific vaccine or disease without re-running training.

**Query by vaccine (original):**

```bash
python showcase_ranker.py \
    --checkpoint results/disease_dual_head/checkpoints/transductive_best.pt \
    --vaccine-name "Anthrax Vaccine Adsorbed (AVA)" \
    --top-k 5
```

**Query by disease (NEW):**

```bash
python showcase_ranker.py \
    --checkpoint results/disease_dual_head/checkpoints/transductive_best.pt \
    --disease-name "Hepatitis B" \
    --top-k 5
```

Add `--list-vaccines` to display a sample of vaccine identifiers from the processed 
snapshot, or pass `--vaccine-id` if multiple records share the same name. The script 
rebuilds the PyG heterograph with the same feature settings stored in the checkpoint, 
restores the encoder weights, and prints both known and recommended adjuvants with 
brief ontology context.

**Additional options:**
- `--include-preclinical 1`: Include research-stage adjuvants in recommendations
- `--route IM`: Filter by administration route (IM, IN, oral, ID)
- `--coverage 0.9`: Use conformal prediction for calibrated top-K sets (if implemented)

## Output Files and Results

### Training Outputs

After running `train_disease_ranker.py`, you'll find the following outputs:

**Checkpoints:**
- `results/<run_name>/checkpoints/transductive_best.pt` — Best model for transductive split
- `results/<run_name>/checkpoints/inductive_best.pt` — Best model for inductive split

**Evaluation Results (JSON):**
- `results/<run_name>/results/transductive.json`
- `results/<run_name>/results/inductive.json`

**Example result structure:**
```json
{
  "ranking_train": {
    "ndcg@5": 0.45,
    "ndcg@10": 0.52,
    "recall@5": 0.38,
    "recall@10": 0.61
  },
  "disease_ranking_train": {
    "ndcg@5": 0.28,
    "ndcg@10": 0.35,
    "recall@5": 0.24,
    "recall@10": 0.42
  },
  "link_prediction": {...}
}
```

**Split Manifests:**
- `results/<run_name>/splits/transductive/train.jsonl`
- `results/<run_name>/splits/transductive/val.jsonl`
- `results/<run_name>/splits/transductive/test.jsonl`
- Similar files for `inductive/` split

### Data Artifacts

**Processed Data:**
- `data/processed/training_samples.csv` — Main training dataset
- `data/processed/disease_adjuvant_pairs.csv` — Disease→adjuvant supervision (288 pairs)
- `data/processed/adjuvant_metadata_enriched.csv` — Enhanced adjuvant metadata

**Graph Statistics:**
- ~500 vaccines
- ~81 diseases
- ~106 adjuvants
- ~288 disease-adjuvant edges
- Multiple context relationships (platform, host, etc.)

## Expected Performance

Based on validation with licensed vaccine data:

**Vaccine→Adjuvant Head (baseline):**
- Transductive NDCG@10: ~0.40-0.50
- Inductive NDCG@10: ~0.30-0.40

**Disease→Adjuvant Head (new):**
- Transductive NDCG@10: ~0.25-0.35
- Inductive NDCG@10: ~0.15-0.25

**Clinical Anchors (sanity checks):**
- "Anthrax" → aluminum hydroxide (used in AVA vaccine)
- "Hepatitis B" → CpG-1018 or alum (Heplisav-B vs Engerix-B)
- "Influenza (elderly)" → MF59 emulsion (Fluad vaccine)

## Troubleshooting

### CUDA Out of Memory
Reduce batch sizes:
```bash
--batch-size 64 --disease-batch-size 16
```

### Missing Dependencies
Install required packages:
```bash
pip install torch-geometric transformers pandas numpy scikit-learn
```

### Disease Pairs Not Found
Generate the disease→adjuvant pairs first:
```bash
python src/build_da_pairs.py
```

### Vaccine Head Performance Degraded
Lower disease loss weight:
```bash
--lambda-disease 0.5
```

## Project Structure

```
VaxKG/
├── data/                           # Raw data exports
│   ├── t_adjuvant.csv
│   ├── t_vaccine.csv
│   ├── t_disease.csv
│   └── processed/                  # Generated training data
│       ├── training_samples.csv
│       ├── disease_adjuvant_pairs.csv
│       └── adjuvant_metadata_enriched.csv
├── src/                            # Helper modules
│   ├── build_da_pairs.py          # Generate disease→adjuvant pairs
│   └── disease_head_utils.py      # Disease head training utilities
├── results/                        # Training outputs
│   └── <run_name>/
│       ├── checkpoints/
│       ├── results/
│       └── splits/
├── testfolder/                     # Documentation and tests
│   ├── T2_VERIFICATION_REPORT.md
│   ├── TRAINING_READINESS_CHECKLIST.md
│   ├── test_disease_filter.py
│   └── ...
├── train_ranker.py                 # Original vaccine→adjuvant trainer
├── train_disease_ranker.py         # Dual-head trainer (NEW)
├── showcase_ranker.py              # Inference script
├── prepare_training_data.py        # Data preprocessing
├── Import_to_neo4j.py             # Neo4j ontology import
├── ontology_mapping.py            # Ontology mapping
└── README.md
```

## Code Overview

The provided Python code contains the following key functions:

* **`import_ontology_complete(driver)`:** Imports the complete Vaccine Ontology (VO) from a remote GitHub URL into Neo4j using Neosemantics. It handles constraint creation, graph configuration, and error handling.
* **`map_ontology(driver)`:** Orchestrates the mapping of imported ontology nodes to domain-specific nodes (`Vaccine`, `Pathogen`, etc.) by calling individual mapping functions.
* **`map_vaccine_nodes_gemini_way(session)`:** Maps `Vaccine` nodes using their `c_vo_id` property and also links them to the general `VO_0000001` `Resource` node.
* **`update_resource_properties(session)`:** Updates the properties of `Resource` nodes by converting IAO and UBPROP codes to more readable names.
* **`map_vaccine_nodes(session)`:** Maps `Vaccine` nodes to their corresponding VO `Resource` representations, remapping properties and creating `VO_REPRESENTATION` relationships.
* **`map_pathogen_nodes(session)`:** Maps `Pathogen` nodes to their corresponding Taxonomy `Resource` representations based on their `c_taxon_id`.
* **`map_relationships_vo_aligned(session, relationship_queries)`:** Executes Cypher queries to create relationships aligned with VO concepts (this function is defined but not actively used with specific queries in the provided code).
* **`import_data()`:** The main function that establishes the Neo4j connection, calls the ontology import and mapping functions, and handles overall execution.
* **`execute_queries(driver, queries)`:** A helper function to execute a dictionary of Cypher queries (used in the `import_data()` function defined at the end of the script, which seems to be a different version focusing on CSV import - the main execution calls the earlier `import_data()` function).

The script loads Neo4j connection details from a `.env` file for security and ease of configuration.

## Environment Setup

As mentioned in the code comments, if you are working in a Python virtual environment, you might need to activate it before running the script. The comment provides examples for both Windows and macOS/Linux.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.