# VaxKG

This project provides Python scripts to import and map the Vaccine Ontology (VO) into a Neo4j graph database. It leverages the Neosemantics (n10s) library for RDF import and performs subsequent mapping to align the ontology with domain-specific nodes and relationships.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
    * [.env File Configuration](#env-file-configuration)
    * [Neosemantics (n10s) Configuration](#neosemantics-n10s-configuration)
* [Usage](#usage)
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

The repository also contains utilities for building machine-learning ready
datasets and training a vaccine→adjuvant ranker:

1.  **Prepare the training snapshot.** This joins the curated VO/Vaxjo/Vaxvec
    metadata into the relational exports and emits the processed CSV/JSON
    artefacts used for modelling.

    ```bash
    python prepare_training_data.py --data-dir data --output-dir data/processed
    ```

2.  **Train the graph-based recommender.** The `train_ranker.py` script builds
    both leave-vaccine-out (transductive) and leave-disease-out (inductive)
    splits, constructs a PyTorch Geometric `HeteroData` graph with hashed text
    features, and optimises a PyG hetero encoder with a ListNet ranking loss
    plus an auxiliary link prediction head.

    ```bash
    python train_ranker.py \
        --data-path data/processed/training_samples.csv \
        --output-dir artifacts \
        --epochs 300 \
        --list-size 50
    ```

    The script writes split manifests to `artifacts/splits/<scheme>/train.jsonl`
    (and corresponding `val`/`test` files) and stores evaluation metrics under
    `artifacts/results/<scheme>.json`. Use `--skip-training` if you only need the
    manifests for downstream experiments.

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