## **TL;DR: *We’ll turn VO/VIOLIN/Vaxjo \+ VAC into a small, explainable adjuvant recommender by November.***

## **Summary** 

We will build a small, explainable system that suggests appropriate **adjuvant classes** (e.g., emulsions, saponins, CpG) for a given disease context, using the resources your lab already maintains. Concretely, we will connect what **VO** and **VIOLIN** know about diseases, vaccines, and platforms with **Vaxjo’s** curated adjuvant entries, and enrich those records with mechanism tags from the **NIAID Vaccine Adjuvant Compendium (VAC)**. On top of this map, a compact learning module will return a short, ranked list of adjuvant options and a plain-English “why,” grounded in known usage and mechanisms. The scope is intentionally small so we can finish by late November and submit a concise paper immediately afterward.

### The objective of this project

Build a small, reliable system that recommends/predicts vaccine **adjuvant classes** for a given vaccine context (disease/pathogen ± platform ± optional antigen).

We do this on top of **VaxKG**, a Neo4j knowledge graph that integrates **VIOLIN** vaccine records with the **Vaccine Ontology (VO)**, with sufficient scale: \~28k VIOLIN nodes, \~101k VO nodes, \~412k relations. This project/idea could also be found in the future work section of **VaxKG** 

### The training data

Since we want to build our system based on VaxKG, we’ll use the VaxKG graph as our data source:

* **Nodes:** Vaccine, Disease/Pathogen, Platform, Adjuvant (from VIOLIN/Vaxjo), plus VO concept nodes used for standardization.  
    
* **Typed edges:** e.g., Vaccine→Disease (**FOR\_DISEASE**), Vaccine→Platform (**USES\_PLATFORM**), Vaccine→Adjuvant (**CONTAINS\_ADJUVANT**) curated in Vaxjo; VaxKG stores these as relationships for learning.  
    
* **Text features:** labels/definitions from **VO** (for diseases, adjuvants, platforms) become biomedical text embeddings for node features. 

All the data above can be found/used directly in VaxKG.

### Why should we use the Graph Neural Network, and what is it?

Now that the goal and the data have been determined, it’s time to design how we should train a model that fits the characteristics of the goal and the data. In this case, the data are structured in a graph with nodes and edges. Hence, in order to learn these patterns, a Graph Neural Network (GNN) might be a good fit. It learns from nodes and typed edges directly, which is exactly how VaxKG stores vaccine knowledge, so it can capture multi-hop patterns like Disease → Vaccine → Adjuvant and generalize using node features (text, types).

### The training pipeline and the training loss.

#### What is the ground-truth label?

We treat each curated   
Vaccine → CONTAINS\_ADJUVANT → Adjuvant   
edge in VaxKG (sourced from VIOLIN/Vaxjo) as supervision. In plain terms, for a vaccine record with its disease (and optionally platform), the adjuvant actually used in Vaxjo is the label. Vaxjo explicitly curates “vaccines that use this adjuvant,” which is why these edges are reliable positives.

#### How we build training examples (from VaxKG).

1. From the graph, collect tuples: *(vaccine, disease, optional platform, used adjuvant)*. VaxKG already integrates **12 core VIOLIN tables with VO in Neo4j**; we simply export the needed node/edge slices. 

2. Convert each vaccine row into a **query** (disease ± platform) and a **candidate set** of adjuvants (or adjuvant classes). The **ground truth** is the candidate that matches the curated **CONTAINS\_ADJUVANT** edge for that vaccine. 

3. Create **node features** from VO/Vaxjo text (labels/definitions) with a frozen biomedical encoder; feed the heterogeneous graph to a small R-GCN/HGT encoder (built for multi-relation graphs).

#### Two complementary training objectives.

* **A) Recommendation head (primary): listwise ranking over candidates.**  
   For each query, the model scores all candidate adjuvants (or **adjuvant classes**) and we optimize a **listwise loss** so the true one ranks highest. Two standard choices are **ListNet** (softmax cross-entropy over the list) and **ListMLE** (likelihood of the ground-truth permutation); both are widely used for learning-to-rank.

* **B) Link-prediction head (auxiliary/baseline):**  
   View supervision as missing-edge completion on the relation **Vaccine —CONTAINS\_ADJUVANT→ Adjuvant**. Train with a hetero-GNN encoder and a decoder such as **RotatE** (handles asymmetric relations) using **binary cross-entropy** on positive edges vs. type-respecting negatives. This improves structural reasoning and gives a strong baseline (MRR/Hits@K). 

#### Training flow (simple).

1. **Mini-batch:** sample a batch of vaccines; fetch their 1–2-hop subgraphs; build the query vector (disease ± platform).

2. **Encode:** run a lightweight **R-GCN/HGT** to produce node embeddings. Optionally add **APPNP** (“predict-then-propagate”) to diffuse predictions a few hops with a Personalized-PageRank kernel.

3. **Score & loss:**

   * For the **listwise head**, apply ListNet/ListMLE over the candidate list; the **ground truth** is the used adjuvant (or its class) for that vaccine.

   * For the **link-prediction head**, score (vaccine, adjuvant) pairs with RotatE (or similar) and optimize BCE with negative sampling.   
4. **Evaluate splits:** hold out some **vaccines** (transductive) and, separately, hold out entire **diseases** (inductive) to test generalization to unseen conditions. VaxKG’s integrated VIOLIN+VO structure makes both splits straightforward.

#### A typical training sample

So a typical training sample would be (maybe not accurate):

**Hepatitis B (HEPLISAV-B → CpG-1018)**

**Query (what we condition on)**

* Disease: **Hepatitis B**

* Platform: **Protein subunit (HBsAg recombinant)**  
   (These come from VaxKG’s typed edges `Vaccine → FOR_DISEASE → Disease` and `Vaccine → USES_PLATFORM → Platform`, since VaxKG integrates VIOLIN tables with VO in Neo4j.)

**Candidate set (adjuvant classes)**

* `TLR9/CpG-1018`, `Alum`, `Emulsion (MF59/AS03)`, `Saponin (AS01/Matrix-M)`, `STING/CDNs`, `None/Unknown`.

**Ground-truth label (from Vaxjo usage)**

* **CpG-1018 (TLR9)** — HEPLISAV-B is an HBsAg vaccine adjuvanted with **CpG-1018**, a TLR9 agonist. (This curated “**vaccines that use this adjuvant**” linkage is exactly what Vaxjo stores and what VaxKG exposes as `Vaccine → CONTAINS_ADJUVANT → Adjuvant`.)

**Node features used by the encoder (frozen text embeddings)**

* VO/Vaxjo text for: “Hepatitis B”, “protein subunit vaccine”, “CpG-1018”; VO terms are openly licensed (CC-BY-4.0) and designed for standardization, which is why we can embed labels/definitions directly.

**Positive/negative pairs (for the link-prediction head)**

* Positive: `(HEPLISAV-B, CpG-1018)`

* Negatives (type-respecting): `(HEPLISAV-B, MF59)`, `(HEPLISAV-B, AS01)`, `(HEPLISAV-B, Alum)`, … (MF59 is typically used in flu vaccines like Fluad, not HepB.)

## Below are obsolete:

Why this, why now?

* VO and VIOLIN already standardize vaccines and relationships; Vaxjo curates adjuvants and their real-world usage. These are mature, He-lab assets we can stand on. [hegroup.org+2PubMed+2](https://hegroup.org/?utm_source=chatgpt.com)

* The NIAID/DAIT **VAC** adds up-to-date mechanism tags (e.g., TLR9, STING, emulsions, saponins) that make recommendations explainable. [niaid.nih.gov+2niaid.nih.gov+2](https://www.niaid.nih.gov/research/vaccine-adjuvant-compendium-vac?utm_source=chatgpt.com)

* Recent work shows VO↔VIOLIN can be integrated as a **knowledge graph (Neo4j)** (VaxKG), which is exactly the backbone we need. [PubMed Central+1](https://pmc.ncbi.nlm.nih.gov/articles/PMC12338677/?utm_source=chatgpt.com)

* For scientific grounding, we’ll follow the Pulendran/O’Hagan **Nature Reviews Drug Discovery** overview of modern adjuvant mechanisms, so our “why this adjuvant?” messages match current immunology. [Nature](https://www.nature.com/articles/s41573-021-00163-y?utm_source=chatgpt.com)

## What we’ll build.

1. **A graph of what’s known:** diseases, vaccines, platforms, adjuvant classes, and mechanisms—drawn from VO/VIOLIN/Vaxjo and tagged with VAC mechanisms. [PubMed Central+1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3312338/?utm_source=chatgpt.com)

2. **A small ML ranker:** given a disease (and optional platform/antigen), it suggests the **top-k adjuvant classes**, and–critically–shows **clear explanations**: “we recommend CpG-1018 because HBV vaccines use it; it targets TLR9 (Th1 bias).” (No deep jargon; it’s essentially “learn from the graph’s patterns.”) [niaid.nih.gov+1](https://www.niaid.nih.gov/research/vaccine-adjuvant-compendium-vac?utm_source=chatgpt.com)

3. **A minimal web demo:** a single form → ranked list → a “Why” panel with VO/VAC paths and mechanism tags.

## What a user sees.

* Input: “Disease=HBV; Platform=subunit; (optional) HBsAg sequence.”

* Output: 3–5 adjuvant classes with short, referenced rationales and confidence (we’ll calibrate probabilities so the model “knows when it doesn’t know”). [Nature](https://www.nature.com/articles/s41573-021-00163-y?utm_source=chatgpt.com)

## How we’ll evaluate (simple and fair).

* **Hold-out check:** hide the adjuvant from some known vaccines; ask the system to recover it from disease (+/- platform). Report top-k accuracy. (Data from VO/VIOLIN/Vaxjo.) [PubMed+1](https://pubmed.ncbi.nlm.nih.gov/18025042/?utm_source=chatgpt.com)

* **Case studies:** influenza↔emulsions (MF59/AS03), zoster↔AS01, HBV↔CpG-1018—each shown with a short mechanism note and citations from VAC/NRDD. [niaid.nih.gov+1](https://www.niaid.nih.gov/research/vaccine-adjuvant-compendium-overview?utm_source=chatgpt.com)

## Timeline (aim: draft by late Nov 2025).

* **Week 1 (Sep):** assemble the graph from VO/VIOLIN/Vaxjo \+ VAC; define fields and mappings. [hegroup.org+2PubMed Central+2](https://hegroup.org/?utm_source=chatgpt.com)

* **Weeks 2–3:** train a small ranker; build “Why” explanations from ontology paths and VAC tags. [niaid.nih.gov](https://www.niaid.nih.gov/research/vaccine-adjuvant-compendium-overview?utm_source=chatgpt.com)

* **Week 4:** evaluation \+ 2–3 case studies.

* **Weeks 5–6:** write-up and a very small demo.

## Next Step.

1. If this looks good, I’ll start with the data mapping, and I can prepare a slide deck so I can present to you next week about our progress.  
2. Graph Neural Network (HGT  
3. Training data / what we really need. (work based on VaxKG database)  
4. DDL October 15th

