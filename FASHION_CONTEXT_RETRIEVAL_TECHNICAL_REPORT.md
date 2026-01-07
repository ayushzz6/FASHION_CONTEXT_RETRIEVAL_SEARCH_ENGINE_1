

## 1. Architectural Strategy & Evaluation

To address the challenge of retrieving images based on dual constraints—**fashion items** (e.g., "red tie") and **environmental context** (e.g., "formal office")—we conducted a comparative analysis of three architectural paradigms.

### 1.1. Evaluated Approaches

1.  **Pure CLIP (Baseline)**
    *   **Mechanism:** utilizes a standard pre-trained CLIP model to encode the full image and query into a shared vector space, ranking by Cosine Similarity.
    *   **Trade-offs:**
        *   *Pros:* Minimal latency, high throughput, native zero-shot capability.
        *   *Cons:* Fails at **compositionality**. CLIP functions largely as a "Bag-of-Words" model, struggling to differentiate binding (e.g., "red shirt, blue pants" vs. "blue shirt, red pants") and often allowing background context to dominate foreground fashion details.

2.  **Fine-Tuned Fashion-CLIP**
    *   **Mechanism:** Transfer learning applied to CLIP using a domain-specific dataset (e.g., DeepFashion).
    *   **Trade-offs:**
        *   *Pros:* Superior vocabulary alignment for fashion terms (e.g., distinguishing "cardigan" vs. "blazer").
        *   *Cons:* High computational cost for training. Significant risk of **Catastrophic Forgetting**, where the model's generalized knowledge of contexts (e.g., "Street", "Cafe") degrades as fashion accuracy improves.

3.  **Hybrid Late-Fusion with Structural Reranking (Selected)**
    *   **Mechanism:** A two-stage pipeline combining a semantic "retrieval" phase (Recall) with a structured "reranking" phase (Precision).
    *   **Trade-offs:**
        *   *Pros:* Explicitly solves the compositionality problem. Decouples "vibe" from "attributes," allowing independent weighting of context and outfit. Interpretable scoring logic.
        *   *Cons:* Increased system complexity requiring orchestration of multiple models (CLIP, Object Detection, Clustering).

### 1.2. Decision

We selected **Approach 3 (Hybrid Late-Fusion)**. This architecture provides the necessary precision for complex multi-attribute queries while maintaining the zero-shot flexibility required for open-world retrieval.

---

## 2. System Architecture: The Hybrid Pipeline

The chosen architecture treats a single image as a composite data structure derived from three distinct signal streams.

### 2.1. Stream A: Semantic Stream (Global Recall)
*   **Model:** `ViT-B-32` (OpenCLIP).
*   **Storage:** **ChromaDB** (HNSW Vector Index).
*   **Function:** Encodes the global semantic "vibe" of the image.
*   **Role:** Responsible for the **ANN (Approximate Nearest Neighbor)** candidate generation phase, retrieving the top $N$ (e.g., 200) raw candidates.

### 2.2. Stream B: Attribute Stream (Hard Filtering)
This stream extracts structured metadata to enforce logical constraints.

1.  **Garment & Object Detection:**
    *   **Primary Backend:** **DETR (DEtection TRansformer)** (`facebook/detr-resnet-50`). Selected for its transformer-based attention mechanism which excels at identifying standard items like "tie", "handbag", and "suitcase".
    *   **Alternative Backend:** **YOLOv8** (`ultralytics`). Integrated as a configurable switch for scenarios prioritizing real-time inference speed.
2.  **Color Extraction:**
    *   **Pipeline:** **OpenCV GrabCut** (Foreground Segmentation) $
ightarrow$ **K-Means Clustering** ($k=5$).
    *   **Rationale:** GrabCut masks out the background, ensuring that the extracted color palette represents the *outfit*, not the environment.
3.  **Environment Classification:**
    *   **Method:** Zero-Shot CLIP Classification.
    *   **Taxonomy:** Bucketed into discrete contexts: *Office, Street, Park, Home, etc.*

### 2.3. Stream C: Compositional Stream (Spatial Grounding)
*   **Method:** Grid-based Patch Embedding.
*   **Process:** The image is divided into fixed-size patches. Each patch is encoded independently by CLIP.
*   **Function:** Enables **Phrase-Grounding**. Specific query phrases (e.g., "red tie") are scored against individual patches rather than the whole image, ensuring spatial correctness.

---

## 3. Query Execution Logic

The system executes a query (e.g., **"A red tie and white shirt in a formal office"**) through the following deterministic stages:

1.  **Query Parsing & Decomposition**
    *   **Entities:** Extracts `["tie", "shirt"]`.
    *   **Bindings:** Binds `red` $
ightarrow$ `tie` and `white` $
ightarrow$ `shirt`.
    *   **Context:** Extracts `["formal", "office"]`.

2.  **Candidate Retrieval (Recall)**
    *   Query ChromaDB with the full sentence embedding.
    *   Retrieve Top-200 candidates based on global semantic similarity.

3.  **Attribute Filtering (Precision)**
    *   **Check:** Does the metadata contain "tie" and "shirt"?
    *   **Action:** If confidence scores are below threshold (e.g., 0.25), penalize or discard the candidate.

4.  **Multi-Factor Reranking (Scoring)**
    *   **Compositional Score:** Compute similarity between "red tie" embedding and the image's patch embeddings. High score requires a specific region to match both "red" and "tie".
    *   **Context Score:** Compare extracted environment label against "office".
    *   **Final Rank:** Weighted sum of Global, Compositional, and Attribute scores.

---

## 4. Implementation & Code Structure

The codebase adheres to a strict separation of concerns between data ingestion and query serving.

*   **`indexer/` (Offline Processing)**
    *   `build_index.py`: Pipeline orchestrator.
    *   `models/`: Abstraction layers for ML models.
        *   `object_detector.py`: Unified interface for DETR and YOLO.
        *   `color_extractor.py`: Implements GrabCut + K-Means logic.
    *   `storage/`: Persistence layer handling ChromaDB (Vectors), SQLite (Metadata), and Disk (NPZ Patches).

*   **`retriever/` (Online Serving)**
    *   `query/parse_query.py`: Regex-based NLP parser for attribute extraction.
    *   `rerank/`: Implementation of scoring algorithms (Greedy assignment, weighted aggregation).
    *   `search.py`: Main entry point for the API/UI.

*   **Configuration**
    *   `config.yaml`: Centralized control plane for model weights, thresholds, and file paths.

---

## 5. System Characteristics

### 5.1. Scalability
*   **Vector Search:** Utilizes **HNSW** graphs for $O(\log N)$ retrieval complexity. Demonstrated readiness for **1 Million+** images with millisecond-level latency.
*   **Reranking:** Complexity is $O(K)$, where $K$ is the constant size of the candidate pool (e.g., 200). System latency remains stable regardless of total dataset size.
*   **Optimization Path:** For hyperscale, patch embeddings currently stored as `.npz` files should be migrated to a binary vector store to reduce I/O overhead.

### 5.2. Zero-Shot Capabilities
The system is fully **Zero-Shot**:
*   **No Training Required:** Relies exclusively on pre-trained Open-Vocabulary models.
*   **Adaptability:** instantly recognizes novel concepts (e.g., "Neon green scarf") without schema updates or fine-tuning.

---

## 6. Future Roadmap

### 6.1. Expansion: Geolocation & Weather
*   **Scene Recognition:** Integration of `Places365` or specialized CLIP prompts to detect landmarks (e.g., "Eiffel Tower") and scene typologies ("Beach", "Desert").
*   **Weather Tagging:** Implementation of a lightweight classification head for atmospheric conditions ("Sunny", "Rainy", "Snowy") to be stored as structured metadata.
*   **Logic:** Update `parse_query.py` to extract Named Entities (Cities) and Weather adjectives for dedicated scoring channels.

### 6.2. Optimization: Precision
*   **Segmentation (SAM):** Replace grid patches with **Segment Anything Model (SAM)** masks. This will isolate exact pixel regions for garments, preventing background color bleed into attribute scores.
*   **Cross-Encoder Reranking:** Deployment of a BERT-based Cross-Encoder to re-score the top-50 results. This captures nuanced linguistic relationships that Bi-Encoders (CLIP) miss.
*   **Relevance Feedback:** Implementation of a "More like this" loop, utilizing the attribute vector of a user-selected result to re-weight the subsequent query.
