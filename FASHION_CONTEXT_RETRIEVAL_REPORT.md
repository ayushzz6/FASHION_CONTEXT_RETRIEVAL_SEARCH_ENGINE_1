

## 1. Approaches Considered

In designing a system to retrieve images based on both **fashion items** ("red tie") and **environmental context** ("formal office"), we evaluated three distinct architectural strategies.

### A. Pure CLIP (Baseline)
The simplest approach uses a standard pre-trained CLIP model to encode the entire image and the user's text into a shared vector space, retrieving results via Cosine Similarity.
*   **Pros:** Extremely fast implementation, low latency, native zero-shot capabilities.
*   **Cons:** Suffer's from "Bag-of-Words" limitations. CLIP struggles to distinguish *compositionality* (e.g., "Person in red shirt and blue pants" vs. "Person in blue shirt and red pants"). It often conflates the background style with the foreground object.

### B. Fine-Tuned Fashion-CLIP
Fine-tuning a CLIP model on a specialized dataset (e.g., DeepFashion) to align visual features better with fashion terminology.
*   **Pros:** Improved domain understanding (e.g., distinguishing "cardigan" from "blazer").
*   **Cons:** Computationally expensive to train. Requires massive labeled datasets. High risk of **Catastrophic Forgetting**, where the model becomes great at fashion but loses its ability to understand general contexts (like "Eiffel Tower" or "Coffee Shop").

### C. Hybrid Late-Fusion with Structural Reranking (Chosen Approach)
A multi-stage pipeline that separates the problem into "Semantic Retrieval" (Global Vibe) and "Structured Reranking" (Specific Details).
*   **Pros:** Solves the compositionality problem explicitly. Allows fine-grained control over the importance of context vs. outfit. Modular and explainable.
*   **Cons:** Higher complexity (requires multiple models: Object Detection, Segmentation/Clustering, CLIP). Slightly higher query latency than Pure CLIP.

---

## 2. Chosen Architecture

We implemented **Approach C (Hybrid Late-Fusion)**.

**Architecture Summary:**
I have designed a comprehensive Hybrid Retrieval Architecture. This solution addresses the compositionality challenge (e.g., distinguishing "red shirt" from "red pants") by combining FashionCLIP embeddings for semantic "vibe" matching with Structured Metadata Filtering for attribute precision.

This architecture treats a fashion image not as a single vector, but as a composite of three distinct signal streams.

### The Three Streams
1.  **Semantic Stream (Global):**
    *   Uses `ViT-B-32` to encode the global image embedding.
    *   Stored in **ChromaDB** for fast Approximate Nearest Neighbor (ANN) search.
    *   *Purpose:* Fast candidate generation (Recall).

2.  **Attribute Stream (Local Metadata):**
    *   **Garments & Objects:** Identified using **Object Detection**. The system is modular and supports:
        *   **DETR (DEtection TRansformer):** The default backend (`facebook/detr-resnet-50`) which is excellent for identifying standard items like "tie", "handbag", "suitcase", and "umbrella".
        *   **YOLO (You Only Look Once):** The system explicitly supports `ultralytics` YOLO models (e.g., YOLOv8) for faster real-time inference if switched in the configuration.
    *   **Colors:** Extracted via **K-Means Clustering** (`sklearn`) on pixels. Crucially, we use **OpenCV GrabCut** to generate a foreground mask, ensuring we analyze the outfit's colors and ignore the background.
    *   **Environment:** Classified into buckets (Office, Street, Park, Home, etc.) using specific text prompts via a zero-shot classifier.
    *   *Purpose:* Hard filtering and precision.

3.  **Compositional Stream (Patches):**
    *   The image is split into a grid of patches, each encoded independently.
    *   *Purpose:* Grounding. We map specific query phrases (e.g., "red tie") to specific image regions to ensure the color and object match spatially.

### How it Handles Fashion Queries
When a user searches for **"A red tie and white shirt in a formal office"**:
1.  **Parsing:** The query is deconstructed into:
    *   *Garments:* "tie", "shirt"
    *   *Colors:* "red" (bound to tie), "white" (bound to shirt)
    *   *Context:* "formal", "office"
2.  **Candidate Retrieval:** ChromaDB finds the top 200 images that semantically match the full sentence.
3.  **Filtering:** Images missing a "tie" or "shirt" tag (detected by DETR/YOLO) are penalized or removed.
4.  **Scoring:**
    *   **Compositional:** The phrase "red tie" is compared against image patches. If the "tie" region is red, score goes up. If the "shirt" region is red, score goes down.
    *   **Context:** The image's environment label is checked against "office".
5.  **Result:** The user gets an image where the *tie* is specifically red, not just an image containing the color red somewhere.

---

## 3. Modular Code Structure

The codebase is strictly separated into `Indexer` (Data Ingestion) and `Retriever` (Query Processing) logic, ensuring maintainability.

*   **`indexer/`**: Handles heavy-lifting offline.
    *   `build_index.py`: Orchestrates the pipeline.
    *   `models/`: Wrappers for **CLIP**, **DETR/YOLO**, and **K-Means/GrabCut**.
    *   `storage/`: Manages ChromaDB (vectors), SQLite (metadata), and Disk (patches).
*   **`retriever/`**: Lightweight query-time logic.
    *   `query/parse_query.py`: Rule-based NLP to extract fashion attributes.
    *   `rerank/`: Contains the scoring formulas (Greedy assignment, weighted sums).
    *   `search.py`: The core entry point for applications.
*   **`config.yaml`**: Centralized configuration for weights, thresholds, and paths.

---

## 4. Scalability & Zero-Shot Capabilities

### Scalability (1 Million Images)
*   **Vector Search:** ChromaDB uses HNSW (Hierarchical Navigable Small World) graphs, which scale logarithmically (`O(log N)`). Searching 1M images takes milliseconds.
*   **Metadata Filtering:** SQLite handles 1M rows efficiently for attribute lookup.
*   **Reranking Bottleneck:** The detailed patch matching is `O(K)` where `K` is the number of *candidates* (e.g., 200), not the total dataset size. Therefore, the system remains fast regardless of dataset size, provided the initial ANN candidate pool (`ann_top_n`) remains constant.
*   *Optimization for Scale:* To scale further, patch embeddings—currently stored as `.npz` files—should be moved to a specialized vector store or binary format to reduce I/O latency during the reranking phase.

### Zero-Shot Capability
The system is entirely **Zero-Shot**.
*   It was not trained on the specific images in the dataset.
*   It handles new descriptions (e.g., "Neon green scarf") immediately because it relies on CLIP's open-vocabulary language model and generic object detectors. It does not require a fixed list of classes.

---

## 5. Future Work

### A. Extending for Locations (Cities, Places) & Weather
To support queries like *"Rainy day in Paris"* or *"Summer at the beach"*:
1.  **Scene Recognition:** Integrate a `Places365` model or use a specialized CLIP head with prompts for specific landmarks (Eiffel Tower, Times Square) and scene types (Beach, Mountain, Desert).
2.  **Weather Classification:** Add a lightweight classifier (or CLIP prompts: "sunny", "rainy", "snowy", "foggy") during indexing. Store this as a `weather_tag` in SQLite.
3.  **Query Parsing:** Update `parse_query.py` to recognize city names (via Named Entity Recognition) and weather adjectives.
4.  **Logic:** Add a `weather_match` component to the scoring formula in `config.yaml`.

### B. Improving Precision
1.  **Segmentation (SAM):** Replace rectangular patches with **Segment Anything Model (SAM)** masks. This allows exact isolation of a "shirt" pixels, ensuring background colors do not bleed into the garment's color score.
2.  **Cross-Encoder Reranking:** Train a small BERT-based Cross-Encoder to take the top 50 results and output a relevance score. Cross-encoders are slower but much better at understanding complex linguistic nuances than bi-encoders (CLIP).
3.  **Feedback Loop:** Implement Relevance Feedback. If a user clicks result #5, use its attributes to re-weight the search vector for the next query.