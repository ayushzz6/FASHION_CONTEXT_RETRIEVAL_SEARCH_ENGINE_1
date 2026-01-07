# Fashion Context Retrieval

This project is a  complete retrieval system for fashion images where the query mixes "what to wear" and "where it is worn." Like a  CLIP-based search with extra structure: you can ask for a red tie, a white shirt, and a formal office, and the system tries to satisfy all three constraints in one ranked list.

The pipeline is intentionally explicit. It separates global semantics (fast ANN retrieval) from local attributes (colors, garments, environment labels, objects), then re-ranks results using compositional matching between query phrases and image patches.

# What this repo contains

- An indexer that extracts embeddings and attributes from images.
- A retriever that parses language, pulls candidates from Chroma, filters by attributes, and re-ranks by composition.
- A Streamlit UI for interactive search.
- Configurable prompts, weights, and thresholds in a single YAML file.


# Indexing pipeline (how images become searchable)

The indexer builds three layers of information for each image:

1. Global embedding (semantic stream)
   - A CLIP model encodes the entire image into one vector.
   - Stored in ChromaDB for fast ANN search.

2. Patch embeddings (compositional stream)
   - Patch-level CLIP tokens are projected and saved.
   - Stored as compressed .npz files with patch coordinates.
   - Used later to match specific query phrases to specific regions.

3. Structured metadata (attribute stream)
   - Colors: KMeans on foreground pixels + CLIP color prompt scores.
   - Garments: CLIP prompt matching on global and patch embeddings.
   - Environment: CLIP prompt matching (office, street, park, home).
   - Objects: optional detector tags (DETR or YOLOv8).
   - Brightness: LAB luminance score.
   - Stored in SQLite for fast metadata lookup.

The CLI lives in indexer/build_index.py and writes artifacts to the paths defined in config.yaml.

## Retrieval pipeline (how queries become results)

Retrieval is handled in retriever/search.py and consists of four steps:

1. Query parsing
   - 'retriever/query/parse_query.py' extracts:
     - colors (red, white, navy, etc)
     - garments (shirt, tie, coat, etc)
     - items (tie, handbag, umbrella, etc)
     - environment hints (office, street, park, home)
     - style hints (formal/casual)
     - brightness hints (bright/dark)
   - If explicit color+garment phrases exist , they are used directly.
   - Otherwise the query is split on "and" and commas to create phrases.

2. ANN candidate retrieval
   - Encode the whole query with CLIP text encoder.
   - Query Chroma for the top-N nearest global embeddings.
   - Chroma returns cosine distances; the code converts to similarity with  1- distance.

3. Attribute filtering (precision boost)
   - Ensure minimum match ratios for colors and garments.
   - Optionally enforce object labels (handbag, suitcase, umbrella, etc).
   - Use stricter thresholds when explicit phrases are present.
   - Optional hard constraint on environment label.

4. Reranking (composition-aware)
   - Compositional score: assign each query phrase to a distinct patch embedding.
     - Greedy assignment by default.
     - Hungarian assignment if method="hungarian" is requested.
   - Color score: overlap between query colors and palette, adjusted by brightness hints.
   - Environment score: exact match + embedding similarity.
   - Garment score: overlap between requested garments and stored garment tags.
   - Final score is a weighted sum configured in config.yaml.

Score formula (from retriever/rerank/scoring.py):

```
score = w0 * global_sim
      + w1 * compositional
      + w2 * color
      + w3 * env
      + w4 * garments
```

# Quick start

1. Install dependencies


pip install -r requirements.txt


2. Update config.yaml if needed

The config uses absolute paths. If you move the repo, update:

- storage.chroma.persist_directory
- storage.patches_dir
- storage.metadata_db
- indexer.default_images_path

3. Build the index


python indexer/build_index.py --config.yaml


4. Run the UI


streamlit run streamlit_app.py


# Example queries

Try these in the UI:

- "A red tie and a white shirt in a formal setting"
- "Blue jeans and a hoodie on a city street"
- "A yellow dress in a green park"
- "Black coat in a rainy scene"

The UI displays each image with a total score and the metadata used for reranking.

# Configuration guide

Key sections in config.yam`:

- model
  - clip_name, clip_pretrained`: ViT-B-32 model and laion2B-s34B-b79K.
  - device: cuda 
  - patch_name, patch_pretrained : ViT-B-32 and laion2b_s34b_b79k.

- storage
  - chroma.persist_directory: persistent Chroma index path.
  - patches_dir: patch embedding storage directory.
  - metadata_db: SQLite database for attributes.

- indexer
  - batch_size: number of images per Chroma write.
  - top_colors: how many color clusters to keep.
  - clip_color_top_n: top color prompts to keep from CLIP.
  - color_palette_weight: blend factor between KMeans palette and CLIP colors.
  - garment_top_n, garment_min_score, garment_patch_weight: garment prompt settings.
  - detector`: object detector backend, labels, and thresholds.

- retriever
  - ann_top_n: number of ANN candidates to fetch.
  - return_top_k: final results returned after rerank.
  - weights: scoring weights for global/compositional/color/env/garments.
  - attribute_filters: thresholds and strictness rules.
  - env_prompts: environment prompt list used by the classifier.

# Storage artifacts

After indexing you should see:

- FASHION_CONTEXT_DATA/chroma_index/
  - chroma.sqlite3 plus HNSW vector data files.
- FASHION_CONTEXT_DATA/patches/
  - One <image_id>.npz per image with patche and coords.
- FASHION_CONTEXT_DATA/metadata.sqlite
  - images, colors, garments, and objects tables.

These outputs are required for retrieval and UI rendering.

# Using the retriever in code

The Streamlit app calls run_search directly, so you can do the same:


from retriever.search import run_search

results = run_search("A red tie in a formal office", k=6, method="greedy")
for hit in results:
    print(hit["path"], hit["score"], hit["env_label"])

