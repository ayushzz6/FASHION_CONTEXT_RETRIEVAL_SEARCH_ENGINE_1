"""Streamlit UI for interactive retrieval."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

# Ensure project root is on sys.path for package imports when run via `streamlit run`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retriever.search import run_search


def main() -> None:
    st.set_page_config(page_title="Fashion Context Retrieval", layout="wide")
    st.title("Fashion & Context Retrieval")
    query = st.text_input("Query", value="A red tie and a white shirt in a formal setting.")
    k = st.slider("Results (k)", min_value=4, max_value=40, value=12, step=2)
    method = st.selectbox("Compositional matching", ["greedy", "hungarian"])

    if st.button("Search") and query:
        with st.spinner("Searching..."):
            results = run_search(query, k=k, method=method)
        st.success(f"Found {len(results)} results")
        cols = st.columns(4)
        for idx, hit in enumerate(results):
            col = cols[idx % len(cols)]
            with col:
                if hit.get("path") and Path(hit["path"]).exists():
                    st.image(hit["path"], caption=f"{hit['score']:.3f} | {hit.get('env_label', '')}", width="stretch")
                st.write(f"Score: {hit['score']:.3f}")
                st.write(f"Env: {hit.get('env_label', '')}")
                st.write(f"Colors: {hit.get('colors', {})}")
                st.write(f"Garments: {hit.get('garments', {})}")
                st.write(f"Objects: {hit.get('objects', {})}")


if __name__ == "__main__":
    main()
