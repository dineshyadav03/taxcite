"""Streamlit web UI over the income-tax-rag pipeline."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from generate import GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL, answer_question  # noqa: E402

st.set_page_config(page_title="TaxCite", page_icon="\U0001f4dc", layout="centered")


@st.cache_resource(show_spinner="Building the search index (one-time setup)...")
def _ensure_index_built():
    # A fresh deployment (e.g. Streamlit Community Cloud) gets a clean clone
    # of this repo -- chroma_db/ is gitignored (derived data, not source), so
    # the index doesn't exist until something builds it. st.cache_resource
    # makes this run exactly once per server process, not once per visitor.
    from embed import build_index
    from retrieve import get_collection

    try:
        count = get_collection().count()
    except Exception:
        count = 0
    if count == 0:
        build_index()


_ensure_index_built()

if "question" not in st.session_state:
    st.session_state.question = ""

st.title("TaxCite")
st.caption(
    "Ask about Indian income tax law. Every answer is grounded in the actual statutory text "
    "of the Income-tax Act, 2025 or the superseded 1961 Act, cited by Act and section -- the "
    "system refuses to answer when the corpus doesn't support a confident response."
)

with st.sidebar:
    st.header("Corpus")
    st.write("**Income-tax Act, 2025** — 536 sections")
    st.write("**Income-tax Act, 1961** — 707 sections (superseded, kept for comparison)")
    st.divider()
    st.caption(
        "Embeddings via Voyage AI's hosted API (free tier). "
        + (
            f"Generation via Groq's free-tier hosted API (`{GROQ_MODEL}`)."
            if GROQ_API_KEY
            else f"Generation via a local Ollama model (`{OLLAMA_MODEL}`)."
        )
    )

EXAMPLE_QUESTIONS = [
    "What does section 139 of the Income-tax Act, 1961 say?",
    "What is the cash transaction limit under the Income-tax Act, 2025?",
    "Who is required to file an income tax return?",
]

st.write("Try an example, or ask your own question below:")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, example in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_input("Your question", key="question")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and generating..."):
        result = answer_question(question)

    if result["refused"]:
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])
        if result["chunks"]:
            st.subheader("Sources")
            seen = set()
            for c in result["chunks"]:
                meta = c["metadata"]
                key = (meta["act_id"], meta["section"])
                if key in seen:
                    continue
                seen.add(key)
                st.markdown(f"- **{meta['act_title']}, Section {meta['section']}**")
