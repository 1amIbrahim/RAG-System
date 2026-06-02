import json
import sys
import tempfile
from pathlib import Path

# Ensure project root is on the path when Streamlit launches from ui/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ingestion.loaders import load_document
from ingestion.chunker import SemanticChunker
from ingestion.metadata import save_chunks
from retrieval.embeddings import EmbeddingModel
from retrieval.vector import VectorIndex
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from reranking.cross_encoder import CrossEncoderReranker
from generation.llm import answer

st.set_page_config(page_title="RAG System", layout="wide")
st.title("RAG System")
st.caption("Upload documents, then ask questions grounded in their content.")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    embed_model = EmbeddingModel()
    reranker = CrossEncoderReranker()
    return embed_model, reranker


embed_model, reranker = load_models()


def _build_indices(chunks):
    vector_index = VectorIndex(embed_model)
    vector_index.build_index(chunks)
    bm25_index = BM25Index()
    bm25_index.build_index(chunks)
    return HybridRetriever(vector_index, bm25_index)


# ── Sidebar: document upload ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Ingest documents"):
        chunker = SemanticChunker()
        all_chunks = []
        progress = st.progress(0)

        for i, f in enumerate(uploaded):
            suffix = Path(f.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = Path(tmp.name)
            # File must be closed before PyMuPDF/docx can open it on Windows

            pages = load_document(tmp_path)
            chunks = chunker.chunk_pages(pages, source=f.name)
            all_chunks.extend(chunks)
            progress.progress((i + 1) / len(uploaded))

        out_path = Path("data/processed/uploaded_chunks.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_chunks(all_chunks, out_path)

        st.session_state["chunks"] = all_chunks
        st.session_state["retriever"] = _build_indices(all_chunks)
        st.success(f"Ingested {len(all_chunks)} chunks from {len(uploaded)} file(s)")

    # Also allow loading existing processed chunks
    if st.button("Load existing chunks"):
        path = Path("data/processed/sample_chunks.json")
        if path.exists():
            with open(path) as f:
                chunks = json.load(f)
            st.session_state["chunks"] = chunks
            st.session_state["retriever"] = _build_indices(chunks)
            st.success(f"Loaded {len(chunks)} existing chunks")
        else:
            st.error("No existing chunks found. Upload documents first.")

# ── Main: chat interface ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about your documents..."):
    if "retriever" not in st.session_state:
        st.warning("Upload and ingest documents first (or load existing chunks).")
    else:
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                retriever: HybridRetriever = st.session_state["retriever"]
                retrieved = retriever.search(query, top_k=10)
                retrieved_chunks = [c for c, _ in retrieved]

                reranked = reranker.rerank(query, retrieved_chunks, top_k=5)
                top_chunks = [c for c, score in reranked]

                result = answer(query, top_chunks)

            st.markdown(result["answer"])

            with st.expander("Sources"):
                for i, chunk in enumerate(top_chunks, 1):
                    st.markdown(
                        f"**[{i}]** `{chunk.get('source', 'unknown')}` "
                        f"— page {chunk.get('page', '?')}"
                    )
                    st.caption(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

            with st.expander("Debug: retrieval scores"):
                for chunk, score in reranked:
                    st.text(f"{score:.4f}  {chunk.get('source','?')} p.{chunk.get('page','?')}")

        st.session_state["messages"].append(
            {"role": "assistant", "content": result["answer"]}
        )
