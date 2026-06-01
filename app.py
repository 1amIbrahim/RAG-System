import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from retrieval.embeddings import EmbeddingModel
from retrieval.vector import VectorIndex
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from reranking.cross_encoder import CrossEncoderReranker
from generation.llm import answer

print("Loading models...")
embed_model = EmbeddingModel()
reranker_model = CrossEncoderReranker()
print("Models ready.")


def _build_retriever(chunks):
    v = VectorIndex(embed_model)
    v.build_index(chunks)
    b = BM25Index()
    b.build_index(chunks)
    return HybridRetriever(v, b)


def ingest(files, state):
    from ingestion.loaders import load_document
    from ingestion.chunker import SemanticChunker

    if not files:
        return state, "No files selected."
    chunker = SemanticChunker()
    all_chunks = []
    for f in files:
        pages = load_document(Path(f.name))
        chunks = chunker.chunk_pages(pages, source=Path(f.name).name)
        all_chunks.extend(chunks)
    state["retriever"] = _build_retriever(all_chunks)
    return state, f"Ingested {len(all_chunks)} chunks from {len(files)} file(s). Ready to chat."


def load_existing(state):
    path = Path("data/processed/sample_chunks.json")
    if not path.exists():
        return state, "No existing chunks found. Upload documents first."
    with open(path) as f:
        chunks = json.load(f)
    state["retriever"] = _build_retriever(chunks)
    return state, f"Loaded {len(chunks)} chunks. Ready to chat."


def chat(message, history, state):
    retriever = state.get("retriever")
    if retriever is None:
        return "Please upload documents or load existing chunks first."
    try:
        retrieved = retriever.search(message, top_k=10)
        retrieved_chunks = [c for c, _ in retrieved]
        reranked = reranker_model.rerank(message, retrieved_chunks, top_k=5)
        top_chunks = [c for c, _ in reranked]
        result = answer(message, top_chunks)
        sources = "\n\n**Sources:**\n" + "\n".join(
            f"[{i}] `{c.get('source', '?')}` p.{c.get('page', '?')}"
            for i, c in enumerate(top_chunks, 1)
        )
        return result["answer"] + sources
    except Exception as exc:
        return f"Error: {exc}"


with gr.Blocks(title="RAG System") as demo:
    gr.Markdown("# RAG System\nUpload documents and ask questions grounded in their content.")
    state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Documents")
            files = gr.File(
                label="Upload PDF / DOCX / TXT",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"],
            )
            ingest_btn = gr.Button("Ingest", variant="primary")
            load_btn = gr.Button("Load existing chunks")
            status = gr.Textbox(interactive=False, show_label=False, placeholder="Status...")
            ingest_btn.click(ingest, [files, state], [state, status])
            load_btn.click(load_existing, [state], [state, status])

        with gr.Column(scale=3):
            gr.ChatInterface(fn=chat, additional_inputs=[state], type="messages")

demo.launch()
