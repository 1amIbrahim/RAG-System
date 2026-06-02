import json
import sys
import tempfile
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retrieval.embeddings import EmbeddingModel
from retrieval.vector import VectorIndex
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from reranking.cross_encoder import CrossEncoderReranker
from generation.llm import answer as rag_answer

print("Loading models...")
embed_model = EmbeddingModel()
reranker_model = CrossEncoderReranker()
print("Models ready.")

_retriever = None


def _build_retriever(chunks):
    v = VectorIndex(embed_model)
    v.build_index(chunks)
    b = BM25Index()
    b.build_index(chunks)
    return HybridRetriever(v, b)


# Auto-load demo chunks baked into the image
_demo_path = Path("data/processed/sample_chunks.json")
if _demo_path.exists():
    try:
        with open(_demo_path) as f:
            _demo_chunks = json.load(f)
        _retriever = _build_retriever(_demo_chunks)
        print(f"Auto-loaded {len(_demo_chunks)} demo chunks from {_demo_path}.")
    except Exception as e:
        print(f"Auto-load failed: {e}")

app = FastAPI()


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/load")
def load_existing():
    global _retriever
    path = Path("data/processed/sample_chunks.json")
    if not path.exists():
        return JSONResponse({"error": f"File not found: {path.resolve()}"}, status_code=404)
    with open(path) as f:
        chunks = json.load(f)
    _retriever = _build_retriever(chunks)
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    global _retriever
    from ingestion.loaders import load_document
    from ingestion.chunker import SemanticChunker

    chunker = SemanticChunker()
    all_chunks = []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        pages = load_document(tmp_path)
        chunks = chunker.chunk_pages(pages, source=file.filename)
        all_chunks.extend(chunks)
    _retriever = _build_retriever(all_chunks)
    return {"status": "ok", "chunks": len(all_chunks)}


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_endpoint(req: QueryRequest):
    if _retriever is None:
        return JSONResponse(
            {"error": "No documents loaded. Click 'Load existing chunks' or upload files first."},
            status_code=400,
        )
    retrieved = _retriever.search(req.query, top_k=10)
    retrieved_chunks = [c for c, _ in retrieved]
    reranked = reranker_model.rerank(req.query, retrieved_chunks, top_k=5)
    top_chunks = [c for c, _ in reranked]
    result = rag_answer(req.query, top_chunks)
    return {
        "answer": result["answer"],
        "sources": [
            {
                "source": c.get("source", "?"),
                "page": c.get("page", "?"),
                "text": c["text"][:300],
            }
            for c in top_chunks
        ],
    }


# Serve React build — must be last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
