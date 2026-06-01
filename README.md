---
title: RAG System
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# RAG System

A Retrieval-Augmented Generation (RAG) system for querying document collections.
Upload PDFs, DOCX, or TXT files and ask questions — answers are grounded in your
documents with inline citations.

---

## Pipeline Overview

```
                        ┌─────────────────────────────────┐
                        │          USER QUERY             │
                        └────────────────┬────────────────┘
                                         │
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        INGESTION  (one-time)                        │
 │                                                                     │
 │  PDF / DOCX / TXT                                                   │
 │       │                                                             │
 │       ▼                                                             │
 │  [ loaders.py ]  ──── PyMuPDF (PDF), python-docx (DOCX)            │
 │       │               Returns list of {page, text} dicts            │
 │       ▼                                                             │
 │  [ chunker.py ]  ──── Paragraph-based, token-aware splitting        │
 │       │               max 500 tokens / chunk, 100-token overlap     │
 │       │               Token counting via tiktoken (cl100k_base)     │
 │       ▼                                                             │
 │  [ metadata.py ] ──── Saves chunks to data/processed/*.json        │
 │                       Each chunk: {chunk_id, source, page,          │
 │                                    text, tokens}                    │
 └─────────────────────────────────────────────────────────────────────┘
                                         │
                              chunks loaded into memory
                                         │
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        INDEXING  (on startup)                       │
 │                                                                     │
 │        chunks                                                       │
 │           ├──────────────────────────────────────────┐             │
 │           ▼                                          ▼             │
 │  [ embeddings.py ]                         [ bm25.py ]             │
 │  HuggingFace SentenceTransformer           rank-bm25 (BM25Okapi)   │
 │  Model: BAAI/bge-small-en-v1.5             Tokenization: regex \w+ │
 │  Dim: 384, L2-normalized                   IDF shift for stability  │
 │           │                                          │             │
 │           ▼                                          │             │
 │  [ vector.py ]                                       │             │
 │  FAISS IndexFlatL2                                   │             │
 │  L2 dist → cosine similarity                         │             │
 │  similarity = 1 - (dist / 2)                         │             │
 │           │                                          │             │
 │           └──────────────┬───────────────────────────┘             │
 │                          ▼                                         │
 │                  [ hybrid.py ]                                      │
 │                  Score fusion                                       │
 │                  vector × 0.7 + BM25 × 0.3                         │
 └─────────────────────────────────────────────────────────────────────┘
                                         │
                              top-N chunks retrieved
                                         │
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        RERANKING                                    │
 │                                                                     │
 │  [ cross_encoder.py ]                                               │
 │  Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (HuggingFace)        │
 │  Scores each (query, chunk) pair jointly                            │
 │  Much more accurate than bi-encoder similarity alone               │
 │  Returns top-K chunks sorted by relevance score                    │
 └─────────────────────────────────────────────────────────────────────┘
                                         │
                              top-K reranked chunks
                                         │
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        GENERATION                                   │
 │                                                                     │
 │  [ prompts.py ]                                                     │
 │  Builds prompt with numbered context chunks + citation instruction  │
 │  "Answer using ONLY the context. Cite as [1], [2]..."              │
 │           │                                                         │
 │           ▼                                                         │
 │  [ llm.py ]                                                         │
 │  Primary:  Ollama  (local server, http://localhost:11434)           │
 │            Default model: mistral  (also works: llama3, phi3)      │
 │  Fallback: HuggingFace transformers pipeline (GPT-2, demo only)    │
 │                                                                     │
 │  Returns: { answer: str, sources: [{source, page}] }               │
 └─────────────────────────────────────────────────────────────────────┘
                                         │
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        STREAMLIT UI  (ui/app.py)                    │
 │                                                                     │
 │  Sidebar: upload docs → ingest → index (or load existing)          │
 │  Main:    chat input → retrieve → rerank → generate → display      │
 │           Sources expander  (cited chunks)                          │
 │           Debug expander    (retrieval scores)                      │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## Models & Libraries

### Embedding model
| Property | Value |
|----------|-------|
| Model | `BAAI/bge-small-en-v1.5` |
| Source | HuggingFace Hub (auto-downloaded on first run) |
| Library | `sentence-transformers` |
| Dimensions | 384 |
| Notes | Fast, strong English retrieval model. Embeddings are L2-normalised so cosine similarity = dot product. |

### Vector index
| Property | Value |
|----------|-------|
| Library | `faiss-cpu` |
| Index type | `IndexFlatL2` (exact search) |
| Similarity | `1 - (L2_distance / 2)` — equivalent to cosine similarity for normalised vectors |

### Keyword index
| Property | Value |
|----------|-------|
| Library | `rank-bm25` |
| Algorithm | BM25Okapi |
| Tokenizer | `re.findall(r'\w+', text.lower())` |
| Notes | IDF is shifted to ≥ 0 when corpus is very small (< ~5 docs) |

### Reranker
| Property | Value |
|----------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Source | HuggingFace Hub (auto-downloaded on first run) |
| Library | `sentence-transformers` (CrossEncoder) |
| Notes | Scores (query, chunk) pairs jointly — much more accurate than embedding similarity alone. Adds ~200ms per query for top-10 candidates. |

### LLM (generation)
| Property | Value |
|----------|-------|
| Primary | Ollama — local inference server |
| Default model | `mistral` (also works: `llama3`, `phi3`, any Ollama model) |
| Fallback | HuggingFace `transformers` GPT-2 pipeline (demo quality only) |
| Notes | Ollama must be running (`ollama serve`) with the model pulled (`ollama pull mistral`). |

### Document loading
| Format | Library |
|--------|---------|
| PDF | `PyMuPDF` (`fitz`) — page-aware extraction |
| DOCX | `python-docx` |
| TXT | Python built-in |

### Token counting
| Property | Value |
|----------|-------|
| Library | `tiktoken` |
| Encoding | `cl100k_base` (same as GPT-4 / text-embedding-ada-002) |

---

## Project Structure

```
rag_system/
├── ingestion/
│   ├── loaders.py        PDF / DOCX / TXT → page dicts
│   ├── chunker.py        Paragraph-based semantic chunker
│   ├── tokenizer.py      tiktoken token counter
│   └── metadata.py       Save chunks to JSON
├── retrieval/
│   ├── embeddings.py     SentenceTransformer wrapper
│   ├── vector.py         FAISS index (build, search, save, load)
│   ├── bm25.py           BM25 index
│   └── hybrid.py         Weighted score fusion
├── reranking/
│   └── cross_encoder.py  CrossEncoder reranker
├── generation/
│   ├── prompts.py        Prompt builder with citation formatting
│   └── llm.py            Ollama / HF generation + answer()
├── ui/
│   └── app.py            Streamlit chat interface
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── data/
│   ├── raw/              Source documents
│   └── processed/        Chunked JSON files
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies (inside your venv)

```powershell
pip install -r requirements.txt
```

### 2. (Optional) Set up Ollama for best generation quality

```powershell
# Install Ollama from https://ollama.ai, then:
ollama pull mistral
ollama serve
```

Without Ollama the system falls back to GPT-2 locally — answers will be poor quality.

### 3. Run tests

```powershell
python -m tests.test_chunking
python -m tests.test_retrieval
python -m tests.test_generation    # downloads ~100 MB models on first run
```

### 4. Launch the UI

```powershell
python -m streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

---

## Configuration

| Setting | Where | Default |
|---------|-------|---------|
| Embedding model | `retrieval/embeddings.py` `EmbeddingModel.__init__` | `BAAI/bge-small-en-v1.5` |
| Reranker model | `reranking/cross_encoder.py` `CrossEncoderReranker.__init__` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Ollama model | `generation/llm.py` `_DEFAULT_MODEL` | `mistral` |
| Hybrid weights | `retrieval/hybrid.py` `HybridRetriever.__init__` | vector=0.7, bm25=0.3 |
| Chunk size | `ingestion/chunker.py` `SemanticChunker.__init__` | max_tokens=500, overlap=100 |
| Retrieval top-N | `ui/app.py` | top_k=10 before rerank, top_k=5 after |
