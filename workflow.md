# RAG-System — Project Status & Roadmap

Short description
A Retrieval-Augmented Generation (RAG) system for querying document collections with hybrid retrieval (BM25 + vector), reranking, and grounded generation. This document summarizes what is implemented, what remains, and the concrete next steps to complete the project.

---

## 1) Current status — Implemented

Files implemented (Week 1 + initial Week 2 work)
- ingestion/
  - loaders.py — PDF/DOCX/TXT loaders (page-aware PDF extraction)
  - chunker.py — Semantic chunker (paragraph-based, overlap, token-aware)
  - metadata.py — Save chunks to JSON
  - tokenizer.py — Token counting utility (tiktoken)
- tests/
  - test_chunking.py — Chunking test (saves processed JSON when updated)
- README.md — minimal (this file will replace)
- Initial retrieval prototypes attempted in repo (tests/test_retrieval.py was executed but retrieval package files may be missing or incomplete)

What currently works
- Document loading from PDF/DOCX/TXT into page objects
- Semantic chunking into metadata-rich chunks and saving to data/processed/*.json
- Local unit test for chunking (fix test to call save_chunks to generate processed files)

Known issues found during tests
- Module import issues resolved by adding __init__.py to packages
- FAISS / vector search produced invalid scores (overflow/negatives) — requires fixes:
  - Ensure embeddings normalization, correct FAISS index type and distance → similarity conversion
- BM25 negative scores indicated tokenization/indexing bug or empty queries — requires validation

---

## 2) Project structure (recommended canonical layout)

rag-system/
- ingestion/
  - __init__.py
  - loaders.py
  - chunker.py
  - tokenizer.py
  - metadata.py
- retrieval/
  - __init__.py
  - embeddings.py
  - vector.py
  - bm25.py
  - hybrid.py
- reranking/
  - __init__.py
  - cross_encoder.py
- generation/
  - __init__.py
  - llm.py
  - prompts.py
- evaluation/
  - __init__.py
  - metrics.py
- ui/
  - __init__.py
  - app.py
- data/
  - raw/
  - processed/
  - index/
- tests/
  - __init__.py
  - test_chunking.py
  - test_retrieval.py
  - test_generation.py
- requirements.txt
- Dockerfile / docker-compose.yml
- README.md

---

## 3) Dependencies (to include in requirements.txt)
- PyMuPDF
- python-docx
- tiktoken
- sentence-transformers
- faiss-cpu
- rank-bm25
- transformers
- torch
- streamlit
- numpy
- pandas

(Use pip install <package> or pip install -r requirements.txt)

---

## 4) Remaining work — high priority (Week 2 essentials)
1. Fix package imports
   - Add ingestion/__init__.py, retrieval/__init__.py, tests/__init__.py
2. Complete retrieval modules
   - retrieval/embeddings.py — wrap SentenceTransformer (bge-small-en-v1.5)
   - retrieval/vector.py — build FAISS index, normalize embeddings, use correct index type, convert distances to similarity
   - retrieval/bm25.py — validate tokenization and index building
   - retrieval/hybrid.py — score normalization and weighted fusion
3. Unit tests & sample data
   - Ensure data/processed/sample_chunks.json exists (test_chunking must save)
   - Add tests/test_retrieval.py and run via python -m tests.test_retrieval
4. Fix numeric issues discovered
   - Normalize embeddings before adding to FAISS
   - Use IndexFlatL2 or appropriate index; compute similarity = 1 - (distance / 2) for normalized vectors
   - Ensure BM25 tokenization returns non-empty tokens

---

## 5) Remaining work — medium priority (Week 3)
1. Re-ranking
   - Implement reranking/cross_encoder.py (cross-encoder model or lightweight transformer)
   - Evaluate top-N → top-K rerank performance
2. Generation layer
   - generation/llm.py — integrate chosen LLM backend (Ollama or HF + vLLM)
   - generation/prompts.py — prompts with explicit citation formatting
3. Tests
   - tests/test_generation.py — end-to-end test to ensure LLM receives context + citations

---

## 6) Remaining work — lower priority / polish (Weeks 4–5)
1. Streamlit UI (ui/app.py)
   - Upload docs, chat interface, sources viewer, debug panel
2. Evaluation & metrics
   - evaluation/metrics.py — faithfulness, precision@k, LLM-as-judge scripts
3. Persistence & deployment
   - Save/load FAISS index and chunk metadata in data/index/
   - Dockerfile + docker-compose.yml
   - GitHub Actions for tests & linting
4. README, diagrams, demo GIF, usage examples

---

## 7) Concrete next steps & commands (start now — Week 2 items)

A. Create package init files
PowerShell:
- New-Item ingestion/__init__.py -Force
- New-Item retrieval/__init__.py -Force
- New-Item tests/__init__.py -Force

B. Ensure chunk saving (update tests/test_chunking.py)
- Add `from ingestion.metadata import save_chunks` and call `save_chunks(chunks, Path("data/processed/sample_chunks.json"))`

C. Implement retrieval files (create these files):
- retrieval/embeddings.py — load SentenceTransformer and provide embed_texts, embed_query
- retrieval/vector.py — build_index, search (normalize + convert L2→similarity), save/load
- retrieval/bm25.py — build_index, search with robust tokenization
- retrieval/hybrid.py — fuse scores (vector_weight default 0.7, bm25_weight 0.3)

D. Install dependencies
PowerShell:
- pip install sentence-transformers faiss-cpu rank-bm25 tiktoken

E. Run tests
PowerShell:
- python -m tests.test_chunking
- python -m tests.test_retrieval

---

## 8) Acceptance criteria (How to know a step is done)

Week 2 done when:
- data/processed/sample_chunks.json exists and is correct
- Embedding model loads and returns consistent vectors
- FAISS index builds without numeric overflow
- BM25 index builds and returns positive scores
- Hybrid retriever returns sensible ranked chunks for sample queries
- tests/test_retrieval.py runs end-to-end without errors

Week 3 done when:
- Cross-encoder reranker improves precision in tests
- LLM generation returns answers with inline citations using retrieved chunks
- tests/test_generation.py passes for sample queries

Final project done when:
- Streamlit UI functional and demonstrates upload/chat/sources
- README and architecture diagram present
- Dockerfile works, basic CI configured
- Evaluation metrics report demonstrates retrieval + generation quality

---

## 9) Timeline (10–12 hours/week)

Week 2 (1 week): Retrieval engine (embeddings, FAISS, BM25, hybrid)  
Week 3 (1 week): Reranking + generation integration (LLM prompts & citation)  
Week 4 (1 week): UI + end-to-end polish  
Week 5 (1 week): Evaluation, Docker, CI, README + deploy demo

---

## 10) Notes & tips
- Use `python -m tests.test_*` to run tests from project root (fixes import path issues).  
- Save indices to data/index/ to avoid re-embedding on each run.  
- Normalize embeddings consistently (both index and query).  
- Tune hybrid weights on a small validation set.  
- Keep chunk tokens ~400–600 with ~80–120 overlap for best trade-off.

---

## 11) Quick checklist (tick off locally)
- [ ] Add __init__.py files
- [ ] Ensure test_chunking saves JSON
- [ ] Implement retrieval modules
- [ ] Fix FAISS normalization and index type
- [ ] Add retrieval tests and validate output
- [ ] Implement reranker + LLM generation
- [ ] Build Streamlit demo
- [ ] Add Docker + CI
- [ ] Final README, diagrams, and demo GIF

---

If you want, I will:
- Create the retrieval module files next (embeddings.py, vector.py, bm25.py, hybrid.py) and tests with exact code.
- Or update tests/test_chunking.py to save chunks and create sample data files.

Reply with which action to run now.