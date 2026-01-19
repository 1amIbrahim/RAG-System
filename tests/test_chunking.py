from ingestion.loaders import load_document
from ingestion.chunker import SemanticChunker
from pathlib import Path
from ingestion.metadata import save_chunks

def test_chunking():
    pages = load_document(Path("data/raw/sample.pdf"))
    chunker = SemanticChunker()
    chunks = chunker.chunk_pages(pages, "sample.pdf")

    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all(c["tokens"] <= 600 for c in chunks)

    output_path = Path("data/processed/sample_chunks.json")
    save_chunks(chunks, output_path)
    print(f"✅ Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    test_chunking()
    print("✅ Chunking works")
