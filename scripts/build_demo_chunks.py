"""
Generate data/processed/sample_chunks.json from a folder of documents.

Usage (run from the repo root):
    python scripts/build_demo_chunks.py demo_docs/

Put your PDF / DOCX / TXT files in demo_docs/ (or any folder you pass).
The output file is committed to the repo and baked into the Docker image
so the demo loads instantly on startup.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.loaders import load_document
from ingestion.chunker import SemanticChunker


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_demo_chunks.py <docs_folder>")
        sys.exit(1)

    docs_dir = Path(sys.argv[1])
    if not docs_dir.exists():
        print(f"Folder not found: {docs_dir}")
        sys.exit(1)

    supported = {".pdf", ".docx", ".txt"}
    files = sorted(f for f in docs_dir.iterdir() if f.suffix.lower() in supported)

    if not files:
        print(f"No PDF / DOCX / TXT files found in {docs_dir}")
        sys.exit(1)

    chunker = SemanticChunker(max_tokens=500, overlap=100)
    all_chunks = []

    for f in files:
        print(f"Processing: {f.name}")
        try:
            pages = load_document(f)
            chunks = chunker.chunk_pages(pages, source=f.name)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks from {len(pages)} page(s)")
        except Exception as e:
            print(f"  SKIP ({e})")

    out_path = Path("data/processed/sample_chunks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(all_chunks, fp, indent=2, ensure_ascii=False)

    total_tokens = sum(c.get("tokens", 0) for c in all_chunks)
    print(f"\nDone.")
    print(f"  Documents : {len(files)}")
    print(f"  Chunks    : {len(all_chunks)}")
    print(f"  Tokens    : {total_tokens:,}")
    print(f"  Saved to  : {out_path}")
    print(f"\nNext steps:")
    print(f"  git add data/processed/sample_chunks.json")
    print(f"  git commit -m 'chore: update demo chunks'")
    print(f"  git push")


if __name__ == "__main__":
    main()
