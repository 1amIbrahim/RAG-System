from typing import List, Dict
from .tokenizer import count_tokens
import uuid

class SemanticChunker:
    def __init__(
        self,
        max_tokens: int = 500,
        overlap: int = 100
    ):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk_pages(self, pages: List[Dict], source: str):
        chunks = []
        buffer = ""
        buffer_tokens = 0
        chunk_id = 0

        for page in pages:
            paragraphs = [
                p.strip() for p in page["text"].split("\n\n") if p.strip()
            ]

            for para in paragraphs:
                para_tokens = count_tokens(para)

                if buffer_tokens + para_tokens > self.max_tokens:
                    chunks.append(self._make_chunk(
                        buffer,
                        source,
                        page["page"],
                        chunk_id
                    ))
                    chunk_id += 1

                    # overlap
                    overlap_text = self._get_overlap(buffer)
                    buffer = overlap_text + para
                    buffer_tokens = count_tokens(buffer)
                else:
                    buffer += "\n\n" + para
                    buffer_tokens += para_tokens

        if buffer.strip():
            chunks.append(self._make_chunk(
                buffer,
                source,
                page["page"],
                chunk_id
            ))

        return chunks

    def _get_overlap(self, text: str):
        tokens = text.split()
        return " ".join(tokens[-self.overlap:]) + "\n\n"

    def _make_chunk(self, text, source, page, chunk_id):
        return {
            "chunk_id": f"{source}_{chunk_id}",
            "source": source,
            "page": page,
            "text": text.strip(),
            "tokens": count_tokens(text)
        }
