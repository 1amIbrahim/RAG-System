from typing import List, Dict


_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided "
    "context. Cite sources inline as [1], [2], etc. If the context does not contain "
    "enough information, say so — do not fabricate facts."
)

_TEMPLATE = """{system}

Context:
{context}

Question: {query}

Answer:"""


def build_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Build a grounded generation prompt from a query and retrieved chunks.

    Each chunk is numbered and shown with its source file and page so the
    model can reference them as [1], [2], etc.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk["text"].strip()
        context_parts.append(f"[{i}] (source: {source}, page {page})\n{text}")

    context = "\n\n".join(context_parts)
    return _TEMPLATE.format(system=_SYSTEM, context=context, query=query)
