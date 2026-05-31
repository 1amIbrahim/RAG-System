import json
import requests
from typing import List, Dict

from .prompts import build_prompt

_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "mistral"


def _ollama_available() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _generate_ollama(prompt: str, model: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(_OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _generate_hf(prompt: str) -> str:
    from transformers import pipeline

    pipe = pipeline("text-generation", model="gpt2", max_new_tokens=256)
    result = pipe(prompt, do_sample=False)[0]["generated_text"]
    # Strip the prompt prefix so we return only the generated continuation
    return result[len(prompt):].strip()


def generate(prompt: str, model: str = _DEFAULT_MODEL) -> str:
    """
    Generate text from a prompt.

    Tries Ollama first (local LLM server); falls back to a HuggingFace
    pipeline if Ollama is not running.
    """
    if _ollama_available():
        return _generate_ollama(prompt, model)
    print("⚠️  Ollama not running — falling back to HuggingFace GPT-2 (demo only)")
    return _generate_hf(prompt)


def answer(
    query: str,
    chunks: List[Dict],
    model: str = _DEFAULT_MODEL,
) -> Dict:
    """
    Full RAG generation step: build prompt → generate → return structured result.

    Returns:
        {"answer": str, "sources": [{"source": str, "page": int/str}]}
    """
    prompt = build_prompt(query, chunks)
    response_text = generate(prompt, model=model)

    sources = [
        {"source": c.get("source", "unknown"), "page": c.get("page", "?")}
        for c in chunks
    ]
    return {"answer": response_text, "sources": sources}
