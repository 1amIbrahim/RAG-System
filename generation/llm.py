import os
import requests
from typing import List, Dict, Optional

from .prompts import build_prompt

_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "mistral"
_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def _ollama_available() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _generate_ollama(prompt: str, model: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(_OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _generate_hf_api(prompt: str) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=_HF_MODEL, token=_hf_token())
    return client.text_generation(prompt, max_new_tokens=512, temperature=0.1).strip()


def _generate_hf_local(prompt: str) -> str:
    from transformers import pipeline
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    return pipe(prompt[:2048])[0]["generated_text"].strip()


def generate(prompt: str, model: str = _DEFAULT_MODEL) -> str:
    if _ollama_available():
        return _generate_ollama(prompt, model)

    token = _hf_token()
    if token:
        try:
            print("[INFO] Using HuggingFace Inference API")
            return _generate_hf_api(prompt)
        except Exception as e:
            print(f"[WARN] HF Inference API failed: {e} -- falling back to local model")

    print("[WARN] No Ollama or HF token -- falling back to flan-t5-base (demo quality)")
    return _generate_hf_local(prompt)


def answer(
    query: str,
    chunks: List[Dict],
    model: str = _DEFAULT_MODEL,
) -> Dict:
    prompt = build_prompt(query, chunks)
    response_text = generate(prompt, model=model)
    sources = [
        {"source": c.get("source", "unknown"), "page": c.get("page", "?")}
        for c in chunks
    ]
    return {"answer": response_text, "sources": sources}
