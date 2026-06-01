import os
import requests
from typing import List, Dict, Optional

from .prompts import build_prompt

_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "mistral"
_HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"


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
    token = _hf_token()
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Add it as a Space secret or set it locally to use the HF Inference API."
        )
    client = InferenceClient(model=_HF_MODEL, token=token, provider="hf-inference")
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def generate(prompt: str, model: str = _DEFAULT_MODEL) -> str:
    if _ollama_available():
        return _generate_ollama(prompt, model)
    return _generate_hf_api(prompt)


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
