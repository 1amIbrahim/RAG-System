import os
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


_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _generate_hf_api(prompt: str) -> str:
    from huggingface_hub import InferenceClient
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add it as a Space secret."
        )
    client = InferenceClient(model=_HF_MODEL, token=token)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def _generate_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _generate_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def generate(prompt: str, model: str = _DEFAULT_MODEL) -> str:
    if _ollama_available():
        return _generate_ollama(prompt, model)

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"):
        return _generate_hf_api(prompt)

    if os.environ.get("ANTHROPIC_API_KEY"):
        return _generate_anthropic(prompt)

    if os.environ.get("OPENAI_API_KEY"):
        return _generate_openai(prompt)

    raise EnvironmentError(
        "No LLM backend available. Set HF_TOKEN, ANTHROPIC_API_KEY, or OPENAI_API_KEY "
        "as a Space secret, or run Ollama locally."
    )


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
