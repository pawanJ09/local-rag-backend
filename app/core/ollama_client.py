"""
Thin wrapper around Ollama's HTTP API. Implements both EmbeddingProvider
and LLMProvider from interfaces.py. No other module should import httpx
to talk to Ollama directly — this is the single seam.
"""
from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(self, host: str, embedding_model: str, llm_model: str, timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self._client = httpx.Client(base_url=self.host, timeout=timeout)

    # ---- EmbeddingProvider ----

    def embed(self, text: str) -> list[float]:
        resp = self._client.post(
            "/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama's /api/embeddings is single-text; batch by looping.
        # (Kept here rather than in the engine so callers never need to
        # know this is sequential under the hood.)
        return [self.embed(t) for t in texts]

    # ---- LLMProvider ----

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.post(
            "/api/chat",
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # ---- health ----

    def is_reachable(self) -> bool:
        try:
            resp = self._client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
