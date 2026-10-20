"""
Isolated tests for ollama_client.py using httpx.MockTransport: no Ollama
running, no network. Covers the embedding prefix behaviour.
"""
import json

import httpx

from app.core.ollama_client import OllamaClient


def _client(captured: list, **kwargs) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    return OllamaClient(
        host="http://ollama:11434",
        embedding_model="nomic-embed-text",
        llm_model="x",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_query_and_document_prefixes_are_applied():
    sent: list = []
    c = _client(sent, query_prefix="search_query", document_prefix="search_document")

    c.embed_query("hello")
    c.embed_documents(["doc one"])

    assert sent[0]["prompt"] == "search_query: hello"
    assert sent[1]["prompt"] == "search_document: doc one"


def test_empty_prefix_sends_text_unchanged():
    sent: list = []
    c = _client(sent, query_prefix="", document_prefix="")

    c.embed_query("hello")
    c.embed_documents(["doc one"])

    assert sent[0]["prompt"] == "hello"
    assert sent[1]["prompt"] == "doc one"
