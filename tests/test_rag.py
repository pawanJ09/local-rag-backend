"""
Per the checklist's Testability section: rag_engine's grounding logic is
the single most important piece of correctness in this project, and it
must be testable with fakes alone — no Docker, no Ollama, no Chroma.
"""
from app.core.rag_engine import RagEngine, NO_CONTEXT_ANSWER


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class FakeLLM:
    def __init__(self, canned_response: str = "A real grounded answer."):
        self.canned_response = canned_response
        self.last_user_prompt: str | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_user_prompt = user_prompt
        return self.canned_response


class FakeStore:
    def __init__(self, results: list[dict]):
        self._results = results

    def upsert(self, ids, embeddings, documents, metadatas):
        raise NotImplementedError

    def query(self, embedding, top_k):
        return self._results[:top_k]

    def delete_by_source(self, source_path):
        raise NotImplementedError

    def get_all_sources(self):
        raise NotImplementedError


def _chunk(distance: float, text: str = "some context", source: str = "doc.txt"):
    return {
        "id": "chunk-1",
        "document": text,
        "metadata": {"source_path": source},
        "distance": distance,
    }


def test_query_within_threshold_calls_llm_and_is_grounded():
    store = FakeStore([_chunk(distance=0.1)])
    llm = FakeLLM("The answer is 42.")
    engine = RagEngine(
        embedder=FakeEmbedder(),
        llm=llm,
        store=store,
        top_k=5,
        similarity_distance_threshold=0.45,
    )

    result = engine.answer("What is the answer?")

    assert result.grounded is True
    assert result.answer == "The answer is 42."
    assert len(result.sources) == 1
    assert llm.last_user_prompt is not None  # LLM was actually called


def test_query_beyond_threshold_refuses_without_calling_llm():
    store = FakeStore([_chunk(distance=0.9)])  # far away
    llm = FakeLLM("This should never be returned.")
    engine = RagEngine(
        embedder=FakeEmbedder(),
        llm=llm,
        store=store,
        top_k=5,
        similarity_distance_threshold=0.45,
    )

    result = engine.answer("Something unrelated to the corpus?")

    assert result.grounded is False
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert llm.last_user_prompt is None  # LLM must NOT have been called


def test_empty_store_refuses_without_calling_llm():
    store = FakeStore([])
    llm = FakeLLM("Should not appear.")
    engine = RagEngine(
        embedder=FakeEmbedder(),
        llm=llm,
        store=store,
        top_k=5,
        similarity_distance_threshold=0.45,
    )

    result = engine.answer("Anything?")

    assert result.grounded is False
    assert result.answer == NO_CONTEXT_ANSWER
    assert llm.last_user_prompt is None


def test_threshold_is_exact_boundary_inclusive():
    # distance exactly AT the threshold should pass (grounded path reached);
    # distance just above should not. Pins the boundary behavior down.
    threshold = 0.45
    store_at = FakeStore([_chunk(distance=0.45)])
    engine_at = RagEngine(
        embedder=FakeEmbedder(), llm=FakeLLM(), store=store_at, top_k=5,
        similarity_distance_threshold=threshold,
    )
    result_at = engine_at.answer("q")
    assert result_at.grounded is True

    store_over = FakeStore([_chunk(distance=0.450001)])
    engine_over = RagEngine(
        embedder=FakeEmbedder(), llm=FakeLLM(), store=store_over, top_k=5,
        similarity_distance_threshold=threshold,
    )
    result_over = engine_over.answer("q")
    assert result_over.grounded is False


def test_llm_ignoring_instructions_is_still_caught_as_ungrounded():
    # Defense-in-depth: even if retrieval passed the threshold, if the LLM
    # itself emits the refusal phrase, the response should report
    # grounded=False rather than silently trusting the LLM's judgment call.
    store = FakeStore([_chunk(distance=0.1)])
    llm = FakeLLM(NO_CONTEXT_ANSWER)
    engine = RagEngine(
        embedder=FakeEmbedder(), llm=llm, store=store, top_k=5,
        similarity_distance_threshold=0.45,
    )

    result = engine.answer("q")

    assert result.grounded is False
