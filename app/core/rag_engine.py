"""
Retrieval + grounded-answer logic. Takes EmbeddingProvider, LLMProvider,
and VectorStore via constructor injection — this is the module the
checklist calls out as needing to be unit-testable with fakes, in pure
Python, with zero Docker/Ollama/Chroma running.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.interfaces import EmbeddingProvider, LLMProvider, VectorStore

SYSTEM_PROMPT = (
    "You are a question-answering assistant restricted to the provided "
    "context. Answer ONLY using the context below. If the answer is not "
    "contained in the context, respond exactly with: "
    "\"I don't have that information in the indexed documents.\" "
    "Do not use outside knowledge, do not guess, and do not fill gaps "
    "with general knowledge even if you are confident in it."
)

NO_CONTEXT_ANSWER = "I don't have that information in the indexed documents."


@dataclass
class RetrievedChunk:
    source_path: str
    chunk_id: str
    text: str
    distance: float


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    grounded: bool = False


class RagEngine:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        store: VectorStore,
        top_k: int,
        similarity_distance_threshold: float,
    ):
        self._embedder = embedder
        self._llm = llm
        self._store = store
        self._top_k = top_k
        self._threshold = similarity_distance_threshold

    def answer(self, question: str) -> RagAnswer:
        query_embedding = self._embedder.embed(question)
        raw_results = self._store.query(query_embedding, top_k=self._top_k)

        retrieved = [
            RetrievedChunk(
                source_path=r["metadata"].get("source_path", "unknown"),
                chunk_id=r["id"],
                text=r["document"],
                distance=r["distance"],
            )
            for r in raw_results
        ]

        # Strict grounding gate: if nothing came back, or the single
        # closest match is still farther than the threshold, refuse
        # WITHOUT calling the LLM at all. This is the behavior the
        # checklist's "Grounding logic" section requires explicitly.
        if not retrieved or retrieved[0].distance > self._threshold:
            return RagAnswer(answer=NO_CONTEXT_ANSWER, sources=[], grounded=False)

        context_block = "\n\n---\n\n".join(
            f"[Source: {c.source_path}]\n{c.text}" for c in retrieved
        )
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        raw_answer = self._llm.generate(SYSTEM_PROMPT, user_prompt)

        # Defense-in-depth: even with a strict system prompt, a model can
        # occasionally ignore instructions. If it produced the refusal
        # phrase itself, respect that as ungrounded too.
        grounded = NO_CONTEXT_ANSWER.strip().lower() not in raw_answer.strip().lower()

        return RagAnswer(answer=raw_answer, sources=retrieved, grounded=grounded)
