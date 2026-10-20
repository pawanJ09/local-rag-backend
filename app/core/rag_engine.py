"""
Retrieval + grounded-answer logic. Takes EmbeddingProvider, LLMProvider,
and VectorStore via constructor injection — this is the module the
checklist calls out as needing to be unit-testable with fakes, in pure
Python, with zero Docker/Ollama/Chroma running.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.interfaces import EmbeddingProvider, LLMProvider, VectorStore

NO_CONTEXT_ANSWER = "I don't have that information in the indexed documents."
# Any answer containing this phrase is treated as a refusal, even if the model
# rewords the rest of the sentence.
REFUSAL_MARKER = "i don't have that information"

_BASE_PROMPT = (
    "You answer questions using ONLY the provided context, which consists of "
    "excerpts from the user's own documents. Each excerpt starts with a "
    "[Source: path] line showing which file it came from. Take facts only "
    "from the excerpt text. Text from forms or tables may show a row of "
    "labels followed by a row of their values, in the same order. Do not use "
    "outside knowledge and do not guess. If the context does not contain the "
    f"answer, respond exactly with: \"{NO_CONTEXT_ANSWER}\""
)

# Opt-in (PATH_AS_EVIDENCE=true). Only appropriate when the user's folder and
# file names reliably describe their documents; for messy folder structures
# it can make the model trust misleading names, so it is off by default.
_PATH_EVIDENCE_PARAGRAPH = (
    " The folder and file names in the [Source: path] line are trusted "
    "metadata about each document (for example the property, client, "
    "location or year). Use them to decide which excerpt a question refers "
    "to, even when the excerpt's own text does not repeat that detail."
)


def build_system_prompt(path_as_evidence: bool = False) -> str:
    return _BASE_PROMPT + (_PATH_EVIDENCE_PARAGRAPH if path_as_evidence else "")


# Strict default, kept as a module constant for scripts that import it.
SYSTEM_PROMPT = build_system_prompt(False)


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
        path_as_evidence: bool = False,
    ):
        self._system_prompt = build_system_prompt(path_as_evidence)
        self._embedder = embedder
        self._llm = llm
        self._store = store
        self._top_k = top_k
        self._threshold = similarity_distance_threshold

    def answer(self, question: str) -> RagAnswer:
        query_embedding = self._embedder.embed_query(question)
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

        raw_answer = self._llm.generate(self._system_prompt, user_prompt)

        # Defense-in-depth: even with a strict system prompt, a model can
        # occasionally ignore instructions. If it produced the refusal
        # phrase itself, respect that as ungrounded too.
        normalized = raw_answer.strip().lower().replace("\u2019", "'")
        grounded = REFUSAL_MARKER not in normalized

        return RagAnswer(answer=raw_answer, sources=retrieved, grounded=grounded)
