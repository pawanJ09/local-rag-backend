from fastapi import APIRouter, Depends

from app.core.ollama_client import OllamaClient
from app.core.vector_store import ChromaStore
from app.deps import get_ollama_client, get_vector_store
from app.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    ollama: OllamaClient = Depends(get_ollama_client),
    store: ChromaStore = Depends(get_vector_store),
) -> HealthResponse:
    ollama_ok = ollama.is_reachable()
    chroma_ok = store.is_reachable()
    return HealthResponse(
        status="ok" if (ollama_ok and chroma_ok) else "degraded",
        ollama_reachable=ollama_ok,
        chroma_reachable=chroma_ok,
    )
