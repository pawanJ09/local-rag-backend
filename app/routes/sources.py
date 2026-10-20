from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.vector_store import ChromaStore
from app.deps import get_vector_store, require_api_key
from app.models import ConfigResponse, SourceInfo, SourcesListResponse

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/sources", response_model=SourcesListResponse)
def list_sources(store: ChromaStore = Depends(get_vector_store)) -> SourcesListResponse:
    sources = store.get_all_sources()
    return SourcesListResponse(
        sources=[
            SourceInfo(
                source_path=path,
                chunk_count=info["chunk_count"],
                file_hash=info["file_hash"],
                last_indexed=info["last_indexed"],
            )
            for path, info in sources.items()
        ]
    )


@router.delete("/sources/{file_path:path}")
def delete_source(file_path: str, store: ChromaStore = Depends(get_vector_store)) -> dict:
    store.delete_by_source(file_path)
    return {"deleted": file_path}


@router.get("/config", response_model=ConfigResponse)
def get_config(settings: Settings = Depends(get_settings)) -> ConfigResponse:
    return ConfigResponse(
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        similarity_distance_threshold=settings.similarity_distance_threshold,
        allowed_extensions=settings.allowed_extensions_list,
    )
