from fastapi import APIRouter, Depends

from app.core.rag_engine import RagEngine
from app.deps import get_rag_engine, require_api_key
from app.models import QueryRequest, QueryResponse, SourceChunk

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest, engine: RagEngine = Depends(get_rag_engine)
) -> QueryResponse:
    result = engine.answer(request.question)
    return QueryResponse(
        answer=result.answer,
        grounded=result.grounded,
        sources=[
            SourceChunk(
                source_path=c.source_path,
                chunk_id=c.chunk_id,
                excerpt=c.text[:300],
                distance=c.distance,
            )
            for c in result.sources
        ],
    )
