from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import health, ingest, query, sources

settings = get_settings()

app = FastAPI(
    title="local-rag-backend",
    description=(
        "Local-only RAG service. Ingests a document directory, embeds it, "
        "and answers questions strictly grounded in retrieved content via "
        "a local Ollama model. See /docs for the full API contract that "
        "local-rag-ui and local-rag-cli build against."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(sources.router)
