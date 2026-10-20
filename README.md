# local-rag-backend

FastAPI service: ingests a local document directory, embeds it via a local
Ollama model, stores vectors in Chroma, and answers questions strictly
grounded in retrieved content. This repo is the **only** one that talks to
Ollama or Chroma — `local-rag-ui` and `local-rag-cli` are pure clients of
the HTTP API below.

## Quickstart

```bash
cp .env.example .env     # adjust models/key/DOCS_HOST_DIR if desired
make up                  # docker compose up -d
make pull-models         # one-time: pulls embedding + LLM models into the ollama container
make ingest               # triggers ingestion of ./docs
```

Then query it:

```bash
curl -s -X POST http://localhost:8000/query \
  -H "X-API-Key: changeme-local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}' | python3 -m json.tool
```

Interactive API docs: `http://localhost:8000/docs`
Machine-readable schema (source of truth for UI/CLI types): `http://localhost:8000/openapi.json`

## API contract

All routes except `/health` require an `X-API-Key` header matching `API_KEY`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness: is Ollama reachable, is Chroma reachable |
| POST | `/ingest` | Triggers an async (re-)scan of the mounted docs directory. Returns `{job_id}` |
| GET | `/ingest/status/{job_id}` | Poll ingestion progress/result |
| POST | `/query` | `{"question": "..."}` → `{"answer", "sources", "grounded"}` |
| GET | `/sources` | List currently indexed files with chunk counts |
| DELETE | `/sources/{file_path}` | Remove one file's chunks from the index |
| GET | `/config` | Active model names, chunk size, similarity threshold |

### `grounded: true/false`

This is the explicit flag the UI/CLI should check — never infer "no answer"
by parsing the text. `grounded: false` means retrieval didn't clear the
similarity threshold (or, as a defense-in-depth check, the LLM itself
emitted the refusal phrase), and `answer` will be the fixed string
`"I don't have that information in the indexed documents."`

## Architecture notes

- `app/core/interfaces.py` defines `EmbeddingProvider`, `LLMProvider`,
  `VectorStore` as `Protocol`s. `rag_engine.py` and `ingest_engine.py`
  depend on these, never on concrete classes — this is what makes them
  unit-testable with fakes (see `tests/test_rag.py`, `tests/test_ingest.py`)
  with zero Docker/Ollama/Chroma running.
- `app/deps.py` is the composition root: it constructs the concrete
  `OllamaClient` / `ChromaStore` and wires them into the engines via
  FastAPI `Depends()`. Route tests override these with fakes via
  `app.dependency_overrides` (see `tests/test_routes.py`).

## Running tests

```bash
make install   # pip install -r app/requirements.txt (once, locally)
make test      # python3 -m pytest tests -v
```

All three test files run with **no Docker, Ollama, or Chroma required** —
they test engines and routes in isolation with fakes. The only tier that
needs the real stack running is a manual end-to-end check via `curl`
against a live `docker compose up` (no automated smoke test lives in this
repo; that's `local-rag-deploy`'s `test-stack.yml`).

## Supported document formats

`.txt`, `.md`, `.pdf`, `.docx` — configurable via `ALLOWED_EXTENSIONS`.
Unsupported or unreadable files are skipped and logged in the ingest job's
`skipped_files`, not treated as a failure.

## Choosing which folder to ingest

Set `DOCS_HOST_DIR` in `.env` to any folder on your machine (default `./docs`).
It is mounted read-only into the container at `/data/docs`, which is what the
backend scans (recursively). After changing it, run `docker compose up -d` so
the container is recreated with the new mount, then `POST /ingest`. On macOS
the folder must be under a path Docker Desktop is allowed to share.

## Known nuances / deviations from the original checklist (flagged, not hidden)

- **`DOCS_HOST_DIR` env var** (compose volume source) wasn't in the original
  checklist; added so the ingest folder is set from `.env` instead of editing
  `docker-compose.yml`.

- **`app/deps.py`** wasn't itemized in the agreed backend file tree. FastAPI's
  dependency-injection pattern needed a composition root somewhere; putting
  it in `main.py` would have made that file do both routing and wiring.
  Flagging this addition rather than silently expanding the tree — happy to
  fold it into `main.py` if you'd rather keep the file count exactly as
  agreed.
- **Per-format text extraction** (`.pdf`/`.docx`/`.txt`/`.md`) lives as
  private functions inside `ingest_engine.py` rather than a separate
  `extractors.py`. Kept it there specifically to avoid adding a file beyond
  the checklist's list; if the format list grows (CSV, OCR, etc.) this
  should probably split out.
- **Incremental re-ingestion** (checklist: "works via file hash/mtime
  tracking") is implemented by reading existing `file_hash` values back out
  of Chroma's chunk metadata (`get_all_sources()`), rather than a separate
  manifest file. This avoids a new persisted artifact outside the agreed
  structure, at the cost of one full metadata scan per ingest run — fine at
  this project's scale, worth revisiting if the corpus grows very large.
- **PDF text uses `pypdf` layout mode** (plus whitespace normalization) so
  filled-in form values stay next to their labels. Found with real lease PDFs,
  where default extraction put "Monthly Rent" and the amount far apart, so the
  amount never matched questions about rent. Existing indexed PDFs must be
  re-ingested (delete their sources or reset `chroma_data/`), since unchanged
  files are skipped by hash. Limit: two-column rows lose their column
  alignment once spaces are collapsed.
- **Chunking is character-count based**, not token-based (no tokenizer
  dependency pulled in). `CHUNK_SIZE`/`CHUNK_OVERLAP` are in characters.
  Simple and dependency-free; can swap for a token-aware splitter later if
  chunk boundaries cut mid-sentence too often in practice.
- **`SIMILARITY_DISTANCE_THRESHOLD` default (0.45)** is a reasonable
  starting point for `nomic-embed-text` cosine distance, not a tuned value
  — expect to adjust it after testing against your actual corpus.
- Tests use **no Docker network** (all fakes/`tmp_path`), so they were
  written but not executed in this sandbox (no outbound network here to
  `pip install`). Run `make install && make test` locally to confirm —
  flagging this so it isn't mistaken for "already verified."
