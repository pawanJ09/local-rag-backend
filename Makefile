.PHONY: up down logs ingest test pull-models build install

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

pull-models:
	docker compose exec ollama ollama pull $${EMBEDDING_MODEL:-nomic-embed-text}
	docker compose exec ollama ollama pull $${LLM_MODEL:-llama3.1:8b}

ingest:
	curl -s -X POST http://localhost:8000/ingest -H "X-API-Key: $${API_KEY:-changeme-local-dev-key}" | python3 -m json.tool

build:
	docker compose build

install:
	pip install -r app/requirements.txt

test:
	python3 -m pytest tests -v
