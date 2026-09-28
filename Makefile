.PHONY: up dev test load

up:
	docker-compose up -d

dev:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest -v

load:
	curl -X POST http://localhost:8000/api/load -F "file=@data/LF Jobs.xlsx" -F "overwrite=true"