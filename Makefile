.PHONY: up migrate dev test load

up:
	docker-compose up -d

migrate:
	psql $$DATABASE_URL -f app/db/migrations/001_init.sql

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

load:
	curl -X POST http://localhost:8000/api/load -F "file=@data/lf_jobs.xlsx" -F "overwrite=true"