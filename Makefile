.PHONY: install run test lint docker-up

install:
	python -m pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload

test:
	pytest -q

lint:
	python -m compileall app tests

docker-up:
	docker compose up --build
