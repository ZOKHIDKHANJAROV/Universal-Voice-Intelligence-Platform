.PHONY: install run test lint docker-up render-prompts

install:
	python -m pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload

test:
	pytest -q

lint:
	python -m compileall app tests scripts

docker-up:
	docker compose up --build

# Start the GPU TTS only for as long as it takes to render the call audio,
# then stop it to give the VRAM back to Whisper.
render-prompts:
	docker compose --profile tts up -d --wait navoiy-tts
	docker compose run --rm --no-deps api python -m scripts.render_prompts
	docker compose stop navoiy-tts
