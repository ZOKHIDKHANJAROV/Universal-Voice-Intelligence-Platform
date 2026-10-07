.PHONY: install run test lint docker-up render-prompts uz-model

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

# Download the Uzbek Whisper fine-tune and convert it for faster-whisper.
# Needs `pip install -e ".[convert]"`. Xet is disabled because it stalls on
# some networks; the download resumes if the connection drops.
UZ_MODEL_REPO ?= navai-uz/whisper-medium-uzbek
uz-model:
	HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('$(UZ_MODEL_REPO)', local_dir='models/_src/whisper-medium-uzbek', allow_patterns=['*.json', '*.txt', '*.safetensors'])"
	python -m scripts.convert_whisper models/_src/whisper-medium-uzbek models/whisper-medium-uzbek-ct2
