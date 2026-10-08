# Start UniVoice on one Windows laptop: Asterisk in Docker, the API on the host GPU.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1
#
# Needs Docker Desktop running, .venv with the project installed, and a .env
# (see docs/local-run.md). Stop with Ctrl+C, then:
#   docker compose -p universal-voice-intelligence-platform stop asterisk
#
# With -Docker the API and web UI run in a container too (no .venv needed):
#   powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1 -Docker
# Stop with:
#   docker compose -p universal-voice-intelligence-platform stop api asterisk

param([switch]$Docker)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Reuse the images and volumes built from the main checkout.
$project = "universal-voice-intelligence-platform"

function Write-Usage {
    Write-Host ""
    Write-Host "Asterisk: SIP 127.0.0.1:5060, user 1000, call extension 1000"
    Write-Host "Operator: register a second softphone as 1001 to receive 'press 0' transfers"
    Write-Host "Console:  http://localhost:8000/   (models warm up for ~10 s after start)"
    Write-Host "Monitor:  http://localhost:8000/monitor"
    Write-Host ""
}

if ($Docker) {
    # Asterisk reaches the api container by its compose name (the defaults).
    Remove-Item Env:UNIVOICE_AUDIOSOCKET, Env:UNIVOICE_API -ErrorAction SilentlyContinue
    docker compose -p $project up -d --build api asterisk
    if ($LASTEXITCODE -ne 0) { throw "Could not start the containers. Is Docker Desktop running?" }
    Write-Usage
    Write-Host "Logs: docker compose -p $project logs -f api"
    exit 0
}

# Asterisk streams call audio to the API on the host, not to an api container.
$env:UNIVOICE_AUDIOSOCKET = "host.docker.internal:9019"
$env:UNIVOICE_API = "host.docker.internal:8000"
docker compose -p $project up -d --no-deps asterisk
if ($LASTEXITCODE -ne 0) { throw "Could not start Asterisk. Is Docker Desktop running?" }

Write-Usage

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
