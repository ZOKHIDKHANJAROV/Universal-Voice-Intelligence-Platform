"""Minimal Ollama client for answer generation."""

import json
import re
import urllib.error
import urllib.request


class LlmError(Exception):
    """The LLM could not produce a usable answer; the caller gets a pre-recorded one."""


_THINK = re.compile(r"<think>.*?</think>", re.S)


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LlmError(f"HTTP {exc.code}: {detail}") from exc
    except (OSError, ValueError) as exc:
        raise LlmError(str(exc)) from exc


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: float,
    max_tokens: int = 200,
    gpu_layers: int | None = None,
) -> str:
    """One non-streaming chat turn; returns the reply text."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Reasoning models (qwen3) would spend seconds thinking out loud.
        "think": False,
        # Keep the model loaded between calls instead of reloading for each one.
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_predict": max_tokens},
    }
    if gpu_layers is not None:
        payload["options"]["num_gpu"] = gpu_layers
    try:
        data = _post(url, payload, timeout)
    except LlmError as exc:
        if "think" not in str(exc):
            raise
        # Models without a thinking mode reject the flag.
        payload.pop("think")
        data = _post(url, payload, timeout)
    try:
        text = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise LlmError(f"Unexpected reply: {str(data)[:200]}") from exc
    return _THINK.sub("", text).strip()


def warm_up(base_url: str, model: str, gpu_layers: int | None = None, timeout: float = 120.0) -> None:
    """Load the model into memory so the first caller does not wait for it."""
    payload = {"model": model, "prompt": "", "keep_alive": "30m"}
    if gpu_layers is not None:
        payload["options"] = {"num_gpu": gpu_layers}
    _post(f"{base_url.rstrip('/')}/api/generate", payload, timeout)


def status(base_url: str, model: str, timeout: float = 3.0) -> dict:
    """Is the server up and the model downloaded?"""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout) as response:
            models = [m["name"] for m in json.loads(response.read().decode("utf-8")).get("models", [])]
    except (OSError, ValueError, KeyError) as exc:
        return {"reachable": False, "models": [], "model_ready": False, "error": str(exc)}
    ready = model in models or f"{model}:latest" in models
    return {"reachable": True, "models": models, "model_ready": ready, "error": None}
