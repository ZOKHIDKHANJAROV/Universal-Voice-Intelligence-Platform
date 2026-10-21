import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.logs import JsonFormatter, _CallFilter, bind_call, current_call, setup_logging
from app.main import app

client = TestClient(app)


def test_metrics_endpoint_lists_the_bot_metrics() -> None:
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for name in (
        "univoice_active_calls",
        'univoice_calls_total{source="phone"}',
        'univoice_answers_total{outcome="not_recognized",source="phone"}',
        'univoice_llm_answers_total{result="fallback",source="phone"}',
        'univoice_operator_transfers_total{trigger="dtmf"}',
        "univoice_operator_unavailable_total",
        'univoice_errors_total{stage="utterance"}',
        'univoice_http_requests_total{method="GET",route="/health",status="200"}',
    ):
        assert name in response.text, name


def test_http_metrics_use_route_templates_not_paths() -> None:
    client.get("/api/v1/monitor/calls/some-call-id")
    client.get("/no/such/page")
    text = client.get("/metrics").text
    assert 'route="/api/v1/monitor/calls/{call_id}"' in text
    assert 'route="unmatched"' in text
    assert "some-call-id" not in text


def test_metrics_need_the_api_key_when_one_is_set(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "api_key", "secret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert client.get("/metrics", headers={"X-API-Key": "secret"}).status_code == 200


def _formatted(logger_name: str, message: str, *args, **kwargs) -> dict:
    record = logging.getLogger(logger_name).makeRecord(
        logger_name, logging.INFO, __file__, 1, message, args, None, extra=kwargs or None
    )
    _CallFilter().filter(record)
    return json.loads(JsonFormatter().format(record))


def test_json_log_lines_carry_the_call() -> None:
    async def call() -> tuple[dict, dict, dict]:
        context = bind_call("conn-1")
        before_uuid = _formatted("univoice.realtime", "connected")
        # Asterisk sends the UUID later; work started before then sees it too.
        context.call_uuid = "1f0e-uuid"
        in_thread = await asyncio.to_thread(_formatted, "univoice.stt", "STT text=%r", "salom")
        in_task = await asyncio.create_task(asyncio.to_thread(_formatted, "univoice.realtime", "x", stage="tts"))
        return before_uuid, in_thread, in_task

    before_uuid, in_thread, in_task = asyncio.run(call())
    assert before_uuid["call_id"] == "conn-1" and "call_uuid" not in before_uuid
    assert in_thread["call_uuid"] == "1f0e-uuid"
    assert in_thread["message"] == "STT text='salom'"
    assert in_thread["logger"] == "univoice.stt" and in_thread["level"] == "INFO"
    assert in_task["call_uuid"] == "1f0e-uuid" and in_task["stage"] == "tts"


def test_lines_outside_a_call_have_no_call_fields() -> None:
    async def outside() -> dict:
        return _formatted("univoice", "started")

    assert current_call() is None
    line = asyncio.run(outside())
    assert "call_uuid" not in line and "call_id" not in line


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_setup_logging_routes_uvicorn_through_one_handler(fmt, capsys) -> None:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        setup_logging("INFO", fmt)
        logging.getLogger("uvicorn.error").info("Started server process")
        out = capsys.readouterr().out.strip().splitlines()[-1]
        if fmt == "json":
            assert json.loads(out)["logger"] == "uvicorn.error"
        else:
            assert "uvicorn.error: Started server process" in out
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
