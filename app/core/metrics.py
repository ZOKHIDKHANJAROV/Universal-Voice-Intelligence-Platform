"""Prometheus metrics served at /metrics (see README "Observability").

Phone calls report with source="phone", web console tests with source="console",
so dashboards can leave console traffic out.
"""

import logging

from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily

LOGGER = logging.getLogger("univoice.metrics")

PHONE = "phone"
CONSOLE = "console"

# Pipeline stages timed per caller turn:
#   vad        last voiced frame -> utterance handed to STT (end-of-speech wait)
#   stt        speech recognition
#   intent     scenario matching and choosing the reply
#   rag, llm   maximum mode: knowledge search, LLM answer
#   tts        reply audio ready (pre-rendered, rendered on first use, or live)
#   end_to_end caller stops talking -> reply audio ready to play
STAGES = ("vad", "stt", "intent", "rag", "llm", "tts", "end_to_end")

ACTIVE_CALLS = Gauge("univoice_active_calls", "Phone calls connected to the bot right now")
CALLS = Counter("univoice_calls_total", "Calls started", ["source"])
CALL_ENDS = Counter(
    "univoice_call_ends_total",
    "Phone sessions with the bot that ended, by reason "
    "(caller_hangup, transferred, done, silence, time_limit, error)",
    ["reason"],
)
STAGE_SECONDS = Histogram(
    "univoice_stage_seconds",
    "Latency of one caller turn per pipeline stage",
    ["stage", "source"],
    buckets=(0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, 30.0),
)
ANSWERS = Counter(
    "univoice_answers_total",
    "Caller turns answered: outcome=scenario (a scenario matched), not_recognized "
    "(heard, but no scenario), llm (maximum mode live answer)",
    ["source", "outcome"],
)
SCENARIO_HITS = Counter("univoice_scenario_hits_total", "Answers per matched scenario", ["scenario"])
UNHEARD = Counter("univoice_unheard_utterances_total", "Utterances STT turned into no text", ["source"])
TRANSFERS = Counter(
    "univoice_operator_transfers_total",
    "Hand-overs to an operator, by trigger (dtmf = pressed 0, intent = asked for one)",
    ["trigger"],
)
OPERATOR_UNAVAILABLE = Counter(
    "univoice_operator_unavailable_total", "Transfers no operator answered (caller came back)"
)
LLM_ANSWERS = Counter(
    "univoice_llm_answers_total",
    "Maximum mode live answers: result=ok, or fallback to the pre-recorded answer",
    ["source", "result"],
)
STT_LANGUAGE = Counter(
    "univoice_stt_language_total", "Language each utterance was transcribed as", ["source", "language"]
)
STT_ROUTE = Counter(
    "univoice_stt_route_total",
    "How the language-routed STT decided: forced (language given), detector (sure "
    "detection), specialist (confident Uzbek fine-tune), compared (several models, "
    "surest won), no_speech",
    ["route", "language"],
)
ERRORS = Counter(
    "univoice_errors_total",
    "Failures by stage (session, utterance, prompt, keepalive)",
    ["stage"],
)

HTTP_REQUESTS = Counter(
    "univoice_http_requests_total", "HTTP requests by route template and status", ["method", "route", "status"]
)
HTTP_SECONDS = Histogram(
    "univoice_http_request_seconds",
    "HTTP request time until the response starts",
    ["method", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


def observe(stage: str, source: str, seconds: float) -> None:
    STAGE_SECONDS.labels(stage, source).observe(max(0.0, seconds))


def count_answer(source: str, outcome: str, scenario_id: str | None = None) -> None:
    ANSWERS.labels(source, outcome).inc()
    if scenario_id:
        SCENARIO_HITS.labels(scenario_id).inc()


class _GpuMemoryCollector:
    """Device memory from NVML, read at scrape time; nothing without an NVIDIA GPU."""

    def __init__(self) -> None:
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception as exc:  # no package, no driver or no GPU
            LOGGER.debug("GPU metrics unavailable: %s", exc)

    def collect(self):
        if self._nvml is None:
            return
        nvml = self._nvml
        used = GaugeMetricFamily("univoice_gpu_memory_used_bytes", "GPU memory in use", labels=["gpu", "name"])
        total = GaugeMetricFamily("univoice_gpu_memory_total_bytes", "GPU memory size", labels=["gpu", "name"])
        try:
            for index in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                name = nvml.nvmlDeviceGetName(handle)
                name = name.decode() if isinstance(name, bytes) else name
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                used.add_metric([str(index), name], memory.used)
                total.add_metric([str(index), name], memory.total)
        except Exception as exc:
            LOGGER.warning("Reading GPU memory failed: %s", exc)
            return
        yield used
        yield total


def _initialise() -> None:
    # Series exist at 0 from the start, so rate() and "no answers" alerts work
    # before the first call instead of seeing no data.
    for source in (PHONE, CONSOLE):
        CALLS.labels(source)
        UNHEARD.labels(source)
        for outcome in ("scenario", "not_recognized", "llm"):
            ANSWERS.labels(source, outcome)
        for result in ("ok", "fallback"):
            LLM_ANSWERS.labels(source, result)
    for reason in ("caller_hangup", "transferred", "done", "silence", "time_limit", "error"):
        CALL_ENDS.labels(reason)
    for trigger in ("dtmf", "intent"):
        TRANSFERS.labels(trigger)
    for stage in ("session", "utterance", "prompt", "keepalive"):
        ERRORS.labels(stage)
    REGISTRY.register(_GpuMemoryCollector())


_initialise()
