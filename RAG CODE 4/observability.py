"""
observability.py
----------------
Phase 3 — Module 5: Observability

Three pieces:
  1. setup_langsmith()  — turns on LangSmith auto-tracing (env-var based)
  2. StructuredLogger   — JSON logs with correlation IDs
  3. MetricsCollector   — simple in-memory accumulator for latency/cost/tokens

For tracing custom functions, use @traceable from langsmith directly:
    from langsmith import traceable

    @traceable(name="my_function")
    def my_function(...): ...

That's it — the function will show up in the LangSmith UI with inputs,
outputs, and duration auto-captured.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any


# ========================================================================
# 1. LangSmith auto-tracing setup
# ========================================================================

def setup_langsmith(project_name: str = "phase3-rag") -> bool:
    """Enable LangSmith tracing for all LangChain calls.

    Sets the environment variables LangChain reads at call time.
    Returns True if LangSmith is configured (API key present), False otherwise.

    Once enabled, these are automatically traced:
      - Every ChatOpenAI / AzureChatOpenAI .invoke() / .stream()
      - Every AzureOpenAIEmbeddings .embed_query() / .embed_documents()
      - Every Chroma .similarity_search() / .from_documents()
      - Any function decorated with @traceable (from langsmith)

    Traces appear at https://smith.langchain.com under the named project.
    """

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        # Fail open: tracing off, pipeline still works.
        # Good for local dev without a LangSmith account.
        print("LANGSMITH_API_KEY not set — tracing disabled. Pipeline will still run.")
        return False

    # These are the three env vars LangChain reads at call time:
    #   LANGSMITH_TRACING — flag to enable
    #   LANGSMITH_API_KEY — auth
    #   LANGSMITH_PROJECT — groups traces in the UI
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name

    print(f"LangSmith tracing enabled — project: {project_name}")
    return True


# ========================================================================
# 2. Structured logging with correlation IDs
# ========================================================================

class StructuredLogger:
    """JSON logger with a correlation ID per query.

    Why JSON? Because Phase 7 (evaluation) will parse these logs to match
    up pipeline runs with their metrics. Plain text logs are unparseable.

    Why correlation IDs? So you can follow ONE query through all its log lines
    when multiple queries run concurrently (or just in rapid succession).

    Usage:
        log = StructuredLogger("pipeline")
        with log.context(query="How does attention work?"):
            log.info("stage_started", stage="retrieval")
            ...
    """

    def __init__(self, name: str):
        # Standard Python logger underneath; we just format messages as JSON.
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            # Just the message — we're encoding structure inside it as JSON.
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

        # Context fields that get attached to every log line within a `with` block.
        # E.g. correlation_id, query — things you want on every related log.
        self._context: dict[str, Any] = {}

    @contextmanager
    def context(self, **fields):
        """Set context fields for the duration of a `with` block.

        Usage:
            with log.context(query="...", correlation_id="abc123"):
                log.info(...)   # will include query and correlation_id
        """

        # Generate a correlation_id if none provided — one per context block.
        if "correlation_id" not in fields:
            fields["correlation_id"] = str(uuid.uuid4())[:8]  # short, readable

        # Save the outer context so nested `with` blocks work (we restore on exit).
        previous = dict(self._context)
        self._context.update(fields)
        try:
            yield self._context["correlation_id"]
        finally:
            self._context = previous

    def info(self, event: str, **fields):
        """Log an event as a JSON line.

        Args:
            event: short event name, e.g. "stage_started", "retrieval_complete"
            **fields: any additional structured data (duration_ms, count, etc.)
        """
        payload = {
            "timestamp": time.time(),
            "level": "INFO",
            "event": event,
            **self._context,   # correlation_id, query, etc.
            **fields,          # event-specific data
        }
        # json.dumps with default=str handles non-serializable types gracefully
        # (e.g., if someone passes a ParentChunk object, it becomes a string
        # rather than raising TypeError).
        self._logger.info(json.dumps(payload, default=str))

    def warning(self, event: str, **fields):
        payload = {
            "timestamp": time.time(),
            "level": "WARNING",
            "event": event,
            **self._context,
            **fields,
        }
        self._logger.warning(json.dumps(payload, default=str))


# ========================================================================
# 3. Metrics collector — latency + cost + token counts per stage
# ========================================================================

@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage (e.g., 'hyde', 'retrieval', 'rerank')."""

    name: str
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    custom: dict = field(default_factory=dict)  # stage-specific: num_docs, top_score, etc.


@dataclass
class QueryMetrics:
    """Aggregate metrics for one user query through the pipeline."""

    query: str
    correlation_id: str
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    stages: list[StageMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for logging or writing to a file."""
        return asdict(self)


class MetricsCollector:
    """Collects per-stage timings for a single query.

    Usage:
        metrics = MetricsCollector(query="...", correlation_id="...")
        with metrics.stage("hyde") as stage:
            result = hyde.transform(query)
            stage.custom["num_queries_out"] = len(result)
        # ... more stages ...
        print(metrics.finalize().to_dict())
    """

    def __init__(self, query: str, correlation_id: str):
        self.metrics = QueryMetrics(query=query, correlation_id=correlation_id)
        self._pipeline_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        """Time a stage and collect metrics via the yielded StageMetrics object.

        The caller can attach custom fields (num_docs_retrieved, top_score, etc.)
        by mutating stage.custom inside the `with` block.
        """
        stage = StageMetrics(name=name, duration_ms=0.0)
        start = time.perf_counter()
        try:
            yield stage
        finally:
            stage.duration_ms = (time.perf_counter() - start) * 1000
            self.metrics.stages.append(stage)

    def finalize(self) -> QueryMetrics:
        """Call at the end of the pipeline to compute totals."""
        self.metrics.total_duration_ms = (time.perf_counter() - self._pipeline_start) * 1000
        self.metrics.total_cost_usd = sum(s.cost_usd for s in self.metrics.stages)
        return self.metrics


# ========================================================================
# Demo — simulate a pipeline run with all three observability pieces
# ========================================================================

if __name__ == "__main__":
    # Piece 1: turn on LangSmith (does nothing here since we're not making LLM calls,
    # but shows how you'd configure it)
    setup_langsmith(project_name="phase3-demo")

    # Piece 2 + 3: combine structured logging and metrics for a simulated query
    log = StructuredLogger("pipeline")
    query = "How does attention work?"

    with log.context(query=query) as corr_id:
        metrics = MetricsCollector(query=query, correlation_id=corr_id)

        log.info("pipeline_started")

        # Simulate HyDE stage
        with metrics.stage("hyde") as s:
            time.sleep(0.05)  # pretend we called an LLM
            s.tokens_in = 95
            s.tokens_out = 120
            s.cost_usd = 0.0003
            s.custom["num_queries_out"] = 1
            log.info("stage_complete", stage="hyde", tokens_in=s.tokens_in)

        # Simulate retrieval stage
        with metrics.stage("retrieval") as s:
            time.sleep(0.03)
            s.custom["num_queries_in"] = 5
            s.custom["num_parents_out"] = 15
            log.info("stage_complete", stage="retrieval", num_parents=s.custom["num_parents_out"])

        # Simulate reranking stage
        with metrics.stage("rerank") as s:
            time.sleep(0.02)
            s.custom["pool_size"] = 15
            s.custom["top_score"] = 0.94
            log.info("stage_complete", stage="rerank", top_score=s.custom["top_score"])

        # Final metrics
        final = metrics.finalize()
        log.info(
            "pipeline_complete",
            total_ms=round(final.total_duration_ms, 1),
            total_cost=round(final.total_cost_usd, 4),
        )

        # Print the full metrics object
        print("\n--- Final metrics ---")
        print(json.dumps(final.to_dict(), indent=2, default=str))