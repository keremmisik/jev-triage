"""HTTP API.

Endpoints
---------
POST /v1/alertmanager   Alertmanager webhook receiver
POST /v1/logs           Push a batch of log lines
GET  /v1/decisions      Recent decisions (newest first)
GET  /v1/decisions/{id} One decision with full probabilities and context
GET  /healthz           Liveness (process is up)
GET  /readyz            Readiness (reports degraded mode when the breaker is open)
GET  /metrics           Prometheus metrics
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .config import Settings, get_settings
from .models import AlertmanagerWebhook, LogBatch, TriageDecision
from .service import BatchResult, TriageService

log = logging.getLogger(__name__)


def _batch_response(result: BatchResult) -> dict[str, Any]:
    return {
        "processed": len(result.decisions),
        "deduplicated": len(result.duplicates),
        "skipped": result.skipped,
        "failed": result.failed,
        "decisions": [d.summary() for d in result.decisions],
        "duplicate_of": result.duplicates,
    }


def create_app(settings: Settings | None = None, service: TriageService | None = None) -> FastAPI:
    """App factory. Tests inject a pre-built ``service``; production builds one from env."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = service or TriageService.from_settings(settings)
        app.state.service = svc
        log.info(
            "jev-triage started",
            extra={"backend": settings.backend.value, "model": settings.typesafe_model},
        )
        try:
            yield
        finally:
            await svc.aclose()
            log.info("jev-triage stopped")

    app = FastAPI(title="jev-triage", version=__version__, lifespan=lifespan)

    def svc(request: Request) -> TriageService:
        return request.app.state.service  # type: ignore[no-any-return]

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Last line of defence: never leak stack traces to webhook callers.
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.post("/v1/alertmanager")
    async def alertmanager(payload: AlertmanagerWebhook, request: Request) -> dict[str, Any]:
        if len(payload.alerts) > settings.max_batch_size:
            raise HTTPException(413, "too many alerts")
        return _batch_response(await svc(request).triage_alertmanager(payload))

    @app.post("/v1/logs")
    async def logs(batch: LogBatch, request: Request) -> dict[str, Any]:
        if len(batch.records) > settings.max_batch_size:
            raise HTTPException(
                413,
                f"max {settings.max_batch_size} records per request",
            )
        return _batch_response(await svc(request).triage_logs(batch.records))

    @app.get("/v1/decisions")
    async def decisions(
        request: Request, limit: int = Query(50, ge=1, le=500)
    ) -> list[dict[str, Any]]:
        return [d.summary() for d in svc(request).recent(limit)]

    @app.get("/v1/decisions/{decision_id}", response_model=TriageDecision)
    async def decision(decision_id: str, request: Request) -> TriageDecision:
        d = svc(request).get(decision_id)
        if d is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "decision not found or expired")
        return d

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        # Still "ready" while degraded: the heuristic fallback keeps triage working.
        degraded = svc(request).breaker_open
        return {"status": "degraded" if degraded else "ok", "backend": settings.backend.value}

    @app.get("/metrics")
    async def prom_metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
