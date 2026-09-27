"""Where decisions go after routing.

``LogSink`` always runs (the audit trail). ``WebhookSink`` posts actionable
decisions to any JSON webhook (Slack/Teams/Mattermost via a small adapter,
PagerDuty Events API, a ticketing system...). Delivery failures are logged and
counted but never fail the triage request itself.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from . import metrics
from .models import Route, TriageDecision

log = logging.getLogger("jev_triage.decisions")


class Sink(Protocol):
    name: str

    async def send(self, decision: TriageDecision) -> None: ...

    async def aclose(self) -> None: ...


class LogSink:
    name = "log"

    async def send(self, decision: TriageDecision) -> None:
        level = logging.WARNING if decision.route is Route.PAGE else logging.INFO
        log.log(level, "triage decision", extra={"decision": decision.summary()})

    async def aclose(self) -> None:
        return None


class WebhookSink:
    name = "webhook"
    ROUTES = frozenset({Route.PAGE, Route.TICKET, Route.HUMAN_REVIEW})

    def __init__(self, url: str, *, timeout_s: float, client: httpx.AsyncClient | None = None):
        self._url = url
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def send(self, decision: TriageDecision) -> None:
        if decision.route not in self.ROUTES:
            return
        try:
            resp = await self._client.post(self._url, json=decision.summary())
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            metrics.SINK_ERRORS.labels(self.name).inc()
            log.error("webhook delivery failed: %s", exc, extra={"decision_id": decision.id})

    async def aclose(self) -> None:
        await self._client.aclose()
