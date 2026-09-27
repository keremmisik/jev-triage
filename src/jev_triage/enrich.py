"""Context enrichment from Loki.

An Alertmanager alert only carries labels and annotations. Operators judge an
alert by looking at the logs around it, so we fetch the last few lines from the
same host/service and add them to the state. Failures here are never fatal:
triage proceeds without context.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta

import httpx

from . import metrics
from .models import TriageInput
from .preprocess import redact

log = logging.getLogger(__name__)

# Only label values matching this are interpolated into LogQL (prevents injection).
_SAFE_LABEL_VALUE = re.compile(r"^[\w.:/-]{1,200}$")


class LokiEnricher:
    def __init__(
        self,
        base_url: str,
        *,
        lookback_s: int,
        max_lines: int,
        timeout_s: float,
        redact_pii: bool,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/loki/api/v1/query_range"
        self._lookback = timedelta(seconds=lookback_s)
        self._max_lines = max_lines
        self._redact = redact_pii
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @staticmethod
    def build_selector(event: TriageInput) -> str | None:
        """LogQL stream selector for the event's source, or None if nothing is usable."""
        matchers = []
        for label, value in (("host", event.host), ("service", event.service)):
            if value:
                # Alert `instance` labels look like "host:9100"; Loki's host label won't.
                value = value.split(":")[0] if label == "host" else value
                if _SAFE_LABEL_VALUE.match(value):
                    matchers.append(f'{label}="{value}"')
        return "{" + ", ".join(matchers) + "}" if matchers else None

    async def enrich(self, event: TriageInput) -> TriageInput:
        if self._max_lines == 0 or event.kind != "alert":
            return event
        selector = self.build_selector(event)
        if selector is None:
            return event
        end = event.timestamp
        start = end - self._lookback
        params = {
            "query": selector,
            "start": str(int(start.timestamp() * 1e9)),
            "end": str(int(end.timestamp() * 1e9) + 1),
            "limit": str(self._max_lines),
            "direction": "backward",
        }
        try:
            resp = await self._client.get(self._url, params=params)
            resp.raise_for_status()
            lines = self._parse(resp.json())
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            metrics.ENRICH_ERRORS.inc()
            log.warning("loki enrichment failed: %s", exc, extra={"selector": selector})
            return event
        if self._redact:
            lines = [redact(line) for line in lines]
        return event.model_copy(update={"context_lines": lines})

    def _parse(self, body: dict) -> list[str]:  # type: ignore[type-arg]
        entries: list[tuple[int, str]] = []
        for stream in body["data"]["result"]:
            for ts, line in stream["values"]:
                entries.append((int(ts), line))
        entries.sort()  # oldest first reads naturally
        return [line for _, line in entries[-self._max_lines :]]

    async def aclose(self) -> None:
        await self._client.aclose()
