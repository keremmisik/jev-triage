"""Send one real request to Jev and print the parsed answer.

Reads TYPESAFE_API_KEY from .env (run from the repository root):

    python scripts/smoke_jev.py
"""

import asyncio
from datetime import datetime, timezone

from jev_triage.classifier import ClassifierError, JevClassifier
from jev_triage.config import get_settings
from jev_triage.models import TriageInput


async def main() -> None:
    settings = get_settings()
    if settings.typesafe_api_key is None:
        raise SystemExit("TYPESAFE_API_KEY is not set (add it to .env)")
    clf = JevClassifier(
        api_key=settings.typesafe_api_key.get_secret_value(), model=settings.typesafe_model
    )
    event = TriageInput(
        kind="log",
        title="OOM",
        message="kernel: Out of memory: Killed process 4812 (java) total-vm:8123456kB",
        host="web1",
        service="kernel",
        timestamp=datetime.now(timezone.utc),
    )
    try:
        c = await clf.classify(event)
    except ClassifierError as exc:
        print(f"ERROR ({exc.reason}): {exc}")
        if exc.__cause__ is not None:
            print("detail:", getattr(exc.__cause__, "body", exc.__cause__))
        return
    finally:
        await clf.aclose()

    print(f"model     : {c.model_version} | {c.latency_ms:.0f} ms | {c.input_tokens} tokens")
    print(f"severity  : {c.severity.label} ({c.severity.confidence:.2f})")
    print(f"subsystem : {c.subsystem.label} ({c.subsystem.confidence:.2f})")
    print("signals   :", {k: round(v, 2) for k, v in c.nouls.items()})


if __name__ == "__main__":
    asyncio.run(main())
