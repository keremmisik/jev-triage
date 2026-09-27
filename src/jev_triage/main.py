"""Entry point: ``jev-triage`` (or ``python -m jev_triage``)."""

from __future__ import annotations

import logging
import sys

import uvicorn
from pydantic import ValidationError

from .config import get_settings
from .logging_config import configure_logging


def run() -> None:
    try:
        settings = get_settings()
    except ValidationError as exc:
        # Fail fast with a readable message instead of a pydantic traceback.
        print(f"configuration error:\n{exc}", file=sys.stderr)
        sys.exit(2)

    configure_logging(settings.log_level, settings.log_json)
    from .api import create_app  # imported after logging is configured

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # keep our JSON logging
        access_log=False,
        proxy_headers=True,
    )
    logging.shutdown()


if __name__ == "__main__":
    run()
