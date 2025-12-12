"""Centralized Logfire configuration.

Goals:
- Ensure `service.name` is set (avoid OpenTelemetry's `unknown_service`).
- Make configuration idempotent and safe when LOGFIRE_TOKEN is missing.
- Bridge stdlib `logging` into Logfire so existing logs become searchable.
- Enable useful instrumentation where available.

Environment variables (optional):
- LOGFIRE_TOKEN
- LOGFIRE_SERVICE_NAME / OTEL_SERVICE_NAME
- LOGFIRE_ENVIRONMENT / ENVIRONMENT
- SERVICE_VERSION / GIT_SHA / COMMIT_SHA
- LOGFIRE_MIN_LEVEL (e.g. INFO, DEBUG)
- LOGFIRE_INSTRUMENT (0/1)
- LOGFIRE_LOGGING_HANDLER (0/1)
- LOGFIRE_INSPECT_ARGUMENTS (0/1)
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator, Optional

_LOGGER = logging.getLogger(__name__)
_CONFIGURED = False


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _guess_service_version() -> str | None:
    for key in ("SERVICE_VERSION", "GIT_SHA", "COMMIT_SHA", "VERSION"):
        value = os.environ.get(key)
        if value:
            return value

    # Best-effort git SHA when running from a checkout.
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if not (repo_root / ".git").exists():
            return None
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        sha = completed.stdout.strip()
        return sha or None
    except Exception:
        return None


def _parse_min_level(value: str | None) -> int | None:
    if not value:
        return None
    value = value.strip().upper()
    if value.isdigit():
        return int(value)
    return getattr(logging, value, None)


def configure_logfire(
    *,
    service_name: str | None = None,
    environment: str | None = None,
    service_version: str | None = None,
) -> Any | None:
    """Configure Logfire once per process.

    Returns the imported `logfire` module instance, or `None` if logfire isn't
    installed.
    """

    global _CONFIGURED

    try:  # pragma: no cover
        import logfire
    except Exception:  # pragma: no cover
        return None

    if _CONFIGURED:
        return logfire

    token = os.environ.get("LOGFIRE_TOKEN")

    resolved_service_name = (
        service_name
        or os.environ.get("LOGFIRE_SERVICE_NAME")
        or os.environ.get("OTEL_SERVICE_NAME")
        or "rag-evaluation"
    )
    resolved_environment = (
        environment
        or os.environ.get("LOGFIRE_ENVIRONMENT")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("RAG_ENV")
        or None
    )
    resolved_service_version = service_version or _guess_service_version()

    min_level = _parse_min_level(os.environ.get("LOGFIRE_MIN_LEVEL"))
    inspect_arguments = _truthy(os.environ.get("LOGFIRE_INSPECT_ARGUMENTS")) if os.environ.get("LOGFIRE_INSPECT_ARGUMENTS") else None

    try:
        logfire.configure(
            token=token,
            send_to_logfire="if-token-present",
            service_name=resolved_service_name,
            service_version=resolved_service_version,
            environment=resolved_environment,
            min_level=min_level,
            inspect_arguments=inspect_arguments,
        )
    except Exception as exc:  # pragma: no cover
        _LOGGER.debug("Logfire configure failed; continuing without telemetry: %s", exc)
        _CONFIGURED = True
        return logfire

    # Bridge stdlib logging -> Logfire (optional).
    if not os.environ.get("LOGFIRE_LOGGING_HANDLER") or _truthy(os.environ.get("LOGFIRE_LOGGING_HANDLER")):
        try:
            root = logging.getLogger()
            if not any(getattr(h, "__class__", None).__name__ == "LogfireLoggingHandler" for h in root.handlers):
                handler = logfire.LogfireLoggingHandler()
                handler.setLevel(min_level or logging.INFO)
                root.addHandler(handler)
        except Exception as exc:  # pragma: no cover
            _LOGGER.debug("Unable to attach Logfire logging handler: %s", exc)

    # Enable a few high-value instrumentations where available (optional).
    if not os.environ.get("LOGFIRE_INSTRUMENT") or _truthy(os.environ.get("LOGFIRE_INSTRUMENT")):
        for instrumenter in (
            "instrument_requests",
            "instrument_httpx",
            "instrument_sqlalchemy",
            "instrument_sqlite3",
            "instrument_openai",
            "instrument_pydantic_ai",
            "instrument_system_metrics",
        ):
            fn = getattr(logfire, instrumenter, None)
            if not fn:
                continue
            try:
                fn()
            except Exception:
                # Instrumentation is best-effort; never break the app.
                pass

    _CONFIGURED = True
    return logfire


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Create a Logfire span when available; otherwise a no-op context manager."""

    logfire = configure_logfire()
    if not logfire:
        yield
        return

    try:
        with logfire.span(name, **attributes):
            yield
    except Exception:
        # If something goes wrong (e.g. misconfiguration), don't block execution.
        yield


def info(event: str, **attributes: Any) -> None:
    logfire = configure_logfire()
    if not logfire:
        return
    try:
        logfire.info(event, **attributes)
    except Exception:
        return


def warning(event: str, **attributes: Any) -> None:
    logfire = configure_logfire()
    if not logfire:
        return
    try:
        logfire.warning(event, **attributes)
    except Exception:
        return
