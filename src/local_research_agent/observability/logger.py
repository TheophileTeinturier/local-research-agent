"""Structured JSON logging to stderr.

One JSON object per line, always on stderr so stdout stays reserved for the CLI's
answer (and, later, for an MCP stdio transport, where any stray stdout byte would
corrupt the protocol).

Usage:
    logger = logging.getLogger(__name__)
    log_event(logger, "tool_execution", tool="calculator", ok=True, duration_ms=0.2)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

ROOT_LOGGER_NAME = "local_research_agent"


class JsonFormatter(logging.Formatter):
    """One JSON line per record; structured fields come from `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: int | str = "INFO") -> None:
    """Attach a stderr JSON handler to the package logger. Idempotent; called by the CLI only."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    logger.propagate = False


def log_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any
) -> None:
    """Log a named event with structured key/value fields."""
    logger.log(level, event, extra={"fields": fields})
