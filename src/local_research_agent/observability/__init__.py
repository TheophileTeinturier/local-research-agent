"""Observability: structured logs (stderr only). Tracing/metrics export will live here later."""

from .logger import JsonFormatter, configure_logging, log_event

__all__ = ["JsonFormatter", "configure_logging", "log_event"]
