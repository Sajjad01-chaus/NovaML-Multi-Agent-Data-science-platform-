"""Structured JSON logging. Every log line carries `run_id` and `agent` when bound."""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(stream=sys.stderr, level=level, format="%(message)s")
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "novaml") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
