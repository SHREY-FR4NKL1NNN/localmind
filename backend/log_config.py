"""Centralised structured logging for LocalMind.

All backend modules obtain their logger via ``get_logger`` so output shares one
format and one level. Logs go to stdout (captured by systemd/Docker/CI) with a
timestamped, levelled, named line. Set ``LOCALMIND_DEBUG=1`` to drop the level
to DEBUG (e.g. to see individual token chunks and internal state changes).
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_ROOT_NAME = "localmind"
_configured = False


def _configure_root() -> None:
    """Attach a single stdout handler to the ``localmind`` root logger once."""
    global _configured
    if _configured:
        return
    level = logging.DEBUG if os.environ.get("LOCALMIND_DEBUG") == "1" else logging.INFO
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
    # Don't double-log through the Python root logger.
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``localmind`` namespace.

    ``name`` is typically the module name (e.g. ``"router"``), yielding a logger
    named ``localmind.router`` that inherits the shared handler and level.
    """
    _configure_root()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
