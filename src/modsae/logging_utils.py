"""Timestamped logging to the console and to logs/<name>_<timestamp>.log.

Global context: every phase calls `get_logger` once, so each run leaves a complete,
timestamped text log next to the machine-readable results in results/.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from modsae.config import Paths

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stdout and a new timestamped file in logs/."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured in this process
        return logger
    logger.setLevel(logging.INFO)
    logs = Paths().make().logs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(logs / f"{name}_{stamp}.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
