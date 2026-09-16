"""
Structured Logging
==================
Replaces bare print() calls throughout the pipeline with a proper
logging setup: file + console handlers, color output, and structured
JSON log records for downstream analysis.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded in %.2fs", elapsed)
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Optional


# ── ANSI colour codes ─────────────────────────────────────────────
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_RED     = "\033[91m"
_YELLOW  = "\033[93m"
_GREEN   = "\033[92m"
_CYAN    = "\033[96m"
_BLUE    = "\033[94m"
_GREY    = "\033[90m"

_LEVEL_COLOURS = {
    logging.DEBUG:    _GREY   + "DEBUG"   + _RESET,
    logging.INFO:     _GREEN  + "INFO "   + _RESET,
    logging.WARNING:  _YELLOW + "WARN "   + _RESET,
    logging.ERROR:    _RED    + "ERROR"   + _RESET,
    logging.CRITICAL: _BOLD + _RED + "CRIT " + _RESET,
}


class _ColourFormatter(logging.Formatter):
    """Console formatter with ANSI colour-coded log levels."""

    FMT = "{time}  {level}  {name}  {msg}"

    def format(self, record: logging.LogRecord) -> str:
        level_str = _LEVEL_COLOURS.get(record.levelno, record.levelname)
        time_str  = _GREY + datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3] + _RESET
        name_str  = _CYAN + f"{record.name:<28}" + _RESET
        msg_str   = record.getMessage()

        if record.exc_info:
            msg_str += "\n" + self.formatException(record.exc_info)

        return self.FMT.format(time=time_str, level=level_str, name=name_str, msg=msg_str)


class _JSONFormatter(logging.Formatter):
    """File formatter that emits one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "module":  record.module,
            "lineno":  record.lineno,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# ── Public API ────────────────────────────────────────────────────

def setup_logging(
    log_dir:      str  = "outputs/logs",
    log_level:    str  = "INFO",
    json_file:    bool = True,
    console:      bool = True,
) -> None:
    """
    Configure root logger with console + rotating JSON file handlers.

    Call once at application entry point (detect_image.py / detect_video.py).

    Args:
        log_dir:   Directory to write log files. Created if absent.
        log_level: Minimum log level string ("DEBUG", "INFO", "WARNING", …).
        json_file: Whether to write a JSON-per-line log file.
        console:   Whether to add a coloured console (stderr) handler.
    """
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(_ColourFormatter())
        root.addHandler(ch)

    if json_file:
        log_path = os.path.join(
            log_dir,
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(_JSONFormatter())
        root.addHandler(fh)

    root.info("Logging initialised | level=%s | log_dir=%s", log_level, log_dir)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.  Call setup_logging() first.

    Args:
        name: Logger name (typically __name__).

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name or "perception")
