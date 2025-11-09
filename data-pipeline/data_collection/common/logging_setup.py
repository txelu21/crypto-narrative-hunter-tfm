import json
import logging
import os
import hashlib
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, Any, Optional
from .config import settings


class StructuredJsonFormatter(logging.Formatter):
    """JSON formatter with required fields for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        # Extract custom fields from record if they exist
        component = getattr(record, 'component', record.name)
        operation = getattr(record, 'operation', record.funcName or 'unknown')
        params_hash = getattr(record, 'params_hash', '')
        status = getattr(record, 'status', 'info')
        duration_ms = getattr(record, 'duration_ms', 0)
        error = getattr(record, 'error', '')

        # Build structured log entry
        data = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "component": component,
            "operation": operation,
            "params_hash": params_hash,
            "status": status,
            "duration_ms": duration_ms,
            "error": error,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            data["error"] = self.formatException(record.exc_info)
            data["status"] = "error"

        # Remove empty fields for cleaner logs
        data = {k: v for k, v in data.items() if v != '' and v is not None}

        return json.dumps(data)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with structured logging support"""
    logger = logging.getLogger(name)
    return StructuredLogger(logger)


class StructuredLogger:
    """Wrapper around logger that adds structured logging methods"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def log_operation(self,
                      operation: str,
                      params: Optional[Dict[str, Any]] = None,
                      status: str = "started",
                      duration_ms: int = 0,
                      error: str = "",
                      message: str = "") -> None:
        """Log an operation with structured fields"""
        params_hash = ""
        if params:
            # Create hash of params for tracking without exposing sensitive data
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        extra = {
            'component': self._logger.name,
            'operation': operation,
            'params_hash': params_hash,
            'status': status,
            'duration_ms': duration_ms,
            'error': error
        }

        if error:
            self._logger.error(message or f"Operation {operation} failed", extra=extra)
        elif status == "completed":
            self._logger.info(message or f"Operation {operation} completed", extra=extra)
        else:
            self._logger.info(message or f"Operation {operation} {status}", extra=extra)

    def __getattr__(self, name):
        """Delegate all other methods to the underlying logger"""
        return getattr(self._logger, name)


def setup_logging() -> None:
    """Configure logging with JSON format and daily rotation"""
    os.makedirs(settings.log_dir, exist_ok=True)

    # Get root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Remove any existing handlers
    root.handlers = []

    # Console handler with JSON format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(root.level)
    console_handler.setFormatter(StructuredJsonFormatter())
    root.addHandler(console_handler)

    # Daily rotating file handler
    log_filename = os.path.join(
        settings.log_dir,
        f"collection_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when='midnight',
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.suffix = "%Y%m%d.log"
    file_handler.setLevel(root.level)
    file_handler.setFormatter(StructuredJsonFormatter())
    root.addHandler(file_handler)

    # Add audit logger for data filtering decisions
    audit_logger = logging.getLogger('audit')
    audit_file = os.path.join(settings.log_dir, 'audit.log')
    audit_handler = logging.FileHandler(audit_file, encoding='utf-8')
    audit_handler.setFormatter(StructuredJsonFormatter())
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Don't propagate to root logger


def log_summary(component: str, shard: str, records_processed: int, duration_seconds: float) -> None:
    """Log a periodic summary for shard processing"""
    logger = get_logger(component)
    logger.log_operation(
        operation="shard_summary",
        params={"shard": shard},
        status="completed",
        duration_ms=int(duration_seconds * 1000),
        message=f"Processed {records_processed} records in shard {shard}"
    )
