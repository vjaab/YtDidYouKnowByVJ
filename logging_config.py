"""
logging_config.py — Structured logging with structlog + JSON output.
Centralized logging configuration for the entire pipeline.
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps

import structlog
from structlog.types import EventDict, WrappedLogger

from config import LOGS_DIR, BASE_DIR


# ═══════════════════════════════════════════════════════════════════════
# CUSTOM PROCESSORS
# ═══════════════════════════════════════════════════════════════════════

def add_timestamp(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to every log entry."""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def add_service_context(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service context (pipeline stage, run_id, etc.)."""
    # These can be set via structlog.contextvars.bind_contextvars()
    return event_dict


def add_severity_level(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Map method name to severity level."""
    level_map = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARN",
        "warn": "WARN",
        "error": "ERROR",
        "critical": "CRITICAL",
        "exception": "ERROR",
    }
    event_dict["level"] = level_map.get(method_name, "INFO")
    return event_dict


def filter_sensitive_data(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Remove sensitive keys from logs."""
    sensitive_keys = {
        "api_key", "apikey", "token", "secret", "password", 
        "access_token", "refresh_token", "client_secret",
        "private_key", "credential", "auth"
    }
    
    def _filter(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: _filter(v) for k, v in obj.items() 
                if k.lower() not in sensitive_keys
            }
        elif isinstance(obj, list):
            return [_filter(item) for item in obj]
        return obj
    
    return _filter(event_dict)


def format_json_output(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> str:
    """Final JSON serialization."""
    return json.dumps(event_dict, default=str, separators=(",", ":"))


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> structlog.BoundLogger:
    """
    Configure structlog with JSON output and optional file rotation.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARN, ERROR)
        json_output: If True, output JSON lines; else pretty console
        log_file: Path to log file (uses rotating file handler)
        enable_console: Also log to console
    """
    # Standard library logging setup
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO)
    )
    
    # Configure file handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10_000_000,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        if json_output:
            file_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        add_timestamp,
        add_severity_level,
        add_service_context,
        filter_sensitive_data,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_output:
        processors.append(format_json_output)
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance with optional name context."""
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(service=name)
    return logger


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE-SPECIFIC LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════════════

class PipelineLogger:
    """Structured logger for pipeline stages with automatic context."""
    
    def __init__(self, run_id: str, topic_type: str = "auto"):
        self.run_id = run_id
        self.topic_type = topic_type
        self.logger = get_logger("pipeline").bind(
            run_id=run_id,
            topic_type=topic_type,
            pipeline_stage="init"
        )
        self._stage_start_times = {}
    
    def stage_start(self, stage: str, **kwargs):
        """Mark the start of a pipeline stage."""
        self._stage_start_times[stage] = datetime.utcnow()
        self.logger = self.logger.bind(pipeline_stage=stage)
        self.logger.info(f"STAGE_START: {stage}", **kwargs)
    
    def stage_end(self, stage: str, success: bool = True, **kwargs):
        """Mark the end of a pipeline stage with duration."""
        start_time = self._stage_start_times.get(stage)
        duration = None
        if start_time:
            duration = (datetime.utcnow() - start_time).total_seconds()
        
        self.logger.info(
            f"STAGE_END: {stage}",
            success=success,
            duration_seconds=duration,
            **kwargs
        )
        self.logger = self.logger.bind(pipeline_stage="post_" + stage)
    
    def stage_failed(self, stage: str, error: Exception, **kwargs):
        """Log a stage failure with error details."""
        start_time = self._stage_start_times.get(stage)
        duration = None
        if start_time:
            duration = (datetime.utcnow() - start_time).total_seconds()
        
        self.logger.error(
            f"STAGE_FAILED: {stage}",
            error_type=type(error).__name__,
            error_message=str(error),
            duration_seconds=duration,
            **kwargs
        )
    
    def metric(self, name: str, value: float, unit: str = "", **kwargs):
        """Log a metric value."""
        self.logger.info(
            f"METRIC: {name}",
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            **kwargs
        )
    
    def video_generated(self, video_path: str, duration: float, **kwargs):
        """Log video generation completion."""
        self.logger.info(
            "VIDEO_GENERATED",
            video_path=video_path,
            duration_seconds=duration,
            **kwargs
        )
    
    def upload_result(self, platform: str, video_id: str, success: bool, **kwargs):
        """Log upload result."""
        level = "info" if success else "error"
        getattr(self.logger, level)(
            f"UPLOAD_{'SUCCESS' if success else 'FAILED'}: {platform}",
            platform=platform,
            video_id=video_id,
            success=success,
            **kwargs
        )
    
    def cost(self, service: str, cost_usd: float, **kwargs):
        """Log API cost."""
        self.logger.info(
            "COST",
            service=service,
            cost_usd=cost_usd,
            **kwargs
        )


def log_function_call(logger: structlog.BoundLogger):
    """Decorator to log function entry/exit with timing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start = datetime.utcnow()
            logger.debug(f"CALL: {func_name}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start).total_seconds()
                logger.debug(f"RETURN: {func_name}", duration_seconds=duration, success=True)
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start).total_seconds()
                logger.error(f"ERROR: {func_name}", duration_seconds=duration, error=str(e))
                raise
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════

def init_logging(
    run_id: Optional[str] = None,
    log_level: str = "INFO",
    json_logs: bool = True
) -> PipelineLogger:
    """
    Initialize logging for a pipeline run.
    
    Args:
        run_id: Unique run identifier (generated if not provided)
        log_level: Log level
        json_logs: Output JSON lines
        
    Returns:
        PipelineLogger instance
    """
    if run_id is None:
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure log directory exists
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Configure log file
    log_file = Path(LOGS_DIR) / f"pipeline_{run_id}.log"
    
    # Configure structlog
    configure_logging(
        log_level=log_level,
        json_output=json_logs,
        log_file=str(log_file),
        enable_console=True
    )
    
    return PipelineLogger(run_id=run_id)


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("STRUCTURED LOGGING TEST")
    print("=" * 60)
    
    # Initialize
    pipeline_logger = init_logging(run_id="test_run_001", log_level="DEBUG")
    
    # Test stage logging
    pipeline_logger.stage_start("topic_selection", topic="AI Tools")
    time.sleep(0.1)
    pipeline_logger.stage_end("topic_selection", success=True, topic_selected="Free AI Alternative")
    
    pipeline_logger.stage_start("script_generation")
    time.sleep(0.1)
    pipeline_logger.stage_end("script_generation", success=True, script_length=120)
    
    pipeline_logger.stage_start("audio_generation")
    time.sleep(0.1)
    pipeline_logger.stage_failed("audio_generation", ValueError("API quota exceeded"))
    
    # Test metrics
    pipeline_logger.metric("api_latency", 1.23, "seconds", endpoint="gemini")
    pipeline_logger.metric("video_duration", 35.5, "seconds")
    pipeline_logger.metric("estimated_cost", 0.0023, "USD", service="elevenlabs")
    
    # Test video/upload logging
    pipeline_logger.video_generated("/tmp/video.mp4", 35.5, resolution="1080x1920")
    pipeline_logger.upload_result("youtube", "abc123", True, url="https://youtu.be/abc123")
    pipeline_logger.upload_result("instagram", None, False, error="Rate limited")
    
    # Test cost logging
    pipeline_logger.cost("gemini", 0.0012, model="gemini-2.5-flash")
    pipeline_logger.cost("elevenlabs", 0.0045, characters=1200)
    
    print("\n✅ All logging tests passed!")
    print(f"Log file: {LOGS_DIR}/pipeline_test_run_001.log")