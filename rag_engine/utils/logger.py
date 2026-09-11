"""
Logger utility — Convenience re-export of the logging factory.

Usage:
    from rag_engine.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from rag_engine.config.logging_config import get_logger

__all__ = ["get_logger"]
