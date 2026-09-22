"""
Unit tests: src/utils/logger.py.
"""

import logging

from src.utils.logger import setup_logger


def test_setup_logger_returns_the_named_logger():
    logger = setup_logger()

    assert isinstance(logger, logging.Logger)
    assert logger.name == "olist_mlops"


def test_setup_logger_is_idempotent_and_never_duplicates_handlers():
    """
    setup_logger()'s `if not logger.handlers:` guard exists specifically so
    calling it more than once (module re-import under pytest, uvicorn
    --reload, repeated setup_logger() calls from different modules) doesn't
    silently double- or triple-log every line. Prove the guard actually
    holds instead of trusting the comment.
    """

    first = setup_logger()
    handler_count = len(first.handlers)

    second = setup_logger()

    assert second is first
    assert len(second.handlers) == handler_count


def test_logger_level_matches_config():
    from src.utils.config_loader import load_config

    config = load_config()
    expected_level = getattr(
        logging, str(config["logging"]["level"]).upper(), logging.INFO
    )

    logger = setup_logger()

    assert logger.level == expected_level
