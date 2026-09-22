import logging
from pathlib import Path

from src.utils.config_loader import load_config


def setup_logger():
    """
    Config-driven: level, log directory, and log filename all come from
    config.yaml's `logging` block instead of being hardcoded, so changing
    verbosity for local debugging vs. the container never means editing code.
    """

    config = load_config()
    log_cfg = config.get("logging", {})

    log_dir = Path(log_cfg.get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)

    log_file = log_cfg.get("log_file", "app.log")
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("olist_mlops")
    logger.setLevel(level)

    if not logger.handlers:

        file_handler = logging.FileHandler(log_dir / log_file)

        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"
        )

        file_handler.setFormatter(formatter)

        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.addHandler(console_handler)

    return logger


# Module-level singleton so `from src.utils.logger import logger` works
# everywhere (routes, pipeline, tests) without every caller remembering to
# call setup_logger() first.
logger = setup_logger()
