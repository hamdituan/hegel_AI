"""Centralized logging configuration for Hegel AI."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.RESET) if self.use_color else ''
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    use_color: bool = True,
    module_levels: Optional[dict] = None,
) -> logging.Logger:
    """Set up centralized logging."""
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hegel_ai")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            use_color=use_color
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if file_output:
        log_file = log_dir / "hegel_ai.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8',
            delay=True
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        error_file = log_dir / "hegel_ai_errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8',
            delay=True
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)

    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(module_level)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for module."""
    return logging.getLogger(f"hegel_ai.{name}")


_logger: Optional[logging.Logger] = None


def get_main_logger() -> logging.Logger:
    """Get main logger."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def log_debug(msg: str, **kwargs) -> None:
    get_main_logger().debug(msg, **kwargs)


def log_info(msg: str, **kwargs) -> None:
    get_main_logger().info(msg, **kwargs)


def log_warning(msg: str, **kwargs) -> None:
    get_main_logger().warning(msg, **kwargs)


def log_error(msg: str, **kwargs) -> None:
    get_main_logger().error(msg, **kwargs)


def log_critical(msg: str, **kwargs) -> None:
    get_main_logger().critical(msg, **kwargs)


def log_exception(msg: str, **kwargs) -> None:
    get_main_logger().exception(msg, **kwargs)
