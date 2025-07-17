from datetime import datetime
import logging
import pathlib
from typing import Optional


class TahoeLogger:
    _instance: Optional["TahoeLogger"] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls) -> "TahoeLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self, log_file: str = "tahoe_agent.log") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_directory = pathlib.Path("logs")
        log_directory.mkdir(exist_ok=True)

        log_path = log_directory / f"{timestamp}_{log_file}"

        self._logger = logging.getLogger("tahoe_agent")
        self._logger.setLevel(logging.INFO)

        if not self._logger.handlers:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_formatter)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)

            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._setup_logger()
        assert self._logger is not None
        return self._logger

    def info(self, message: str) -> None:
        self.logger.info(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)


def get_logger() -> logging.Logger:
    return TahoeLogger().logger


def setup_logger(log_file: str = "tahoe_agent.log") -> logging.Logger:
    tahoe_logger = TahoeLogger()
    tahoe_logger._setup_logger(log_file)
    return tahoe_logger.logger


def log_info(message: str) -> None:
    TahoeLogger().info(message)


def log_error(message: str) -> None:
    TahoeLogger().error(message)


def log_warning(message: str) -> None:
    TahoeLogger().warning(message)


def log_debug(message: str) -> None:
    TahoeLogger().debug(message)


def log_critical(message: str) -> None:
    TahoeLogger().critical(message)
