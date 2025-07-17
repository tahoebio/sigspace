"""Centralized logging configuration for Tahoe Agent.

This module provides a shared logger instance that can be used across
all modules in the tahoe_agent system.
"""

import logging
import pathlib
from datetime import datetime
from typing import Optional


class TahoeLogger:
    """Centralized logger for the Tahoe Agent system."""
    
    _instance: Optional['TahoeLogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls) -> 'TahoeLogger':
        """Ensure singleton pattern for the logger."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the logger if not already initialized."""
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self, log_file: str = "tahoe_agent.log") -> None:
        """Set up the centralized logger."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory
        log_directory = pathlib.Path("logs")
        log_directory.mkdir(exist_ok=True)
        
        log_path = log_directory / f"{timestamp}_{log_file}"
        
        # Get or create logger
        self._logger = logging.getLogger("tahoe_agent")
        self._logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self._logger.handlers:
            # Create formatters
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            
            # File handler
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_formatter)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)
            
            # Add handlers
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)
    
    @property
    def logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        if self._logger is None:
            self._setup_logger()
        return self._logger
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)


def get_logger() -> logging.Logger:
    """Get the shared logger instance.
    
    Returns:
        Configured logger instance that can be used across the system.
    """
    return TahoeLogger().logger


def setup_logger(log_file: str = "tahoe_agent.log") -> logging.Logger:
    """Set up and return the shared logger instance.
    
    This function maintains backward compatibility with existing code.
    
    Args:
        log_file: Name of the log file (timestamp will be prepended)
        
    Returns:
        Configured logger instance
    """
    tahoe_logger = TahoeLogger()
    tahoe_logger._setup_logger(log_file)
    return tahoe_logger.logger


# Convenience functions for direct logging
def log_info(message: str) -> None:
    """Log info message using the shared logger."""
    TahoeLogger().info(message)


def log_error(message: str) -> None:
    """Log error message using the shared logger."""
    TahoeLogger().error(message)


def log_warning(message: str) -> None:
    """Log warning message using the shared logger."""
    TahoeLogger().warning(message)


def log_debug(message: str) -> None:
    """Log debug message using the shared logger."""
    TahoeLogger().debug(message)


def log_critical(message: str) -> None:
    """Log critical message using the shared logger."""
    TahoeLogger().critical(message) 