"""Path configurations for Tahoe Agent.

This module provides a flexible path management system that supports:
- Default path configurations
- Environment variable overrides
- Runtime path modifications
- Centralized path management for tools
"""

import os
import pathlib
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Configuration class for managing agent paths."""

    # Base directories
    data_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("/Users/rohit/Desktop/tahoe_data")
    )
    results_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("results"))

    # Vision scores specific files
    vision_diff_filename: str = "vision_scores_sparse_all.h5ad"
    drugs_filename: str = "drugs.csv"

    def __post_init__(self) -> None:
        """Initialize paths from environment variables if available."""
        # Allow environment variable overrides
        if env_data_dir := os.getenv("TAHOE_DATA_DIR"):
            self.data_dir = pathlib.Path(env_data_dir)
        if env_results_dir := os.getenv("TAHOE_RESULTS_DIR"):
            self.results_dir = pathlib.Path(env_results_dir)
        if env_vision_diff := os.getenv("TAHOE_VISION_DIFF_FILE"):
            self.vision_diff_filename = env_vision_diff
        if env_drugs := os.getenv("TAHOE_DRUGS_FILE"):
            self.drugs_filename = env_drugs

    @property
    def vision_diff_file(self) -> pathlib.Path:
        """Get the full path to the vision diff scores file."""
        return self.data_dir / self.vision_diff_filename

    @property
    def drugs_file(self) -> pathlib.Path:
        """Get the full path to the drugs file."""
        return self.data_dir / self.drugs_filename

    def get_data_file(self, filename: str) -> pathlib.Path:
        """Get a file path in the data directory."""
        return self.data_dir / filename

    def get_results_file(self, filename: str) -> pathlib.Path:
        """Get a file path in the results directory."""
        return self.results_dir / filename

    def update_paths(self, **kwargs: Any) -> None:
        """Update path configurations at runtime.

        Args:
            **kwargs: Path configurations to update (data_dir, results_dir, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key in ["data_dir", "results_dir"]:
                    setattr(self, key, pathlib.Path(value))
                else:
                    setattr(self, key, value)
            else:
                raise ValueError(f"Unknown path configuration: {key}")

    def ensure_directories(self) -> None:
        """Create directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert path config to dictionary."""
        return {
            "data_dir": str(self.data_dir),
            "results_dir": str(self.results_dir),
            "vision_diff_filename": self.vision_diff_filename,
            "vision_diff_file": str(self.vision_diff_file),
        }


# Global path configuration instance
_path_config: Optional[PathConfig] = None


def get_paths() -> PathConfig:
    """Get the global path configuration instance."""
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    return _path_config


def configure_paths(**kwargs: Any) -> None:
    """Configure paths globally.

    Args:
        **kwargs: Path configurations to set
    """
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    _path_config.update_paths(**kwargs)


def reset_paths() -> None:
    """Reset paths to default configuration."""
    global _path_config
    _path_config = PathConfig()


# Convenience functions for common path operations
def get_data_dir() -> pathlib.Path:
    """Get the data directory path."""
    return get_paths().data_dir


def get_results_dir() -> pathlib.Path:
    """Get the results directory path."""
    return get_paths().results_dir


def get_vision_diff_file() -> pathlib.Path:
    """Get the vision diff scores file path."""
    return get_paths().vision_diff_file


def get_data_file(filename: str) -> pathlib.Path:
    """Get a file path in the data directory."""
    return get_paths().get_data_file(filename)


def get_results_file(filename: str) -> pathlib.Path:
    """Get a file path in the results directory."""
    return get_paths().get_results_file(filename)


# Backward compatibility - these match the original paths.py exports
DATA_DIR = get_data_dir()
RESULTS_DIR = get_results_dir()
VISION_DIFF_FILE = get_vision_diff_file()
