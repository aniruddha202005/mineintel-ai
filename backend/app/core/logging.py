"""Logging setup."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once with a compact format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
