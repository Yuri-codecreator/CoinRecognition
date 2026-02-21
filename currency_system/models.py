from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CurrencyTemplate:
    """Metadata and feature descriptors for a known bill/coin image."""

    label: str
    value_php: float
    image_path: Path
    version: str = "default"
    side: str = "front"
    descriptors: Any = None


@dataclass(slots=True)
class MatchResult:
    """Result of matching an input image against the reference library."""

    detected: bool
    label: str | None = None
    value_php: float = 0.0
    good_matches: int = 0
    confidence: float = 0.0
    notes: str = ""
