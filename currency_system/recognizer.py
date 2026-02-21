from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import CurrencyTemplate, MatchResult

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


class SmartCurrencyRecognizer:
    """ORB-based currency recognizer + running value counter for PHP."""

    def __init__(
        self,
        reference_index_path: Path,
        min_good_matches: int = 25,
        ratio_test: float = 0.75,
        enable_ocr: bool = True,
    ) -> None:
        self.reference_index_path = Path(reference_index_path)
        self.min_good_matches = min_good_matches
        self.ratio_test = ratio_test
        self.enable_ocr = enable_ocr

        self.templates: list[CurrencyTemplate] = []
        self.breakdown: dict[str, int] = defaultdict(int)
        self.total_value: float = 0.0

        self._ensure_cv_ready()
        self.orb = cv2.ORB_create(nfeatures=2500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def load_reference_database(self) -> None:
        with self.reference_index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        base_dir = self.reference_index_path.parent
        templates: list[CurrencyTemplate] = []

        for item in payload.get("templates", []):
            image_path = (base_dir / item["image"]).resolve()
            gray = self._read_grayscale(image_path)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) == 0:
                continue

            templates.append(
                CurrencyTemplate(
                    label=item["label"],
                    value_php=float(item["value_php"]),
                    image_path=image_path,
                    version=item.get("version", "default"),
                    side=item.get("side", "front"),
                    descriptors=descriptors,
                )
            )

        self.templates = templates

    def process_image(self, image_path: Path) -> MatchResult:
        if not self.templates:
            raise RuntimeError("Reference database is empty. Call load_reference_database() first.")

        query = self._read_grayscale(image_path)
        q_keypoints, q_descriptors = self.orb.detectAndCompute(query, None)
        if q_descriptors is None or len(q_keypoints) == 0:
            return MatchResult(detected=False, notes="No features found in input image")

        best_template = None
        best_good_matches = 0

        for template in self.templates:
            raw_matches = self.matcher.knnMatch(q_descriptors, template.descriptors, k=2)
            good_matches = [m for m, n in raw_matches if m.distance < self.ratio_test * n.distance]

            if len(good_matches) > best_good_matches:
                best_good_matches = len(good_matches)
                best_template = template

        if best_template is None or best_good_matches < self.min_good_matches:
            return MatchResult(
                detected=False,
                good_matches=best_good_matches,
                confidence=(best_good_matches / max(self.min_good_matches, 1)),
                notes="No reliable match found",
            )

        confidence = min(1.0, best_good_matches / (self.min_good_matches * 2))
        ocr_note = self._ocr_hint(query) if self.enable_ocr else ""

        self.total_value += best_template.value_php
        self.breakdown[best_template.label] += 1

        return MatchResult(
            detected=True,
            label=best_template.label,
            value_php=best_template.value_php,
            good_matches=best_good_matches,
            confidence=confidence,
            notes=ocr_note,
        )

    def process_batch(self, image_paths: Iterable[Path]) -> list[tuple[Path, MatchResult]]:
        return [(path, self.process_image(path)) for path in image_paths]

    def summary(self) -> str:
        lines = ["--- Session Summary ---"]
        for label, count in sorted(self.breakdown.items()):
            lines.append(f"{label}: x{count}")
        lines.append(f"Current Total: PHP {self.total_value:,.2f}")
        return "\n".join(lines)

    def _read_grayscale(self, path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def _ocr_hint(self, gray_image) -> str:
        if pytesseract is None:
            return "OCR unavailable: pytesseract not installed"

        raw_text = pytesseract.image_to_string(gray_image, config="--psm 6")
        if not raw_text.strip():
            return "OCR executed but no text extracted"

        normalized = raw_text.lower()
        hit = "peso" in normalized or any(token in normalized for token in ["twenty", "fifty", "hundred"])
        return f"OCR hint: {'currency text likely detected' if hit else 'no clear currency text'}"

    @staticmethod
    def _ensure_cv_ready() -> None:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required but not installed. Install dependencies with `pip install -r requirements.txt`."
            )
