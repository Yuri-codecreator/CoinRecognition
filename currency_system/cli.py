from __future__ import annotations

import argparse
from pathlib import Path

from .recognizer import SmartCurrencyRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smart Currency Recognition & Value Counter (PHP)",
    )
    parser.add_argument(
        "--reference-index",
        type=Path,
        default=Path("data/reference/index.json"),
        help="Path to JSON index containing reference template metadata.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input image paths to process.",
    )
    parser.add_argument(
        "--min-good-matches",
        type=int,
        default=25,
        help="Minimum number of good ORB matches to accept a detection.",
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Disable OCR hinting step.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    recognizer = SmartCurrencyRecognizer(
        reference_index_path=args.reference_index,
        min_good_matches=args.min_good_matches,
        enable_ocr=not args.disable_ocr,
    )
    recognizer.load_reference_database()

    for image_path, result in recognizer.process_batch(args.inputs):
        if result.detected:
            print(
                f"{image_path}: Detected: {result.label} | Value: PHP {result.value_php:.2f} "
                f"| Matches: {result.good_matches} | Confidence: {result.confidence:.2f}"
            )
            if result.notes:
                print(f"  {result.notes}")
        else:
            print(
                f"{image_path}: Not detected | Matches: {result.good_matches} "
                f"| Confidence: {result.confidence:.2f} | {result.notes}"
            )

    print(recognizer.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
