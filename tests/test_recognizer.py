import unittest
from pathlib import Path

from currency_system import recognizer
from currency_system.recognizer import SmartCurrencyRecognizer


class TestRecognizer(unittest.TestCase):
    def test_missing_opencv_raises_clear_error(self):
        original = recognizer.cv2
        recognizer.cv2 = None
        try:
            with self.assertRaises(RuntimeError) as ctx:
                SmartCurrencyRecognizer(reference_index_path=Path("data/reference/index.json"))
            self.assertIn("OpenCV is required", str(ctx.exception))
        finally:
            recognizer.cv2 = original


if __name__ == "__main__":
    unittest.main()
