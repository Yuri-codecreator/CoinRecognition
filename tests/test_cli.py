import unittest

from currency_system.cli import build_parser


class TestCli(unittest.TestCase):
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["img1.jpg"])
        self.assertEqual(args.min_good_matches, 25)
        self.assertFalse(args.disable_ocr)


if __name__ == "__main__":
    unittest.main()
