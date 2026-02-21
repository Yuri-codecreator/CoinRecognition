# Smart Currency Recognition & Value Counter (M2A Prototype)

This Python prototype implements your proposed **Smart Currency Recognition & Value Counter** for Philippine currency.

## Implemented Concepts
- **Feature Detection + Matching**: ORB keypoints and descriptors are extracted from each input image and matched against a reference database.
- **OCR Hinting**: Optional text extraction (via `pytesseract`) gives additional context for detected currency.
- **Counting and Value Accumulation**: Each accepted detection updates a breakdown counter and running PHP total.

## How It Works
1. Load validated currency template metadata from `data/reference/index.json`.
2. Extract ORB features for each template image.
3. For each input image:
   - Extract ORB features.
   - Perform KNN matching (`BFMatcher`, Hamming distance).
   - Keep the best template based on "good" matches (Lowe ratio test).
   - Accept match when good matches exceed a threshold (default: `25`).
4. If matched, add denomination value to session total and print running summary.

## Project Structure
- `currency_system/recognizer.py` — ORB/OCR processing and running total logic.
- `currency_system/cli.py` — command-line entrypoint.
- `currency_system/models.py` — result and template data models.
- `data/reference/index.json` — sample metadata for old/new/front/back/coin variants.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: OCR requires the **Tesseract** binary installed on your OS in addition to `pytesseract`.

## Usage
```bash
python -m currency_system.cli \
  --reference-index data/reference/index.json \
  sample_inputs/img1.jpg sample_inputs/img2.jpg
```

Useful flags:
- `--min-good-matches 25` (default): detection acceptance threshold.
- `--disable-ocr`: run pure ORB matching without OCR hints.

## Expected Output Example
```text
sample_inputs/img1.jpg: Detected: 100_Peso_Old_Back | Value: PHP 100.00 | Matches: 49 | Confidence: 0.98
sample_inputs/img2.jpg: Detected: 20_Peso_New_Front | Value: PHP 20.00 | Matches: 37 | Confidence: 0.74
--- Session Summary ---
100_Peso_Old_Back: x1
20_Peso_New_Front: x1
Current Total: PHP 120.00
```

## Notes
- Add your actual validated reference images in `data/reference/` and keep filenames aligned with `index.json`.
- You can include multiple versions of the same denomination (old/new, front/back, heads/tails) by adding more templates with the same `value_php` but different labels/images.
