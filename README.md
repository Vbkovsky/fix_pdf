# PDF Scan Fixer

Deskews and crops scanned PDF documents. Designed for flatbed-scanned passport spreads and ID cards but works with any scanned document PDFs.

## What it does

1. **Rough crop** — removes white scanner margins using edge detection
2. **Deskew** — detects and corrects page tilt using contour analysis
3. **Fine crop** — trims to document bounds with proportional padding (extra room at the bottom for stamps/signatures)
4. **Normalize** — makes all pages the same size (centered on white canvas)

## Requirements

- Python 3.10+

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

**Process all PDFs in `PDFs/` folder:**

```
python fix_pdf.py
```

Results are saved to `PDFs/processed/` with a `_fixed` suffix.

**Process a single file:**

```
python fix_pdf.py input.pdf output.pdf
```

**Process a single file (auto-named output):**

```
python fix_pdf.py input.pdf
```

Output will be saved as `input_fixed.pdf`.

**Process a custom folder:**

```
python fix_pdf.py my_scans/ my_scans/output/
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dpi` | 300 | Render resolution. Higher = better quality but larger files |
| `--quality` | 85 | JPEG compression quality (1–100) |
| `--original` | off | Auto-detect and keep source DPI and JPEG quality |

**Examples:**

```
# Default settings (300 DPI, quality 85)
python fix_pdf.py

# Higher quality output
python fix_pdf.py --dpi 400 --quality 95

# Preserve original source quality (auto-detected per file)
python fix_pdf.py --original

# Single file with custom settings
python fix_pdf.py scan.pdf --dpi 200 --quality 90
```

When `--original` is used, the tool inspects each source PDF's embedded images to detect the native DPI and JPEG quality, then uses those settings for output. This avoids unnecessary downsampling or recompression.

## Project structure

```
fix_pdf.py          - Main processing script
requirements.txt    - Python dependencies
PDFs/               - Input PDF files (default)
PDFs/processed/     - Output cropped PDF files (default)
```
