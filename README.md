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
| `--size` | auto | Target image size as `WxH` in pixels (e.g. `1110x1452`) |

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

# Match a specific page size (e.g. to replace pages in another PDF)
python fix_pdf.py scan.pdf --size 1110x1452 --dpi 200
```

When `--original` is used, the tool inspects each source PDF's embedded images to detect the native DPI and JPEG quality, then uses those settings for output. This avoids unnecessary downsampling or recompression.

When `--size` is used, all output pages are fitted to exactly `WxH` pixels (preserving aspect ratio, centered on a white canvas). Useful when the output must match another PDF's page dimensions for page replacement.

## Utility scripts

### analyze_pdf.py

Inspect page sizes, image dimensions, and DPI of one or more PDFs:

```
python analyze_pdf.py document.pdf

python analyze_pdf.py original.pdf processed.pdf
```

### replace_pages.py

Replace specific pages in a target PDF with pages from a source PDF. Page mappings use `TARGET:SOURCE` format (1-indexed):

```
# Replace pages 4, 7, 9 with source pages 1, 2, 3
python replace_pages.py target.pdf source.pdf 4:1 7:2 9:3

# Custom output filename
python replace_pages.py target.pdf source.pdf 7:1 8:2 -o result.pdf

# Skip post-replacement verification
python replace_pages.py target.pdf source.pdf 5:1 --no-verify
```

Output defaults to `target_updated.pdf`. Automatically verifies all pages have matching dimensions after replacement.

### rotate_pages.py

Rotate specific pages in a PDF. Rotations use `PAGE:DEGREES` format (1-indexed, clockwise):

```
# Rotate pages 1-4 by 90° and page 5 by 180°
python rotate_pages.py doc.pdf 1:90 2:90 3:90 4:90 5:180

# Custom output filename
python rotate_pages.py doc.pdf 1:90 -o result.pdf
```

Output defaults to `input_rotated.pdf`.

## Project structure

```
fix_pdf.py          - Main processing script
analyze_pdf.py      - PDF page size analyzer
replace_pages.py    - Page replacement utility
rotate_pages.py     - Page rotation utility
requirements.txt    - Python dependencies
PDFs/               - Input PDF files (default)
PDFs/processed/     - Output cropped PDF files (default)
```
