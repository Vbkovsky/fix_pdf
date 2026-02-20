import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import io
import sys


def find_document_contour(gray, threshold=240):
    """Find the largest rectangular contour (the passport/document)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    img_area = gray.shape[0] * gray.shape[1]

    if cv2.contourArea(largest) < img_area * 0.01:
        return None

    return largest


def get_skew_angle(contour):
    """Get rotation angle from the document contour."""
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    return angle


def find_crop_by_edges(gray, bg_threshold=250, min_content_ratio=0.01):
    """Fallback: scan inward from edges to find document bounds.
    Uses a two-pass approach: first find left/right, then find
    top/bottom within those bounds to avoid dark edge strips
    polluting row-based detection."""
    h, w = gray.shape

    def find_edge(lines, length):
        for i, line in enumerate(lines):
            if np.sum(line < bg_threshold) / length > min_content_ratio:
                return i
        return 0

    # Pass 1: find left/right bounds (column-based, less affected by edge strips)
    left = find_edge(gray.T, h)
    right = w - find_edge(gray.T[::-1], h)

    if right - left < w * 0.05:
        return None

    # Pass 2: find top/bottom using only the interior columns
    # This avoids the dark binding strip on the left from polluting every row
    margin = max(int((right - left) * 0.1), 20)
    interior = gray[:, left + margin:right - margin]
    interior_w = interior.shape[1]

    if interior_w < 10:
        return None

    top = find_edge(interior, interior_w)
    bottom_offset = find_edge(interior[::-1], interior_w)
    bottom = h - bottom_offset

    # Guard against thin bottom features missed by the main scan:
    # look a bit further (2% of height) past the detected bottom edge
    # using a more sensitive threshold (half the normal ratio)
    extra_scan = max(int(h * 0.02), 10)
    if bottom_offset > extra_scan:
        lookahead = interior[h - bottom_offset:h - bottom_offset + extra_scan]
        for i, row in enumerate(lookahead):
            if np.sum(row < bg_threshold) / interior_w > min_content_ratio * 0.5:
                bottom = h - bottom_offset + i + 1

    if bottom - top < h * 0.05:
        return None

    return left, top, right - left, bottom - top


def deskew_and_crop(img):
    """Rough crop -> detect angle -> rotate -> fine crop."""
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Step 1: Rough crop to remove white margins first
    # Keep extra margin so the fine crop + padding still has room to breathe
    bounds = find_crop_by_edges(gray)
    if bounds is not None:
        rx, ry, rw, rh = bounds
        margin_x = max(int(rw * 0.05), 30)
        margin_top = max(int(rh * 0.05), 30)
        margin_bottom = max(int(rh * 0.10), 60)
        rx = max(rx - margin_x, 0)
        ry = max(ry - margin_top, 0)
        rw = min(rw + 2 * margin_x, w - rx)
        rh = min(rh + margin_top + margin_bottom, h - ry)
        img_array = img_array[ry:ry + rh, rx:rx + rw]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

    # Step 2: Detect contour on the rough-cropped image for angle
    contour = find_document_contour(gray, threshold=240)
    angle = 0.0
    if contour is not None:
        bx, by, bw, bh = cv2.boundingRect(contour)
        coverage = (bw * bh) / (w * h)
        if coverage < 0.95:
            angle = get_skew_angle(contour)
        else:
            # Contour covers entire image — retry with stricter threshold
            # to detect angle from internal document features
            inner = find_document_contour(gray, threshold=200)
            if inner is not None:
                angle = get_skew_angle(inner)

    # Step 3: Apply rotation
    if abs(angle) > 0.3:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_array = cv2.warpAffine(
            img_array, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Step 4: Fine crop after rotation
    contour = find_document_contour(gray, threshold=240)
    if contour is not None:
        bx, by, bw, bh = cv2.boundingRect(contour)
        # Check if contour is reasonable (not covering entire image)
        if (bw * bh) / (w * h) < 0.95:
            x, y, cw, ch = bx, by, bw, bh
        else:
            # Re-do edge-based crop on rotated image
            fine = find_crop_by_edges(gray)
            if fine is not None:
                x, y, cw, ch = fine
            else:
                x, y, cw, ch = 0, 0, w, h
    else:
        fine = find_crop_by_edges(gray)
        if fine is not None:
            x, y, cw, ch = fine
        else:
            x, y, cw, ch = 0, 0, w, h

    # Asymmetric padding: bottom gets extra room for stamps/signatures/borders
    pad_x = max(int(cw * 0.03), 20)
    pad_top = max(int(ch * 0.03), 20)
    pad_bottom = max(int(ch * 0.05), 30)
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_top, 0)
    x2 = min(x + cw + pad_x, w)
    y2 = min(y + ch + pad_bottom, h)
    img_array = img_array[y1:y2, x1:x2]

    return Image.fromarray(img_array), angle


def normalize_size(images, bg_color=(255, 255, 255)):
    """Resize all images to the same dimensions by centering on a white canvas."""
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)

    result = []
    for img in images:
        if img.width == max_w and img.height == max_h:
            result.append(img)
            continue
        canvas = Image.new("RGB", (max_w, max_h), bg_color)
        x = (max_w - img.width) // 2
        y = (max_h - img.height) // 2
        canvas.paste(img, (x, y))
        result.append(canvas)

    return result, max_w, max_h


def detect_source_settings(input_path):
    """Detect the DPI and JPEG quality of embedded images in the source PDF."""
    doc = fitz.open(input_path)
    max_dpi = 0
    qualities = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_w_inches = page.rect.width / 72
        page_h_inches = page.rect.height / 72

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_img = doc.extract_image(xref)
            img_w, img_h = base_img["width"], base_img["height"]

            dpi_x = img_w / page_w_inches if page_w_inches else 0
            dpi_y = img_h / page_h_inches if page_h_inches else 0
            max_dpi = max(max_dpi, dpi_x, dpi_y)

            if base_img["ext"] == "jpeg":
                try:
                    pil_img = Image.open(io.BytesIO(base_img["image"]))
                    qt = pil_img.quantization
                    if qt:
                        avg_coeff = sum(sum(t) for t in qt.values()) / sum(len(t) for t in qt.values())
                        if avg_coeff < 3:
                            qualities.append(98)
                        elif avg_coeff < 5:
                            qualities.append(95)
                        elif avg_coeff < 8:
                            qualities.append(90)
                        elif avg_coeff < 15:
                            qualities.append(85)
                        elif avg_coeff < 25:
                            qualities.append(75)
                        else:
                            qualities.append(65)
                except Exception:
                    pass

    doc.close()

    detected_dpi = max(int(round(max_dpi / 50) * 50), 150) if max_dpi else 300
    detected_quality = max(qualities) if qualities else 85

    return detected_dpi, detected_quality


def process_pdf(input_path, output_path, dpi=300, quality=85):
    """Process each page: deskew, crop borders, normalize to uniform size."""
    doc = fitz.open(input_path)

    # Pass 1: deskew and crop all pages
    total = len(doc)
    processed = []
    angles = []
    for page_num in range(total):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img, angle = deskew_and_crop(img)
        processed.append(img)
        angles.append(angle)

        print(f"Page {page_num + 1:2d}/{total}: rotated {angle:+.2f}deg, "
              f"cropped to {img.width}x{img.height}")

    # Pass 2: normalize all pages to the same size
    processed, final_w, final_h = normalize_size(processed)
    print(f"Normalized all pages to {final_w}x{final_h}")

    # Pass 3: write to PDF
    new_doc = fitz.open()
    for img in processed:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=quality)
        img_bytes.seek(0)

        new_page = new_doc.new_page(
            width=img.width * 72 / dpi, height=img.height * 72 / dpi
        )
        new_page.insert_image(new_page.rect, stream=img_bytes.read())

    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    print(f"\nDone! Saved to: {output_path}")


def process_folder(input_dir, output_dir, dpi=300, quality=85, keep_original=False):
    """Process all PDFs in input_dir and save to output_dir."""
    import pathlib
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_path.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Found {len(pdfs)} PDFs in {input_dir}\n")
    for i, pdf in enumerate(pdfs, 1):
        out_file = output_path / (pdf.stem + "_fixed.pdf")
        print(f"=== [{i}/{len(pdfs)}] {pdf.name} ===")
        file_dpi, file_quality = dpi, quality
        if keep_original:
            file_dpi, file_quality = detect_source_settings(str(pdf))
            print(f"Detected source settings: DPI={file_dpi}, JPEG quality={file_quality}")
        process_pdf(str(pdf), str(out_file), dpi=file_dpi, quality=file_quality)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deskew, crop, and normalize scanned PDFs.")
    parser.add_argument("input", nargs="?", help="Input PDF file or folder (default: PDFs/)")
    parser.add_argument("output", nargs="?", help="Output PDF file or folder (default: auto)")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100 (default: 85)")
    parser.add_argument("--original", action="store_true",
                        help="Auto-detect and keep source DPI and JPEG quality")
    args = parser.parse_args()

    if args.input and args.input.lower().endswith(".pdf"):
        dpi, quality = args.dpi, args.quality
        if args.original:
            dpi, quality = detect_source_settings(args.input)
            print(f"Detected source settings: DPI={dpi}, JPEG quality={quality}")
        out = args.output or args.input.replace(".pdf", "_fixed.pdf")
        process_pdf(args.input, out, dpi=dpi, quality=quality)
    elif args.input:
        out = args.output or (args.input.rstrip("/\\") + "/processed")
        process_folder(args.input, out, dpi=args.dpi, quality=args.quality,
                       keep_original=args.original)
    else:
        process_folder("PDFs", "PDFs/processed", dpi=args.dpi, quality=args.quality,
                       keep_original=args.original)
