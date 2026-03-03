"""Analyze page sizes and embedded images in PDF files."""
import fitz
import argparse


def analyze(path):
    doc = fitz.open(path)
    print(f"\n{path} ({len(doc)} pages)")
    for i in range(len(doc)):
        page = doc[i]
        r = page.rect
        w_mm = r.width / 72 * 25.4
        h_mm = r.height / 72 * 25.4
        imgs = page.get_images(full=True)
        iw, ih, dpi_x = 0, 0, 0
        if imgs:
            base = doc.extract_image(imgs[0][0])
            iw, ih = base["width"], base["height"]
            dpi_x = iw / (r.width / 72) if r.width else 0
        print(f"  Page {i+1:2d}: {r.width:.1f} x {r.height:.1f} pts  "
              f"{w_mm:.1f} x {h_mm:.1f} mm  Image: {iw}x{ih}  ~{dpi_x:.0f} DPI")
    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze page sizes in PDF files.")
    parser.add_argument("files", nargs="+", help="One or more PDF files to analyze")
    args = parser.parse_args()

    for f in args.files:
        analyze(f)
