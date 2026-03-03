"""Replace pages in a target PDF with pages from a source PDF."""
import fitz
import argparse
import sys


def replace_pages(target_path, source_path, mapping, output_path):
    """Replace pages in target PDF.

    mapping: list of (target_page, source_page) tuples, 1-indexed.
    """
    target = fitz.open(target_path)
    source = fitz.open(source_path)

    # Sort by target page descending so deletions don't shift indices
    for target_page, source_page in sorted(mapping, key=lambda x: x[0], reverse=True):
        ti = target_page - 1
        si = source_page - 1
        if ti < 0 or ti >= len(target):
            print(f"Error: target page {target_page} out of range (1-{len(target)})")
            sys.exit(1)
        if si < 0 or si >= len(source):
            print(f"Error: source page {source_page} out of range (1-{len(source)})")
            sys.exit(1)
        target.delete_page(ti)
        target.insert_pdf(source, from_page=si, to_page=si, start_at=ti)
        print(f"  Replaced page {target_page} with source page {source_page}")

    target.save(output_path)
    target.close()
    source.close()
    print(f"\nSaved to: {output_path}")


def verify(path):
    doc = fitz.open(path)
    sizes = set()
    for i in range(len(doc)):
        page = doc[i]
        r = page.rect
        w_mm = r.width / 72 * 25.4
        h_mm = r.height / 72 * 25.4
        imgs = page.get_images(full=True)
        iw, ih = 0, 0
        if imgs:
            base = doc.extract_image(imgs[0][0])
            iw, ih = base["width"], base["height"]
        sizes.add((iw, ih))
        print(f"  Page {i+1:2d}: {r.width:.1f} x {r.height:.1f} pts  "
              f"{w_mm:.1f} x {h_mm:.1f} mm  Image: {iw}x{ih}")
    doc.close()
    if len(sizes) == 1:
        print("\nAll pages match!")
    else:
        print(f"\nWARNING: {len(sizes)} different image sizes found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace pages in a target PDF with pages from a source PDF.",
        epilog="Example: python replace_pages.py target.pdf source.pdf 4:1 7:2 9:3")
    parser.add_argument("target", help="Target PDF file")
    parser.add_argument("source", help="Source PDF with replacement pages")
    parser.add_argument("pages", nargs="+",
                        help="Page mappings as TARGET:SOURCE (1-indexed), e.g. 7:1 8:2")
    parser.add_argument("-o", "--output", help="Output file (default: target_updated.pdf)")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()

    mapping = []
    for p in args.pages:
        parts = p.split(":")
        if len(parts) != 2:
            parser.error(f"Invalid mapping '{p}', expected TARGET:SOURCE (e.g. 7:1)")
        mapping.append((int(parts[0]), int(parts[1])))

    output = args.output or args.target.replace(".pdf", "_updated.pdf")
    replace_pages(args.target, args.source, mapping, output)

    if not args.no_verify:
        print(f"\nVerifying {output}:")
        verify(output)
