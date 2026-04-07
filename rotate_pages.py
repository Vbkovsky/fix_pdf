"""Rotate specific pages in a PDF."""
import fitz
import argparse


def rotate_pages(input_path, rotations, output_path):
    """Rotate pages in a PDF.

    rotations: list of (page_number, degrees) tuples, 1-indexed.
    """
    doc = fitz.open(input_path)

    for page_num, degrees in rotations:
        if page_num < 1 or page_num > len(doc):
            print(f"Error: page {page_num} out of range (1-{len(doc)})")
            return
        doc[page_num - 1].set_rotation(doc[page_num - 1].rotation + degrees)
        print(f"  Page {page_num}: rotated {degrees}deg clockwise")

    doc.save(output_path)
    doc.close()
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rotate pages in a PDF.",
        epilog="Example: python rotate_pages.py doc.pdf 1:90 2:90 5:180")
    parser.add_argument("input", help="Input PDF file")
    parser.add_argument("pages", nargs="+",
                        help="Page rotations as PAGE:DEGREES (1-indexed), e.g. 1:90 5:180")
    parser.add_argument("-o", "--output", help="Output file (default: input_rotated.pdf)")
    args = parser.parse_args()

    rotations = []
    for p in args.pages:
        parts = p.split(":")
        if len(parts) != 2:
            parser.error(f"Invalid rotation '{p}', expected PAGE:DEGREES (e.g. 1:90)")
        rotations.append((int(parts[0]), int(parts[1])))

    output = args.output or args.input.replace(".pdf", "_rotated.pdf")
    rotate_pages(args.input, rotations, output)
