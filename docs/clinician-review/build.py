#!/usr/bin/env python3
"""Render the clinician review package to PDF.

We render docs/clinician-review/msdx-clinician-review.html with the system Chromium through Playwright, because
Playwright's page.pdf() gives us a footer with page numbers, which Chromium's --print-to-pdf flag does not.
The table of contents needs page numbers, so we render twice: the first pass finds the page each section starts
on (pdftotext per page, matched on the heading text), the second pass renders with those numbers filled in.

Usage: python3 docs/clinician-review/build.py [--png DIR]
Outputs: docs/clinician-review/msdx-clinician-review.pdf and a copy at
/home/claude/.artifacts/2026-09-27-msdx-clinician-review.pdf. With --png DIR, one PNG per page at 60 dpi for a
layout check (pdftoppm).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
HTML = HERE / "msdx-clinician-review.html"
PDF = HERE / "msdx-clinician-review.pdf"
ARTIFACT = Path("/home/claude/.artifacts/2026-09-27-msdx-clinician-review.pdf")
CHROMIUM = "/usr/bin/chromium"

FOOTER = (
    '<div style="font-family: Liberation Sans, Arial, sans-serif; font-size: 8pt; color: #555; width: 100%; '
    'padding: 0 16mm; display: flex; justify-content: space-between;">'
    "<span>MedSafe-Dx: triage ratings for clinical review. Draft for review, confidential.</span>"
    '<span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>'
)
MARGIN = {"top": "18mm", "bottom": "20mm", "left": "16mm", "right": "16mm"}


def render(html_text: str, out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=HERE, delete=False) as f:
        f.write(html_text)
        tmp = Path(f.name)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(tmp.as_uri())
            page.pdf(path=str(out), format="A4", print_background=True, display_header_footer=True,
                     header_template="<div></div>", footer_template=FOOTER, margin=MARGIN)
            browser.close()
    finally:
        tmp.unlink(missing_ok=True)


def page_texts(pdf: Path) -> list[str]:
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
    return [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(pdf), "-"],
                           capture_output=True, text=True).stdout for i in range(1, n + 1)]


def fill_toc(html_text: str, pages: list[str]) -> str:
    """Replace each empty TOC page span with the first page after the contents page whose text starts with the heading."""
    def page_of(heading: str) -> str:
        for i, text in enumerate(pages[2:], start=3):  # skip the title page and the contents page
            first_lines = "\n".join(text.strip().splitlines()[:3])
            if heading in first_lines:
                return str(i)
        return "?"
    return re.sub(r'<span class="pg" data-h="([^"]+)"></span>',
                  lambda m: f'<span class="pg" data-h="{m.group(1)}">{page_of(m.group(1))}</span>', html_text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, help="write one PNG per page here (60 dpi) for a layout check")
    args = ap.parse_args()
    src = HTML.read_text()
    render(src, PDF)
    pages = page_texts(PDF)
    render(fill_toc(src, pages), PDF)
    pages = page_texts(PDF)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PDF, ARTIFACT)
    print(f"{PDF}: {len(pages)} pages; copy at {ARTIFACT}")
    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-r", "60", "-png", str(PDF), str(args.png / "page")], check=True)
        print(f"PNGs in {args.png}")
    unresolved = [m.group(1) for m in re.finditer(r'data-h="([^"]+)">\?<', fill_toc(src, pages))]
    if unresolved:
        print("TOC headings not found:", unresolved, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
