"""Parser for UIC Course Descriptions PDF.

PDF format (per course block):
    {CODE} {NAME, may wrap across multiple lines}
    ({N} units)
    Pre-requisite(s): {prereq text, may span multiple lines}
    Course Description: {description text, may span multiple lines}

Note: The PDF uses a two-column layout. Full-page text extraction interleaves
columns, producing garbled blocks. parse_pdf() uses column-aware extraction
(crop left half then right half per page) to get clean per-column text before
splitting into course blocks.
"""
import logging
import re
from pathlib import Path
from typing import Iterator

import pdfplumber

# Course code: 2-6 letter prefix + 4 digits (e.g., ACCT2003, GFQR1003, COMP4263)
COURSE_CODE_RE = re.compile(r"^([A-Z]{2,6}\d{4})\s", re.MULTILINE)
UNITS_RE = re.compile(r"\((\d+)\s*units?\)")
WHITESPACE_RE = re.compile(r"\s+")


def parse_course_block(block: str) -> dict:
    """Parse a single course block into a structured record.

    Returns dict with keys: code, name, units, prerequisites_text, description.
    Raises ValueError if the block doesn't match expected format.

    Handles multi-line names (where the name wraps before `(N units)`).
    """
    block = block.strip()
    code_match = COURSE_CODE_RE.match(block)
    if not code_match:
        raise ValueError(f"No course code at start of block: {block[:80]!r}")
    code = code_match.group(1)
    rest = block[code_match.end():]

    # Name spans from after code to just before "(N units)"
    units_match = UNITS_RE.search(rest)
    if not units_match:
        raise ValueError(f"No units marker found for {code}")
    name = WHITESPACE_RE.sub(" ", rest[: units_match.start()]).strip()
    units = int(units_match.group(1))

    after_units = rest[units_match.end():]

    # Prerequisites: between "Pre-requisite(s):" and "Course Description:"
    # [:：] matches both ASCII colon and Unicode fullwidth colon (U+FF1A) used
    # by a small number of courses in the PDF (e.g. PSY4073, PSY4083).
    prereq_match = re.search(
        r"Pre-requisite\(s\)[:：]\s*(.*?)(?=Course Description[:：]|$)",
        after_units,
        re.DOTALL,
    )
    prerequisites_text = WHITESPACE_RE.sub(
        " ", prereq_match.group(1).strip() if prereq_match else ""
    )

    # Description: after "Course Description:" to end
    # [:：] matches both ASCII colon and Unicode fullwidth colon (U+FF1A).
    desc_match = re.search(r"Course Description[:：]\s*(.+)$", after_units, re.DOTALL)
    description = WHITESPACE_RE.sub(
        " ", desc_match.group(1).strip() if desc_match else ""
    )

    # A real course block must have at least one of prereqs or description
    if not prerequisites_text and not description:
        raise ValueError(
            f"Block for {code} has empty prereq AND description; likely a fragment"
        )

    return {
        "code": code,
        "name": name,
        "units": units,
        "prerequisites_text": prerequisites_text,
        "description": description,
    }


def extract_course_blocks(full_text: str) -> Iterator[str]:
    """Yield course blocks from PDF text.

    A line starting with a course code is only a real block boundary if a
    `(N units)` marker appears before the next candidate code. This filters
    out course codes that appear inside prerequisite text.
    """
    positions = [m.start() for m in COURSE_CODE_RE.finditer(full_text)]
    real_starts = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(full_text)
        if UNITS_RE.search(full_text[start:end]):
            real_starts.append(start)
    for i, start in enumerate(real_starts):
        end = real_starts[i + 1] if i + 1 < len(real_starts) else len(full_text)
        yield full_text[start:end].strip()


def _extract_text(pdf_path: Path) -> str:
    """Extract raw text from the PDF using column-aware extraction.

    Crops each page into left and right halves to avoid interleaving the
    two-column layout. Skips the tall header on page 1 and footer on all pages.

    Crop values are in pdfplumber page units (PDF points, 1/72 inch):
      top_crop=90 (page 1) / 30 (other pages) — strips the page header.
      bottom_crop=height-50 — strips the page-number footer at the bottom.
    """
    with pdfplumber.open(pdf_path) as pdf:
        parts = []
        for i, page in enumerate(pdf.pages):
            width = page.width
            height = page.height
            # Page 1 has a taller header ("Rev YYYYMMDD / Course Description")
            top_crop = 90 if i == 0 else 30
            # Strip footer (page number "N / 172") from bottom
            bottom_crop = height - 50
            left = page.crop((0, top_crop, width / 2, bottom_crop))
            right = page.crop((width / 2, top_crop, width, bottom_crop))
            parts.append(left.extract_text() or "")
            parts.append(right.extract_text() or "")
    return "\n".join(parts)


def parse_text(full_text: str) -> list[dict]:
    """Parse the full extracted PDF text into a list of course records.

    Filters out blocks that fail to parse (e.g., fragment blocks with neither
    prerequisites nor description).
    """
    courses = []
    seen_codes: set[str] = set()
    for block in extract_course_blocks(full_text):
        try:
            course = parse_course_block(block)
        except ValueError as exc:
            logging.debug("Skipping unparseable block: %s", exc)
            continue
        if course["code"] in seen_codes:
            continue  # PDF sometimes has duplicates from page-break artifacts
        seen_codes.add(course["code"])
        courses.append(course)
    return courses


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Parse the full UIC Course Descriptions PDF into a list of course records.

    Uses column-aware extraction (left column then right column per page) to
    avoid interleaving of the two-column layout. Skips the first-page header.
    Filters out blocks that fail to parse (e.g., course codes appearing in
    prerequisite lists that look like block starts).
    """
    full_text = _extract_text(pdf_path)
    return parse_text(full_text)
