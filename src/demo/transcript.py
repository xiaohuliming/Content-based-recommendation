"""Parse uploaded transcript PDFs into structured profile data.

Pipeline: pdfplumber extracts raw text → LLM (existing OpenAI-compatible client)
returns JSON {major, year, gpa, completed_courses, current_courses}.

Works for text-based transcript PDFs (most official ones). Image-only scans
will produce empty text — the LLM is told to return all-empties in that case.
"""
import io
import json
import logging
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You will be given the raw text extracted from a student's
university academic transcript. Extract the following fields and return ONLY a
valid JSON object — no markdown, no commentary.

Schema:
{
  "major": "<string, e.g. 'Computer Science' or '计算机科学'. Use the most specific name. Empty string if not found.>",
  "year": <integer 1-5 indicating current academic year. 1=freshman, 4=senior. Use 0 if unknown.>,
  "gpa": <float, cumulative GPA on the document's scale (usually 0-4 or 0-5). null if not found.>,
  "completed_courses": ["<course_code>", ...],
  "current_courses": ["<course_code>", ...]
}

Rules for `completed_courses`:
  - Include every course with a final grade (letter grade A-F or numeric).
  - Use the exact course CODE as printed (e.g. "COMP1003", "ACCT2003").
  - Strip whitespace. Do NOT include the course title.
  - Exclude courses marked withdrawn, dropped, in-progress, or audit.

Rules for `current_courses`:
  - Courses with no grade yet — often labeled "in progress", "current term",
    "registered", or appearing under the most recent semester without a grade.
  - Same code-only format.

If the text is empty, garbled, or clearly not a transcript, return all-empty:
{"major":"","year":0,"gpa":null,"completed_courses":[],"current_courses":[]}

Raw transcript text:
\"\"\"
{text}
\"\"\"
"""


def extract_transcript_text(pdf_bytes: bytes) -> str:
    """Extract concatenated text from every page of a PDF byte stream."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages.append(t)
    return "\n\n".join(pages)


def parse_transcript_with_llm(
    text: str,
    client,
    model: str,
    *,
    max_tokens: int = 2048,
) -> dict:
    """Send extracted text to LLM and return structured dict.

    On any failure (LLM error, bad JSON, missing keys) returns an all-empty
    result rather than raising — caller can detect via len(completed_courses)==0.
    """
    empty = {"major": "", "year": 0, "gpa": None, "completed_courses": [], "current_courses": []}
    if not text or not text.strip():
        return empty

    # Cap input to avoid token blowup on huge transcripts
    text_capped = text[:20000]
    prompt = EXTRACTION_PROMPT.replace("{text}", text_capped)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.warning("transcript LLM call failed: %s", exc)
        return empty

    raw = ""
    if response.choices and response.choices[0].message:
        raw = response.choices[0].message.content or ""

    # Locate first JSON object in response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return empty
    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return empty

    return {
        "major": str(parsed.get("major") or "").strip(),
        "year": int(parsed.get("year") or 0),
        "gpa": parsed.get("gpa") if isinstance(parsed.get("gpa"), (int, float)) else None,
        "completed_courses": [
            str(c).strip() for c in (parsed.get("completed_courses") or [])
            if isinstance(c, str) and c.strip()
        ],
        "current_courses": [
            str(c).strip() for c in (parsed.get("current_courses") or [])
            if isinstance(c, str) and c.strip()
        ],
    }
