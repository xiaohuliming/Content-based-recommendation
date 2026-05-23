"""Parse UIC prerequisite text into a list of course codes."""
import re

COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s*([0-9]{3,4})\b", re.IGNORECASE)


def parse_prerequisites(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    matches = COURSE_CODE_RE.findall(text)
    seen: dict[str, None] = {}
    for prefix, num in matches:
        seen.setdefault(f"{prefix.upper()}{num}", None)
    return list(seen.keys())
