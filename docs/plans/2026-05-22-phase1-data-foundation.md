# SkillPath Phase 1: Data Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline that produces three canonical CSVs — `courses_master.csv`, `jobs_clean.csv`, and `skill_taxonomy.csv` — that downstream phases (PageRank graph, clustering, recommendation) consume.

**Architecture:** Python data pipeline with focused modules. PDF parsing via `pdfplumber`, Excel via `pandas`+`xlrd`, LinkedIn via chunked `pandas.read_csv`. LLM-based skill extraction via Anthropic API (Claude Haiku for cost), with on-disk caching and mocked tests. Each module is independently testable; an orchestrator script runs the full pipeline end-to-end.

**Tech Stack:** Python 3.11+, pandas, pdfplumber, xlrd, anthropic, pytest, tqdm. No frameworks (FastAPI, Django, etc.) — this is a data-processing layer.

---

## File Structure

```
小组项目/
├── src/
│   ├── __init__.py
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── pdf_courses.py          # PDF → course records
│   │   ├── excel_timetable.py      # Excel → session records
│   │   └── linkedin_jobs.py        # archive.zip → cleaned jobs CSV
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── extractor.py            # LLM-based skill extraction
│   │   ├── prompts.py              # extraction prompts
│   │   ├── cache.py                # on-disk cache (json by hash)
│   │   └── taxonomy.py             # dedup + canonicalize + categorize
│   └── pipeline/
│       ├── __init__.py
│       └── build_phase1.py         # orchestrator
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # shared fixtures
│   ├── test_pdf_courses.py
│   ├── test_excel_timetable.py
│   ├── test_linkedin_jobs.py
│   ├── test_extractor.py
│   └── test_taxonomy.py
├── output/                          # generated artifacts (gitignored)
│   ├── courses_master.csv
│   ├── jobs_clean.csv
│   ├── jobs_sample_skills.csv
│   ├── skill_taxonomy.csv
│   └── cache/                       # LLM response cache
├── requirements.txt
└── .gitignore                       # extend existing
```

**Decomposition principle:** each parser is one file (one source = one module). Skill extraction is split into `extractor` (LLM call mechanics) + `prompts` (prompt templates) + `cache` (caching) + `taxonomy` (post-processing). Pipeline orchestrator is a thin script that calls modules in order.

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore` (extend existing)
- Create: `src/__init__.py`, `src/parsers/__init__.py`, `src/skills/__init__.py`, `src/pipeline/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create `requirements.txt`**

```text
pandas==2.2.3
pdfplumber==0.11.4
xlrd==2.0.1
anthropic==0.39.0
pytest==8.3.3
tqdm==4.66.5
python-dotenv==1.0.1
```

- [ ] **Step 2: Extend `.gitignore`**

Append to existing `.gitignore`:

```text

# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.venv/
venv/

# Output artifacts (regenerable)
output/

# Secrets
.env
```

- [ ] **Step 3: Create directories and empty `__init__.py` files**

Run:

```bash
cd "/Users/xhlm/Desktop/Study/大数据/小组项目"
mkdir -p src/parsers src/skills src/pipeline tests output/cache
touch src/__init__.py src/parsers/__init__.py src/skills/__init__.py src/pipeline/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

- [ ] **Step 5: Create `tests/conftest.py`**

```python
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

@pytest.fixture
def data_dir():
    return DATA_DIR

@pytest.fixture
def output_dir(tmp_path):
    """Use temp dir for test outputs to avoid polluting real output/."""
    return tmp_path
```

- [ ] **Step 6: Create Python virtual environment and install deps**

```bash
cd "/Users/xhlm/Desktop/Study/大数据/小组项目"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 7: Verify pytest discovers tests**

```bash
source .venv/bin/activate
pytest --collect-only
```

Expected: "collected 0 items" (no tests yet, but pytest itself runs).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .gitignore pytest.ini src/ tests/
git commit -m "feat: project scaffolding for Phase 1 data pipeline"
```

---

## Task 2: PDF Course Description Parser

**Goal:** Parse `data/Course Descriptions_20260421.pdf` into structured course records. Each record has `code, name, units, prerequisites_text, description`.

**Files:**
- Create: `src/parsers/pdf_courses.py`
- Create: `tests/test_pdf_courses.py`

**Approach:** The PDF format is highly regular. Each course block starts with a course code (e.g., `ACCT2003`) followed by name, then `(N units)`, then `Pre-requisite(s):`, then `Course Description:`. Use regex on extracted text.

- [ ] **Step 1: Write failing test for parsing a single course block**

Create `tests/test_pdf_courses.py`:

```python
from src.parsers.pdf_courses import parse_course_block

SAMPLE_BLOCK = """ACCT2003 PRINCIPLES OF ACCOUNTING I
(3 units)
Pre-requisite(s): None
Course Description: The objective of this course is to provide
students with a general understanding of basic accounting concepts."""

def test_parse_single_course_block():
    course = parse_course_block(SAMPLE_BLOCK)
    assert course["code"] == "ACCT2003"
    assert course["name"] == "PRINCIPLES OF ACCOUNTING I"
    assert course["units"] == 3
    assert course["prerequisites_text"] == "None"
    assert "general understanding of basic accounting" in course["description"]
```

- [ ] **Step 2: Run test, verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_pdf_courses.py -v
```

Expected: FAIL with `ImportError: cannot import name 'parse_course_block'`.

- [ ] **Step 3: Implement `parse_course_block`**

Create `src/parsers/pdf_courses.py`:

```python
"""Parser for UIC Course Descriptions PDF.

PDF format (per course block):
    {CODE} {NAME, may wrap across multiple lines}
    ({N} units)
    Pre-requisite(s): {prereq text, may span multiple lines}
    Course Description: {description text, may span multiple lines}
"""
import re
from pathlib import Path
from typing import Iterator

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
    code_match = re.match(r"^([A-Z]{2,6}\d{4})\s+", block)
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
    prereq_match = re.search(
        r"Pre-requisite\(s\):\s*(.*?)(?=Course Description:|$)",
        after_units,
        re.DOTALL,
    )
    prerequisites_text = WHITESPACE_RE.sub(
        " ", prereq_match.group(1).strip() if prereq_match else ""
    )

    # Description: after "Course Description:" to end
    desc_match = re.search(r"Course Description:\s*(.+)$", after_units, re.DOTALL)
    description = WHITESPACE_RE.sub(
        " ", desc_match.group(1).strip() if desc_match else ""
    )

    return {
        "code": code,
        "name": name,
        "units": units,
        "prerequisites_text": prerequisites_text,
        "description": description,
    }
```

- [ ] **Step 4: Run test, verify it passes**

```bash
pytest tests/test_pdf_courses.py::test_parse_single_course_block -v
```

Expected: PASS.

- [ ] **Step 5: Add test for multi-line prerequisite and description**

Append to `tests/test_pdf_courses.py`:

```python
SAMPLE_BLOCK_MULTILINE = """AI2003 DATA STRUCTURES AND ALGORITHM
ANALYSIS
(3 units)
Pre-requisite(s): AI1013 OBJECT-ORIENTED PROGRAMMING
Course Description: This course aims to develop the students'
knowledge in data structures and the associated algorithms.
This course introduces the concepts and techniques."""

def test_parse_multiline_block():
    course = parse_course_block(SAMPLE_BLOCK_MULTILINE)
    assert course["code"] == "AI2003"
    # Multi-line name should be fully concatenated with single spaces
    assert course["name"] == "DATA STRUCTURES AND ALGORITHM ANALYSIS"
    assert "AI1013 OBJECT-ORIENTED PROGRAMMING" in course["prerequisites_text"]
    assert "data structures and the associated algorithms" in course["description"]
    assert "concepts and techniques" in course["description"]
```

- [ ] **Step 6: Run test, verify it passes**

```bash
pytest tests/test_pdf_courses.py -v
```

Expected: both tests PASS. If the multi-line name fails, adjust the `name` regex to capture across lines until `(N units)`.

- [ ] **Step 7: Implement `extract_course_blocks` to split the full PDF text**

Append to `src/parsers/pdf_courses.py`:

```python
def extract_course_blocks(full_text: str) -> Iterator[str]:
    """Yield course blocks from PDF text.

    Splits on course code at start of line; each block runs until the next
    course code (or end of text).
    """
    # Find all start positions of course codes at line starts
    positions = [m.start() for m in COURSE_CODE_RE.finditer(full_text)]
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(full_text)
        yield full_text[start:end].strip()
```

- [ ] **Step 8: Add test for block splitter**

Append to `tests/test_pdf_courses.py`:

```python
TWO_COURSES = SAMPLE_BLOCK + "\n\n" + SAMPLE_BLOCK_MULTILINE

def test_extract_course_blocks_splits_correctly():
    from src.parsers.pdf_courses import extract_course_blocks
    blocks = list(extract_course_blocks(TWO_COURSES))
    assert len(blocks) == 2
    assert blocks[0].startswith("ACCT2003")
    assert blocks[1].startswith("AI2003")
```

- [ ] **Step 9: Run test, verify it passes**

```bash
pytest tests/test_pdf_courses.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 10: Implement full PDF parser**

Append to `src/parsers/pdf_courses.py`:

```python
import pdfplumber


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Parse the full UIC Course Descriptions PDF into a list of course records.

    Skips the first page header. Filters out blocks that fail to parse
    (e.g., page footers that accidentally start with a code-like token).
    """
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )

    courses = []
    seen_codes = set()
    for block in extract_course_blocks(full_text):
        try:
            course = parse_course_block(block)
        except ValueError:
            continue
        if course["code"] in seen_codes:
            continue  # PDF sometimes has duplicates from page-break artifacts
        seen_codes.add(course["code"])
        courses.append(course)
    return courses
```

- [ ] **Step 11: Add integration test against the real PDF**

Append to `tests/test_pdf_courses.py`:

```python
def test_parse_pdf_against_real_file(data_dir):
    from src.parsers.pdf_courses import parse_pdf
    pdf_path = data_dir / "Course Descriptions_20260421.pdf"
    courses = parse_pdf(pdf_path)
    # The PDF has ~1300 courses; allow generous bounds for parser quirks
    assert 1000 < len(courses) < 1600
    # Spot-check known courses
    codes = {c["code"] for c in courses}
    assert "ACCT2003" in codes
    assert "AI3013" in codes  # Machine Learning
    assert "COMP4263" in codes  # 3D Computer Vision
```

- [ ] **Step 12: Run integration test**

```bash
pytest tests/test_pdf_courses.py -v
```

Expected: all 4 tests PASS. If integration fails, inspect a few blocks with `parse_pdf` interactively and adjust regex.

- [ ] **Step 13: Commit**

```bash
git add src/parsers/pdf_courses.py tests/test_pdf_courses.py
git commit -m "feat(parsers): PDF course description parser"
```

---

## Task 3: Excel Timetable Parser

**Goal:** Parse `data/Course List and Timetable_*.xls` into session records. Output one row per session (a course may have multiple sessions for different times).

**Files:**
- Create: `src/parsers/excel_timetable.py`
- Create: `tests/test_excel_timetable.py`

**Approach:** The Excel file's first row is a banner title; the real header is on row 1 (0-indexed). Use `pandas.read_excel` with `skiprows=1` and `engine="xlrd"`.

- [ ] **Step 1: Inspect the real file structure interactively (no test yet)**

Run:

```bash
source .venv/bin/activate
python -c "
import pandas as pd
df = pd.read_excel(
    'data/Course List and Timetable_Semester 2 of AY2025-26_20260224.xls',
    skiprows=1, engine='xlrd'
)
print(df.shape)
print(df.columns.tolist())
print(df.head(3).to_string())
"
```

Expected: shape ~(2443, 13). Columns include Course Code, Title, Offering Unit, Offering Programme, Units, Class Schedule, Classroom, Requirements, etc.

**If the column names differ from what's used below, update the code accordingly.**

- [ ] **Step 2: Write failing test for `load_timetable`**

Create `tests/test_excel_timetable.py`:

```python
def test_load_timetable_returns_dataframe(data_dir):
    from src.parsers.excel_timetable import load_timetable
    df = load_timetable(data_dir / "Course List and Timetable_Semester 2 of AY2025-26_20260224.xls")
    assert len(df) > 1000  # ~2443 sessions expected
    assert "course_code" in df.columns
    assert "schedule" in df.columns
    # Spot-check that course codes are uppercase alphanumeric
    sample_codes = df["course_code"].dropna().head(5).tolist()
    assert all(isinstance(c, str) and c[:4].isalpha() for c in sample_codes)
```

- [ ] **Step 3: Run test, verify it fails**

```bash
pytest tests/test_excel_timetable.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 4: Implement `load_timetable`**

Create `src/parsers/excel_timetable.py`:

```python
"""Parser for UIC course timetable Excel (.xls)."""
from pathlib import Path
import pandas as pd


# Map source column names -> our canonical names. Update if Excel changes.
COLUMN_MAP = {
    "Course Code": "course_code",
    "Course Title & Session": "title_session",
    "Offering Unit": "offering_unit",
    "Offering Programme": "offering_programme",
    "Units": "units",
    "Class Schedule": "schedule",
    "Classroom": "classroom",
    "Teachers": "teacher",
    "Requirements": "requirements",
}


def load_timetable(xls_path: Path) -> pd.DataFrame:
    """Load timetable Excel into a normalized DataFrame.

    One row per session. Course codes appear multiple times if the course
    has multiple session slots.
    """
    df = pd.read_excel(xls_path, skiprows=1, engine="xlrd")
    # Rename only columns we actually need; ignore extras
    rename = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    df = df.rename(columns=rename)[list(rename.values())]
    # Drop rows where course_code is NaN (empty rows, footers)
    df = df.dropna(subset=["course_code"]).copy()
    df["course_code"] = df["course_code"].astype(str).str.strip()
    return df.reset_index(drop=True)
```

- [ ] **Step 5: Run test, verify it passes**

```bash
pytest tests/test_excel_timetable.py -v
```

Expected: PASS. If columns are named differently in the actual file, update `COLUMN_MAP` and re-run.

- [ ] **Step 6: Add test for collapsing sessions to one row per course**

Append to `tests/test_excel_timetable.py`:

```python
def test_collapse_to_unique_courses(data_dir):
    from src.parsers.excel_timetable import load_timetable, collapse_sessions
    df = load_timetable(data_dir / "Course List and Timetable_Semester 2 of AY2025-26_20260224.xls")
    unique = collapse_sessions(df)
    assert len(unique) < len(df)  # multiple sessions collapsed
    assert "course_code" in unique.columns
    assert "schedules" in unique.columns  # list of all sessions
    # No duplicate codes
    assert unique["course_code"].is_unique
```

- [ ] **Step 7: Implement `collapse_sessions`**

Append to `src/parsers/excel_timetable.py`:

```python
def collapse_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-session rows into one row per course.

    Aggregates schedule + classroom + teacher into lists.
    """
    def agg_list(s):
        return [v for v in s.dropna().astype(str).unique() if v]

    grouped = df.groupby("course_code").agg(
        title_session=("title_session", "first"),
        offering_unit=("offering_unit", "first"),
        offering_programme=("offering_programme", "first"),
        units=("units", "first"),
        schedules=("schedule", agg_list),
        classrooms=("classroom", agg_list),
        teachers=("teacher", agg_list),
        requirements=("requirements", "first"),
    ).reset_index()
    return grouped
```

- [ ] **Step 8: Run tests, verify both pass**

```bash
pytest tests/test_excel_timetable.py -v
```

Expected: both PASS.

- [ ] **Step 9: Commit**

```bash
git add src/parsers/excel_timetable.py tests/test_excel_timetable.py
git commit -m "feat(parsers): Excel timetable parser with session collapsing"
```

---

## Task 4: Course Master Record Builder

**Goal:** Join PDF (descriptions) + Excel (timetable) on course code. Produce `output/courses_master.csv` with all UIC courses.

**Files:**
- Create: `src/pipeline/build_courses.py`
- Create: `tests/test_build_courses.py`

- [ ] **Step 1: Write failing test for joining logic**

Create `tests/test_build_courses.py`:

```python
import pandas as pd

def test_build_courses_master_joins_correctly():
    from src.pipeline.build_courses import build_courses_master

    pdf_courses = [
        {"code": "AI3013", "name": "MACHINE LEARNING", "units": 3,
         "prerequisites_text": "AI1003 PYTHON PROGRAMMING", "description": "ML course."},
        {"code": "FOO9999", "name": "GHOST COURSE", "units": 3,
         "prerequisites_text": "None", "description": "Not offered."},
    ]
    excel_df = pd.DataFrame([
        {"course_code": "AI3013", "title_session": "ML Sec 1", "units": 3,
         "schedules": ["Mon 10:00"], "classrooms": ["E101"], "teachers": ["Dr X"],
         "offering_unit": "DST", "offering_programme": "AI", "requirements": ""},
    ])

    master = build_courses_master(pdf_courses, excel_df)
    assert len(master) == 2  # both PDF courses kept
    ai_row = master[master["code"] == "AI3013"].iloc[0]
    assert ai_row["is_offered_current_sem"] is True
    assert ai_row["schedules"] == ["Mon 10:00"]
    foo_row = master[master["code"] == "FOO9999"].iloc[0]
    assert foo_row["is_offered_current_sem"] is False
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_build_courses.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `build_courses_master`**

Create `src/pipeline/build_courses.py`:

```python
"""Build the course master table by joining PDF and Excel sources."""
import pandas as pd


def build_courses_master(pdf_courses: list[dict], excel_df: pd.DataFrame) -> pd.DataFrame:
    """Join PDF course descriptions with Excel timetable.

    PDF is the canonical course catalog. Excel adds current-semester
    metadata (schedule, classroom, teacher) where the course is offered.
    """
    pdf_df = pd.DataFrame(pdf_courses).rename(columns={"code": "course_code"})

    excel_cols = ["course_code", "schedules", "classrooms", "teachers",
                  "offering_unit", "offering_programme", "requirements"]
    excel_subset = excel_df[excel_cols] if all(c in excel_df.columns for c in excel_cols) else excel_df

    merged = pdf_df.merge(excel_subset, on="course_code", how="left")
    merged["is_offered_current_sem"] = merged["schedules"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    # Rename back for downstream consistency
    return merged.rename(columns={"course_code": "code"})
```

- [ ] **Step 4: Run test, verify it passes**

```bash
pytest tests/test_build_courses.py -v
```

Expected: PASS.

- [ ] **Step 5: Add integration test against real data**

Append to `tests/test_build_courses.py`:

```python
def test_build_courses_master_real_data(data_dir):
    from src.parsers.pdf_courses import parse_pdf
    from src.parsers.excel_timetable import load_timetable, collapse_sessions
    from src.pipeline.build_courses import build_courses_master

    pdf_courses = parse_pdf(data_dir / "Course Descriptions_20260421.pdf")
    excel_df = collapse_sessions(load_timetable(
        data_dir / "Course List and Timetable_Semester 2 of AY2025-26_20260224.xls"
    ))
    master = build_courses_master(pdf_courses, excel_df)

    # Sanity: at least some courses offered this semester
    offered = master["is_offered_current_sem"].sum()
    assert 300 < offered < 900
    # AI3013 ML should exist
    assert (master["code"] == "AI3013").any()
```

- [ ] **Step 6: Run integration test**

```bash
pytest tests/test_build_courses.py -v
```

Expected: both tests PASS. (Slow — runs full PDF + Excel parse.)

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/build_courses.py tests/test_build_courses.py
git commit -m "feat(pipeline): course master record builder (PDF+Excel join)"
```

---

## Task 5: LinkedIn Job Loader

**Goal:** Unpack `data/archive.zip`, load `postings.csv`, keep essential fields only (drop large blobs), output `output/jobs_clean.csv`.

**Files:**
- Create: `src/parsers/linkedin_jobs.py`
- Create: `tests/test_linkedin_jobs.py`

**Approach:** `postings.csv` is 516MB unzipped. Use `pandas.read_csv` with `usecols` to load only needed columns. Strip multi-line descriptions to flatten.

- [ ] **Step 1: Write failing test using a synthetic small CSV**

Create `tests/test_linkedin_jobs.py`:

```python
import pandas as pd
import pytest


@pytest.fixture
def fake_postings_csv(tmp_path):
    csv = tmp_path / "postings.csv"
    csv.write_text(
        "job_id,title,description,normalized_salary,location,"
        "formatted_experience_level,listed_time\n"
        "1,Data Analyst,SQL and Python required.,80000,NY,Entry,1700000000000\n"
        "2,ML Engineer,Build ML models.,150000,SF,Mid-Senior,1700000050000\n"
    )
    return csv


def test_load_postings_keeps_essential_columns(fake_postings_csv):
    from src.parsers.linkedin_jobs import load_postings
    df = load_postings(fake_postings_csv)
    assert len(df) == 2
    assert set(df.columns) >= {"job_id", "title", "description", "normalized_salary",
                                "location", "formatted_experience_level", "listed_time"}
    assert df["title"].iloc[0] == "Data Analyst"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_linkedin_jobs.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `load_postings`**

Create `src/parsers/linkedin_jobs.py`:

```python
"""Loader for LinkedIn postings CSV from the Kaggle archive."""
from pathlib import Path
import pandas as pd
import zipfile


ESSENTIAL_COLS = [
    "job_id", "title", "description", "normalized_salary",
    "location", "formatted_experience_level", "listed_time",
]


def load_postings(csv_path: Path) -> pd.DataFrame:
    """Load LinkedIn postings CSV with only essential columns.

    Drops rows with missing job_id or empty description.
    """
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in ESSENTIAL_COLS,
        dtype={"job_id": str},
    )
    df = df.dropna(subset=["job_id"])
    df["description"] = df["description"].fillna("")
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run test, verify it passes**

```bash
pytest tests/test_linkedin_jobs.py -v
```

Expected: PASS.

- [ ] **Step 5: Add helper to extract archive.zip if not already extracted**

Append to `src/parsers/linkedin_jobs.py`:

```python
def ensure_extracted(archive_path: Path, extract_dir: Path) -> Path:
    """Extract postings.csv from archive.zip if not already present.

    Returns path to postings.csv inside extract_dir.
    """
    postings_path = extract_dir / "postings.csv"
    if postings_path.exists():
        return postings_path
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extract("postings.csv", extract_dir)
    return postings_path
```

- [ ] **Step 6: Add test for archive extraction**

Append to `tests/test_linkedin_jobs.py`:

```python
import zipfile


def test_ensure_extracted_unzips_postings(tmp_path):
    from src.parsers.linkedin_jobs import ensure_extracted

    # Build a small fake archive
    archive = tmp_path / "fake.zip"
    postings_csv = tmp_path / "src_postings.csv"
    postings_csv.write_text("job_id,title\n1,Foo\n")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(postings_csv, arcname="postings.csv")

    extract_dir = tmp_path / "extracted"
    result = ensure_extracted(archive, extract_dir)
    assert result.exists()
    assert result.read_text().startswith("job_id,title")
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_linkedin_jobs.py -v
```

Expected: both PASS.

- [ ] **Step 8: Run a manual smoke test against the real archive**

```bash
source .venv/bin/activate
python -c "
from pathlib import Path
from src.parsers.linkedin_jobs import ensure_extracted, load_postings
archive = Path('data/archive.zip')
csv = ensure_extracted(archive, Path('output/linkedin_unpacked'))
df = load_postings(csv)
print(f'Loaded {len(df)} postings')
print(df.head(3))
"
```

Expected: ~124K rows loaded, sample shows real job titles.

- [ ] **Step 9: Commit**

```bash
git add src/parsers/linkedin_jobs.py tests/test_linkedin_jobs.py
git commit -m "feat(parsers): LinkedIn postings loader with archive extraction"
```

---

## Task 6: LLM-Based Skill Extractor

**Goal:** Given a chunk of text (job description or course description), return a list of technical skill strings using Claude Haiku. Cached on disk by input hash to avoid re-calling. Unit tests use mocked client.

**Files:**
- Create: `src/skills/prompts.py`
- Create: `src/skills/cache.py`
- Create: `src/skills/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Create the extraction prompt**

Create `src/skills/prompts.py`:

```python
"""Prompts for LLM-based skill extraction."""

EXTRACTION_PROMPT = """Extract concrete technical skills from the following text.

Rules:
- Include: programming languages, frameworks, tools, methodologies, technical concepts (e.g., "Python", "Machine Learning", "SQL", "TensorFlow", "Statistical Analysis").
- Exclude: soft skills (e.g., "communication", "teamwork"), job titles, company names, locations, vague phrases.
- Use canonical names (e.g., "Python" not "python programming"; "Machine Learning" not "ML").
- Limit: maximum 15 skills per text.

Text:
\"\"\"
{text}
\"\"\"

Return ONLY a valid JSON array of skill strings. No explanation, no markdown fences.
Example output: ["Python", "SQL", "Machine Learning", "Data Visualization"]
"""
```

- [ ] **Step 2: Write failing test for the cache module**

Create `tests/test_extractor.py`:

```python
def test_cache_roundtrip(tmp_path):
    from src.skills.cache import SkillCache
    cache = SkillCache(tmp_path / "cache.json")
    assert cache.get("hello world") is None
    cache.set("hello world", ["Python", "ML"])
    assert cache.get("hello world") == ["Python", "ML"]


def test_cache_persists_across_instances(tmp_path):
    from src.skills.cache import SkillCache
    path = tmp_path / "cache.json"
    cache1 = SkillCache(path)
    cache1.set("foo", ["SQL"])
    cache1.flush()

    cache2 = SkillCache(path)
    assert cache2.get("foo") == ["SQL"]
```

- [ ] **Step 3: Run test, verify it fails**

```bash
pytest tests/test_extractor.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 4: Implement `SkillCache`**

Create `src/skills/cache.py`:

```python
"""On-disk cache for skill extraction results, keyed by SHA-1 of input text."""
import hashlib
import json
from pathlib import Path


class SkillCache:
    """Simple JSON-backed cache for skill extraction.

    Use `get`/`set` for individual entries. Call `flush()` to write to disk.
    Pass `autoflush=True` to write on every `set` (slower, safer).
    """

    def __init__(self, path: Path, autoflush: bool = False):
        self.path = Path(path)
        self.autoflush = autoflush
        self._data: dict[str, list[str]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[str] | None:
        return self._data.get(self._key(text))

    def set(self, text: str, skills: list[str]) -> None:
        self._data[self._key(text)] = skills
        if self.autoflush:
            self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2))
```

- [ ] **Step 5: Run cache tests**

```bash
pytest tests/test_extractor.py -v
```

Expected: 2 PASS.

- [ ] **Step 6: Write failing test for `SkillExtractor` with mock**

Append to `tests/test_extractor.py`:

```python
from unittest.mock import MagicMock


def _mock_client_returning(content: str):
    """Build a mock Anthropic client whose messages.create returns the given text."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=content)]
    client.messages.create.return_value = response
    return client


def test_extractor_returns_parsed_skills(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('["Python", "SQL", "Machine Learning"]')
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("We need Python and SQL skills.")
    assert skills == ["Python", "SQL", "Machine Learning"]


def test_extractor_handles_text_around_json(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('Sure! Here are the skills:\n["Python"]\nThanks.')
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("Python required.")
    assert skills == ["Python"]


def test_extractor_returns_empty_on_bad_json(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning("not json at all")
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("Some text.")
    assert skills == []


def test_extractor_uses_cache(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('["Python"]')
    cache = SkillCache(tmp_path / "c.json")
    extractor = SkillExtractor(client=client, cache=cache)

    extractor.extract("hello")
    extractor.extract("hello")  # second call should hit cache
    assert client.messages.create.call_count == 1


def test_extractor_skips_short_text(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('["Python"]')
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("hi")  # too short
    assert skills == []
    assert client.messages.create.call_count == 0
```

- [ ] **Step 7: Run tests, verify they fail**

```bash
pytest tests/test_extractor.py -v
```

Expected: 5 new tests FAIL with ImportError or AttributeError.

- [ ] **Step 8: Implement `SkillExtractor`**

Create `src/skills/extractor.py`:

```python
"""LLM-based skill extraction with caching."""
import json
from typing import Optional

from src.skills.cache import SkillCache
from src.skills.prompts import EXTRACTION_PROMPT


class SkillExtractor:
    """Extract skills from text using an LLM, with on-disk caching.

    The `client` parameter accepts any object with a `messages.create(...)`
    method returning an object with `.content[0].text` (Anthropic-style).
    Passing a mock client is the supported test strategy.
    """

    MIN_LENGTH = 20

    def __init__(
        self,
        client,
        cache: SkillCache,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 512,
    ):
        self.client = client
        self.cache = cache
        self.model = model
        self.max_tokens = max_tokens

    def extract(self, text: str) -> list[str]:
        text = (text or "").strip()
        if len(text) < self.MIN_LENGTH:
            return []

        cached = self.cache.get(text)
        if cached is not None:
            return cached

        prompt = EXTRACTION_PROMPT.format(text=text)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        skills = self._parse_json_array(raw)
        self.cache.set(text, skills)
        return skills

    @staticmethod
    def _parse_json_array(raw: str) -> list[str]:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            value = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return []
        if not isinstance(value, list):
            return []
        return [str(s).strip() for s in value if str(s).strip()]
```

- [ ] **Step 9: Run tests, verify all pass**

```bash
pytest tests/test_extractor.py -v
```

Expected: 7 PASS.

- [ ] **Step 10: Commit**

```bash
git add src/skills/ tests/test_extractor.py
git commit -m "feat(skills): LLM extractor with on-disk cache and mocked tests"
```

---

## Task 7: Skill Taxonomy Builder

**Goal:** Take all extracted skill strings (from many job + course extractions), deduplicate aggressively (case-insensitive, alias mapping), and produce a canonical taxonomy CSV.

**Files:**
- Create: `src/skills/taxonomy.py`
- Create: `tests/test_taxonomy.py`

- [ ] **Step 1: Write failing test for normalization**

Create `tests/test_taxonomy.py`:

```python
def test_normalize_strips_and_titlecases():
    from src.skills.taxonomy import normalize_skill
    assert normalize_skill("  python  ") == "Python"
    assert normalize_skill("PYTHON") == "Python"
    assert normalize_skill("machine learning") == "Machine Learning"
    # Acronyms preserved
    assert normalize_skill("sql") == "SQL"
    assert normalize_skill("ML") == "Machine Learning"  # via alias map


def test_normalize_returns_empty_for_junk():
    from src.skills.taxonomy import normalize_skill
    assert normalize_skill("") == ""
    assert normalize_skill("   ") == ""
    assert normalize_skill("a") == ""  # single char
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_taxonomy.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `normalize_skill` with alias map**

Create `src/skills/taxonomy.py`:

```python
"""Skill taxonomy: deduplicate, canonicalize, categorize raw extracted skills."""
from collections import Counter
import pandas as pd


# Acronyms that should stay uppercase
ACRONYMS = {
    "SQL", "NLP", "AI", "API", "REST", "ETL", "AWS", "GCP", "TCP", "CSS",
    "HTML", "CRUD", "JSON", "XML", "CI/CD", "GUI", "UI", "UX", "OS", "PHP",
    "CSV", "PDF", "URL", "HTTP", "HTTPS", "VPN", "RPC", "RAG", "LLM", "GAN",
    "CNN", "RNN", "GPU", "CPU", "SVM", "PCA", "KNN",
}

# Map common aliases / variants -> canonical form
ALIASES = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "ai": "Artificial Intelligence",
    "py": "Python",
    "python programming": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "ts": "TypeScript",
    "k8s": "Kubernetes",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "powerbi": "Power BI",
    "power bi": "Power BI",
}


def normalize_skill(raw: str) -> str:
    """Canonicalize a single raw skill string. Returns "" for junk."""
    s = (raw or "").strip()
    if len(s) < 2:
        return ""
    low = s.lower()
    if low in ALIASES:
        return ALIASES[low]
    if s.upper() in ACRONYMS:
        return s.upper()
    # Title-case multi-word skills, preserving acronyms within
    parts = []
    for word in s.split():
        if word.upper() in ACRONYMS:
            parts.append(word.upper())
        else:
            parts.append(word.capitalize())
    return " ".join(parts)
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/test_taxonomy.py -v
```

Expected: PASS.

- [ ] **Step 5: Add test for taxonomy DataFrame builder**

Append to `tests/test_taxonomy.py`:

```python
def test_build_taxonomy_dedups_and_counts():
    from src.skills.taxonomy import build_taxonomy

    raw_skills_per_doc = [
        ["python", "SQL", "ML"],
        ["Python", "machine learning"],
        ["PYTHON", "Excel"],
        [],
    ]
    tax = build_taxonomy(raw_skills_per_doc)
    # Three unique canonical skills: Python, SQL, Machine Learning, Excel
    assert set(tax["skill"]) == {"Python", "SQL", "Machine Learning", "Excel"}
    py_count = tax.loc[tax["skill"] == "Python", "doc_frequency"].iloc[0]
    assert py_count == 3
```

- [ ] **Step 6: Run test, verify it fails**

```bash
pytest tests/test_taxonomy.py -v
```

Expected: 1 new test FAIL.

- [ ] **Step 7: Implement `build_taxonomy`**

Append to `src/skills/taxonomy.py`:

```python
def build_taxonomy(extracted_per_doc: list[list[str]]) -> pd.DataFrame:
    """Aggregate per-document skill lists into a canonical taxonomy table.

    Returns DataFrame with columns: skill_id, skill, doc_frequency.
    `doc_frequency` = number of documents where the skill appears (after canonicalization).
    """
    counter: Counter[str] = Counter()
    for doc_skills in extracted_per_doc:
        canonical = {normalize_skill(s) for s in doc_skills}
        canonical.discard("")
        for s in canonical:
            counter[s] += 1

    items = counter.most_common()
    df = pd.DataFrame(
        [{"skill": s, "doc_frequency": c} for s, c in items]
    )
    df.insert(0, "skill_id", [f"skill_{i:04d}" for i in range(len(df))])
    return df
```

- [ ] **Step 8: Run tests, verify all pass**

```bash
pytest tests/test_taxonomy.py -v
```

Expected: 3 PASS.

- [ ] **Step 9: Add categorization step**

Append to `tests/test_taxonomy.py`:

```python
def test_categorize_assigns_buckets():
    from src.skills.taxonomy import categorize_skill
    assert categorize_skill("Python") == "Programming Languages"
    assert categorize_skill("Machine Learning") == "ML & AI"
    assert categorize_skill("SQL") == "Data Systems"
    assert categorize_skill("Statistics") == "Math & Stats"
    assert categorize_skill("Foo Bar Unknown") == "Other"
```

- [ ] **Step 10: Run test, verify it fails**

```bash
pytest tests/test_taxonomy.py -v
```

Expected: 1 new FAIL.

- [ ] **Step 11: Implement `categorize_skill`**

Append to `src/skills/taxonomy.py`:

```python
CATEGORY_KEYWORDS = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "php", "scala", "kotlin", "swift", "r", "matlab", "sas",
    ],
    "ML & AI": [
        "machine learning", "deep learning", "artificial intelligence", "neural",
        "tensorflow", "pytorch", "scikit-learn", "computer vision", "nlp",
        "natural language", "reinforcement learning", "generative", "llm",
        "transformer", "bert", "gpt", "cnn", "rnn", "gan",
    ],
    "Data Systems": [
        "sql", "nosql", "mongodb", "postgresql", "mysql", "spark", "hadoop",
        "kafka", "etl", "data warehouse", "snowflake", "bigquery", "redshift",
        "airflow", "dbt", "data pipeline", "data engineering",
    ],
    "Math & Stats": [
        "statistics", "statistical", "probability", "regression", "linear algebra",
        "calculus", "optimization", "bayesian", "hypothesis testing", "anova",
        "time series", "econometrics",
    ],
    "Tools & Platforms": [
        "git", "docker", "kubernetes", "aws", "gcp", "azure", "jenkins", "ci/cd",
        "tableau", "power bi", "excel", "jupyter", "vscode", "linux",
    ],
    "Domain Knowledge": [
        "marketing", "finance", "accounting", "supply chain", "healthcare",
        "law", "education", "biology", "chemistry", "physics", "economics",
    ],
}


def categorize_skill(skill: str) -> str:
    """Assign a skill to a category bucket based on keyword matching."""
    low = skill.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return category
    return "Other"


def build_taxonomy_with_categories(extracted_per_doc: list[list[str]]) -> pd.DataFrame:
    """Build taxonomy and attach category column."""
    df = build_taxonomy(extracted_per_doc)
    df["category"] = df["skill"].apply(categorize_skill)
    return df
```

- [ ] **Step 12: Run all taxonomy tests**

```bash
pytest tests/test_taxonomy.py -v
```

Expected: 4 PASS.

- [ ] **Step 13: Commit**

```bash
git add src/skills/taxonomy.py tests/test_taxonomy.py
git commit -m "feat(skills): taxonomy builder with normalization and categorization"
```

---

## Task 8: Phase 1 Pipeline Orchestrator

**Goal:** A single script that runs the full Phase 1 pipeline end-to-end. Produces all output CSVs. Resumable (uses cached LLM calls).

**Files:**
- Create: `src/pipeline/build_phase1.py`

**Approach:** This is a script, not a library function. It calls modules in order, with progress bars and clear logging. No formal unit test — the modules are tested individually. Manual smoke test after running.

- [ ] **Step 1: Write the orchestrator script**

Create `src/pipeline/build_phase1.py`:

```python
"""Phase 1 orchestrator: produce all data-foundation outputs.

Run from project root:
    source .venv/bin/activate
    python -m src.pipeline.build_phase1

Outputs (in output/):
    - courses_master.csv         all UIC courses with descriptions + timetable
    - jobs_clean.csv             all 124K LinkedIn jobs, essential cols only
    - jobs_sample_skills.csv     5K random sample with extracted skills
    - courses_skills.csv         courses with extracted skills column
    - skill_taxonomy.csv         canonical skill list with categories

Environment:
    ANTHROPIC_API_KEY must be set (e.g., in .env or shell).

Reruns are resumable: LLM calls are cached on disk in output/cache/.
"""
import os
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from src.parsers.excel_timetable import collapse_sessions, load_timetable
from src.parsers.linkedin_jobs import ensure_extracted, load_postings
from src.parsers.pdf_courses import parse_pdf
from src.pipeline.build_courses import build_courses_master
from src.skills.cache import SkillCache
from src.skills.extractor import SkillExtractor
from src.skills.taxonomy import build_taxonomy_with_categories, normalize_skill

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

PDF_PATH = DATA_DIR / "Course Descriptions_20260421.pdf"
EXCEL_PATH = DATA_DIR / "Course List and Timetable_Semester 2 of AY2025-26_20260224.xls"
LINKEDIN_ARCHIVE = DATA_DIR / "archive.zip"

JOB_SAMPLE_SIZE = 5000
RANDOM_SEED = 42


def main() -> None:
    load_dotenv()
    assert os.getenv("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY in .env or shell"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step A: courses (PDF + Excel) ----
    print("[1/5] Parsing PDF course descriptions...")
    pdf_courses = parse_pdf(PDF_PATH)
    print(f"  parsed {len(pdf_courses)} courses")

    print("[2/5] Loading Excel timetable...")
    excel_df = collapse_sessions(load_timetable(EXCEL_PATH))
    print(f"  collapsed to {len(excel_df)} unique offerings")

    courses_master = build_courses_master(pdf_courses, excel_df)
    courses_master.to_csv(OUTPUT_DIR / "courses_master.csv", index=False)
    print(f"  wrote courses_master.csv ({len(courses_master)} rows)")

    # ---- Step B: LinkedIn jobs ----
    print("[3/5] Loading LinkedIn postings...")
    postings_csv = ensure_extracted(LINKEDIN_ARCHIVE, OUTPUT_DIR / "linkedin_unpacked")
    jobs = load_postings(postings_csv)
    jobs_clean = jobs.drop(columns=["description"]).copy()
    jobs_clean.to_csv(OUTPUT_DIR / "jobs_clean.csv", index=False)
    print(f"  wrote jobs_clean.csv ({len(jobs_clean)} rows)")

    # ---- Step C: skill extraction (sample of jobs + all courses) ----
    print("[4/5] Extracting skills via LLM (this may take 20-40 min on first run)...")
    client = Anthropic()
    cache = SkillCache(OUTPUT_DIR / "cache" / "skill_extraction.json")
    extractor = SkillExtractor(client=client, cache=cache)

    job_sample = jobs.sample(n=min(JOB_SAMPLE_SIZE, len(jobs)), random_state=RANDOM_SEED)
    job_skills = []
    for desc in tqdm(job_sample["description"], desc="  jobs"):
        job_skills.append(extractor.extract(desc))
    cache.flush()
    job_sample = job_sample.assign(extracted_skills=job_skills)
    job_sample.drop(columns=["description"]).assign(
        extracted_skills=[",".join(s) for s in job_skills]
    ).to_csv(OUTPUT_DIR / "jobs_sample_skills.csv", index=False)
    print(f"  wrote jobs_sample_skills.csv ({len(job_sample)} rows)")

    course_skills = []
    for desc in tqdm(courses_master["description"].fillna(""), desc="  courses"):
        course_skills.append(extractor.extract(desc))
    cache.flush()
    courses_master["extracted_skills"] = [",".join(s) for s in course_skills]
    courses_master.to_csv(OUTPUT_DIR / "courses_skills.csv", index=False)
    print(f"  wrote courses_skills.csv ({len(courses_master)} rows)")

    # ---- Step D: taxonomy ----
    print("[5/5] Building skill taxonomy...")
    all_per_doc = job_skills + course_skills
    taxonomy = build_taxonomy_with_categories(all_per_doc)
    taxonomy.to_csv(OUTPUT_DIR / "skill_taxonomy.csv", index=False)
    print(f"  wrote skill_taxonomy.csv ({len(taxonomy)} unique skills)")

    # ---- Summary ----
    print("\nPhase 1 complete. Outputs in output/:")
    for name in ["courses_master.csv", "jobs_clean.csv", "jobs_sample_skills.csv",
                 "courses_skills.csv", "skill_taxonomy.csv"]:
        path = OUTPUT_DIR / name
        size_kb = path.stat().st_size / 1024
        print(f"  {name:30s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `.env` file template (do NOT commit actual key)**

```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env.example
```

The user creates `.env` from this template with their real key. `.env` is already in `.gitignore`.

- [ ] **Step 3: Manually verify the orchestrator runs end-to-end**

```bash
source .venv/bin/activate
# Set your API key in .env first
python -m src.pipeline.build_phase1
```

Expected:
- All 5 steps complete
- 5 CSVs in `output/`
- First run takes 20-40 min (skill extraction); subsequent runs much faster (cache)

If LLM extraction errors out partway, the cache preserves progress; just rerun.

- [ ] **Step 4: Spot-check output CSVs**

```bash
head -3 output/courses_master.csv
head -3 output/skill_taxonomy.csv
wc -l output/*.csv
```

Expected:
- `courses_master.csv`: ~1000-1500 rows
- `jobs_clean.csv`: ~124K rows
- `jobs_sample_skills.csv`: ~5000 rows
- `skill_taxonomy.csv`: ~400-800 unique skills

- [ ] **Step 5: Run full test suite once more**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/build_phase1.py .env.example
git commit -m "feat(pipeline): Phase 1 orchestrator script"
```

---

## Phase 1 Done — Verification

After Task 8, the following must be true:

- [ ] All tests pass: `pytest -v`
- [ ] Five CSVs exist in `output/` with sensible row counts (see Task 8 Step 4)
- [ ] `output/cache/skill_extraction.json` exists and is non-empty (resumability)
- [ ] No secrets committed (`.env` should NOT appear in `git status`)
- [ ] Spot-check 10 random rows of `courses_master.csv` — descriptions look complete
- [ ] Spot-check 10 random rows of `skill_taxonomy.csv` — skills look sensible (no obvious junk like "the" or "a")

If any of these fail, fix before moving to Phase 2.

---

## What Phase 2 Will Build On

The three canonical CSVs produced here are the inputs to Phase 2:

- `courses_master.csv` + `courses_skills.csv` → course nodes + course→skill edges
- `jobs_sample_skills.csv` → career→skill weights (after grouping by job title)
- `skill_taxonomy.csv` → skill node list

Phase 2 plan will be written after Phase 1 is verified working. Don't write Phase 2 code until then.
