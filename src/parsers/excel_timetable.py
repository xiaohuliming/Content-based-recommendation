"""Parser for UIC course timetable Excel (.xls)."""
from pathlib import Path
import pandas as pd


# Map source column names -> our canonical names. Update if Excel changes.
#
# The source Excel has 13 columns; we keep 9. Intentionally dropped:
#   - "Curriculum Type" (MR/ME/FE/GE etc.) — may be useful later for
#     gap-filling logic; revisit if needed
#   - "Elective Type" — programme-internal taxonomy
#   - "Hours" — derivable from schedule
#   - "Remarks" — unstructured free text
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
    df = df.dropna(subset=["course_code"]).reset_index(drop=True)
    df.loc[:, "course_code"] = df["course_code"].astype(str).str.strip()
    return df


def collapse_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-session rows into one row per course.

    Aggregates schedule + classroom + teacher into lists of distinct values.
    Other fields use the first occurrence.

    Note: `title_session` is taken raw from the Excel and contains the
    section suffix in parentheses (e.g., "Principles of Accounting I (1010)").
    Downstream consumers should prefer the PDF-derived `name` field for
    display purposes, joining on `course_code`.
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
