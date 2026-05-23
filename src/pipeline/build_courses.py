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

    merged = pdf_df.merge(excel_subset, on="course_code", how="left").copy()
    # Use object dtype to store Python native bools so `is True` identity checks work
    offered_flags = pd.array(
        [bool(isinstance(x, list) and len(x) > 0) for x in merged["schedules"]],
        dtype=object,
    )
    merged.loc[:, "is_offered_current_sem"] = offered_flags
    # Rename back for downstream consistency
    return merged.rename(columns={"course_code": "code"})
