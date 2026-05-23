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

    Drops rows with missing job_id. Fills missing description with "".
    """
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in ESSENTIAL_COLS,
        dtype={"job_id": str, "description": "object"},
    )
    df = df.dropna(subset=["job_id"]).copy()
    df.loc[:, "description"] = df["description"].fillna("")
    return df.reset_index(drop=True)


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
