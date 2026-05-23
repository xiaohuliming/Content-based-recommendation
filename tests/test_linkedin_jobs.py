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
        "3,Researcher,,90000,Chicago,Director,1700000100000\n"
    )
    return csv


def test_load_postings_keeps_essential_columns(fake_postings_csv):
    from src.parsers.linkedin_jobs import load_postings
    df = load_postings(fake_postings_csv)
    assert len(df) == 3
    assert set(df.columns) >= {"job_id", "title", "description", "normalized_salary",
                                "location", "formatted_experience_level", "listed_time"}
    assert df["title"].iloc[0] == "Data Analyst"
    # Null description should be normalized to empty string
    null_row = df[df["job_id"] == "3"]
    assert len(null_row) == 1
    assert null_row["description"].iloc[0] == ""


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
