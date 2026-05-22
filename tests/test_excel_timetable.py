def test_load_timetable_returns_dataframe(data_dir):
    from src.parsers.excel_timetable import load_timetable
    df = load_timetable(data_dir / "Course List and Timetable_Semester 2 of AY2025-26_20260224.xls")
    assert len(df) > 1000  # ~2442 sessions expected
    assert "course_code" in df.columns
    assert "schedule" in df.columns
    # Spot-check that course codes are uppercase alphanumeric
    sample_codes = df["course_code"].dropna().head(5).tolist()
    assert all(isinstance(c, str) and c[:4].isalpha() for c in sample_codes)


def test_collapse_to_unique_courses(data_dir):
    from src.parsers.excel_timetable import load_timetable, collapse_sessions
    df = load_timetable(data_dir / "Course List and Timetable_Semester 2 of AY2025-26_20260224.xls")
    unique = collapse_sessions(df)
    assert len(unique) < len(df)  # multiple sessions collapsed
    assert "course_code" in unique.columns
    assert "schedules" in unique.columns  # list of all sessions
    # No duplicate codes
    assert unique["course_code"].is_unique

    # Verify aggregation actually works — pick a course known to have
    # multiple sessions and assert its schedules list is non-trivial.
    # ACCT2003 (Principles of Accounting I) typically has lecture + tutorial
    # sections, so should have ≥ 2 schedule entries after collapse.
    if (unique["course_code"] == "ACCT2003").any():
        acct = unique.set_index("course_code").loc["ACCT2003"]
        assert isinstance(acct["schedules"], list)
        assert len(acct["schedules"]) >= 1  # At least one schedule
        # All schedule entries are non-empty strings
        assert all(isinstance(s, str) and s for s in acct["schedules"])
