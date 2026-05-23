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
