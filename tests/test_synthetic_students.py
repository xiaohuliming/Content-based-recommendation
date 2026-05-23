import pandas as pd
from src.students.synthetic import (
    sample_completed_courses,
    pick_career_goal,
    generate_students,
)


def test_year_1_student_has_few_courses():
    """Year-1 students shouldn't have completed Year-3 courses."""
    courses = pd.DataFrame([
        {"code": "COMP1003", "name": "Intro CS"},
        {"code": "COMP2003", "name": "Algorithms"},
        {"code": "COMP3003", "name": "Advanced ML"},
        {"code": "COMP4003", "name": "Capstone"},
    ])
    completed = sample_completed_courses(
        courses, major_prefix="COMP", year=1, seed=42
    )
    # Year-1 students should have completed 0-6 Year-1 courses, no higher
    for code in completed:
        first_digit = next(ch for ch in code if ch.isdigit())
        assert first_digit == "1", f"Year-1 student completed {code} (not year-1)"


def test_year_4_student_has_many_courses():
    courses = pd.DataFrame([
        {"code": f"COMP{y}00{i}", "name": f"Course {y}-{i}"}
        for y in [1, 2, 3, 4] for i in range(10)
    ])
    completed = sample_completed_courses(courses, major_prefix="COMP", year=4, seed=42)
    # Senior should have completed ~25+ courses across all years
    assert 20 <= len(completed) <= 40


def test_pick_career_goal_uses_provided_list():
    careers = ["Data Engineer", "ML Engineer", "Product Manager"]
    goal = pick_career_goal(careers, seed=1)
    assert goal in careers


def test_generate_students_produces_300_rows():
    courses = pd.DataFrame([
        {"code": f"{m}{y}00{i}", "name": "x"}
        for m in ["COMP", "BUS", "STAT", "ENG"]
        for y in [1, 2, 3, 4]
        for i in range(5)
    ])
    careers = ["DE", "MLE", "PM", "SDE", "Analyst"]
    df = generate_students(courses, careers, n=300, seed=42)
    assert len(df) == 300
    assert set(df.columns) == {
        "student_id", "major_prefix", "year", "completed_courses", "career_goal"
    }
    # IDs are unique
    assert df["student_id"].nunique() == 300
    # Year distribution is plausible (no year=0 or year=5)
    assert df["year"].between(1, 4).all()


def test_generate_students_is_deterministic():
    courses = pd.DataFrame([{"code": "COMP1001", "name": "x"}, {"code": "COMP2001", "name": "y"}])
    careers = ["A", "B"]
    df1 = generate_students(courses, careers, n=50, seed=99)
    df2 = generate_students(courses, careers, n=50, seed=99)
    pd.testing.assert_frame_equal(df1, df2)
