import pandas as pd
from src.eval.baseline import recommend_by_cosine


def test_cosine_recommends_courses_with_most_shared_skills_with_career():
    courses = pd.DataFrame([
        {"code": "A", "canonical_skills": "Python;SQL"},
        {"code": "B", "canonical_skills": "Python;ML;TensorFlow"},
        {"code": "C", "canonical_skills": "Marketing"},
    ])
    career_skills = {"Python": 0.5, "ML": 0.5, "TensorFlow": 0.3}
    recs = recommend_by_cosine(courses, career_skills, top_k=3)
    rec_codes = [r[0] for r in recs]
    assert rec_codes.index("B") < rec_codes.index("A")
    assert rec_codes.index("A") < rec_codes.index("C")


def test_cosine_handles_empty_course_skills():
    courses = pd.DataFrame([
        {"code": "A", "canonical_skills": ""},
        {"code": "B", "canonical_skills": "Python"},
    ])
    recs = recommend_by_cosine(courses, {"Python": 1.0}, top_k=2)
    # A has no skills, should rank last (or have 0 score)
    assert recs[0][0] == "B"
