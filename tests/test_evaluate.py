import pandas as pd
from src.eval.evaluate import skill_coverage_at_k, intra_list_diversity


def test_skill_coverage_at_k():
    course_skills_map = {
        "A": {"Python", "SQL"},
        "B": {"ML", "TensorFlow"},
        "C": {"Marketing"},
    }
    career_needed = {"Python", "SQL", "ML", "TensorFlow", "Spark"}
    # Top-2 [A, B] covers 4/5
    cov = skill_coverage_at_k(["A", "B"], course_skills_map, career_needed)
    assert cov == 0.8


def test_skill_coverage_with_overlap():
    """Coverage shouldn't double-count a skill that appears in multiple top-K courses."""
    course_skills_map = {
        "A": {"Python", "SQL"},
        "B": {"Python", "ML"},
    }
    career_needed = {"Python", "SQL", "ML"}
    # Union = {Python, SQL, ML} = 3/3 = 1.0
    assert skill_coverage_at_k(["A", "B"], course_skills_map, career_needed) == 1.0


def test_intra_list_diversity_max_when_no_overlap():
    course_skills_map = {
        "A": {"Python"},
        "B": {"Marketing"},
    }
    # Jaccard distance = 1 - 0/2 = 1.0
    div = intra_list_diversity(["A", "B"], course_skills_map)
    assert div == 1.0


def test_intra_list_diversity_zero_when_identical():
    course_skills_map = {
        "A": {"Python", "SQL"},
        "B": {"Python", "SQL"},
    }
    assert intra_list_diversity(["A", "B"], course_skills_map) == 0.0
