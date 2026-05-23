import pandas as pd
import networkx as nx
from src.graph.builder import build_graph


def _tiny_inputs():
    courses = pd.DataFrame([
        {"code": "COMP1003", "name": "Intro CS", "canonical_skills": "Python;Algorithms",
         "prerequisites_text": "None"},
        {"code": "COMP3003", "name": "ML", "canonical_skills": "Python;Machine Learning",
         "prerequisites_text": "COMP1003"},
    ])
    jobs = pd.DataFrame([
        {"title": "ML Engineer", "canonical_skills": "Python;Machine Learning;TensorFlow"},
        {"title": "Data Engineer", "canonical_skills": "Python;SQL"},
    ])
    students = pd.DataFrame([
        {"student_id": "S0001", "completed_courses": "COMP1003", "career_goal": "ML Engineer"},
    ])
    skills = pd.DataFrame([
        {"canonical_name": s, "doc_frequency": 10}
        for s in ["Python", "Algorithms", "Machine Learning", "TensorFlow", "SQL"]
    ])
    return courses, jobs, students, skills


def test_graph_has_all_four_node_types():
    g = build_graph(*_tiny_inputs())
    types = {data["type"] for _, data in g.nodes(data=True)}
    assert types == {"course", "career", "skill", "student"}


def test_course_skill_edges_present():
    g = build_graph(*_tiny_inputs())
    assert g.has_edge("course:COMP1003", "skill:Python")
    assert g.has_edge("skill:Python", "course:COMP1003")


def test_career_skill_edges_present():
    g = build_graph(*_tiny_inputs())
    assert g.has_edge("career:ML Engineer", "skill:Machine Learning")


def test_prereq_creates_directed_course_edge():
    g = build_graph(*_tiny_inputs())
    assert g.has_edge("course:COMP1003", "course:COMP3003")
    assert g["course:COMP1003"]["course:COMP3003"].get("edge_type") == "prereq"


def test_student_edges_include_completed_and_implied_skills():
    g = build_graph(*_tiny_inputs())
    assert g.has_edge("student:S0001", "course:COMP1003")
    assert g.has_edge("student:S0001", "skill:Python")


def test_skills_not_in_canonical_universe_skipped():
    courses, jobs, students, skills = _tiny_inputs()
    skills_pruned = skills[skills["canonical_name"] != "TensorFlow"]
    g = build_graph(courses, jobs, students, skills_pruned)
    assert not g.has_node("skill:TensorFlow")


def test_career_node_created_even_if_no_skills_survive():
    """Job titles with no valid skills (all pruned) should still get career nodes,
    so students targeting them don't trigger ValueError in PPR.
    """
    courses, jobs, students, skills = _tiny_inputs()
    # All ML Engineer's skills are still in the canonical universe, but let's
    # add a job title whose ONLY skill gets pruned.
    jobs = pd.concat([jobs, pd.DataFrame([
        {"title": "Underwater Welder", "canonical_skills": "Underwater Welding"},
    ])], ignore_index=True)
    # "Underwater Welding" is NOT in the skills DataFrame → effectively pruned
    g = build_graph(courses, jobs, students, skills)
    assert g.has_node("career:Underwater Welder"), \
        "Career node must exist even when all its skills were pruned"
    assert g.nodes["career:Underwater Welder"].get("type") == "career"


def test_course_skill_weight_includes_idf():
    """Course-skill weights must include an IDF factor — rare skills should weigh more
    than common ones for the SAME course.
    """
    courses = pd.DataFrame([
        {"code": "C1", "name": "x", "canonical_skills": "Common;Rare",
         "prerequisites_text": "None"},
    ])
    jobs = pd.DataFrame([{"title": "X", "canonical_skills": "Common"}])
    students = pd.DataFrame([{"student_id": "S0001", "completed_courses": "", "career_goal": "X"}])
    skills = pd.DataFrame([
        {"canonical_name": "Common", "doc_frequency": 1000},
        {"canonical_name": "Rare", "doc_frequency": 2},
    ])
    g = build_graph(courses, jobs, students, skills)
    w_common = g["course:C1"]["skill:Common"]["weight"]
    w_rare = g["course:C1"]["skill:Rare"]["weight"]
    # Rare skill (low doc_freq) should weigh more than common skill (high doc_freq)
    assert w_rare > w_common, f"IDF not applied: w_rare={w_rare:.4f}, w_common={w_common:.4f}"
