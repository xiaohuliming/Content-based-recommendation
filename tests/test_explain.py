import networkx as nx
import pytest
from src.ppr.explain import explain_recommendation, format_explanation


def _toy():
    g = nx.DiGraph()
    g.add_node("student:S1", type="student")
    g.add_node("career:ML Eng", type="career")
    g.add_node("course:C1", type="course", name="Machine Learning")
    g.add_node("skill:Python", type="skill")
    g.add_node("skill:Machine Learning", type="skill")
    # Edges
    g.add_edge("course:C1", "skill:Python", weight=0.7, edge_type="course-skill")
    g.add_edge("course:C1", "skill:Machine Learning", weight=0.7, edge_type="course-skill")
    g.add_edge("skill:Python", "career:ML Eng", weight=0.3, edge_type="skill-career")
    g.add_edge("skill:Machine Learning", "career:ML Eng", weight=0.6, edge_type="skill-career")
    g.add_edge("career:ML Eng", "skill:Python", weight=0.3, edge_type="career-skill")
    g.add_edge("career:ML Eng", "skill:Machine Learning", weight=0.6, edge_type="career-skill")
    g.add_edge("student:S1", "skill:Python", weight=0.5, edge_type="student-skill")
    return g


def test_explain_returns_bridge_skills_ranked():
    g = _toy()
    expl = explain_recommendation(g, "student:S1", "career:ML Eng", "course:C1", top_n=3)
    skills = [item["skill"] for item in expl["bridge_skills"]]
    # ML should rank above Python because career weights ML higher
    assert skills.index("Machine Learning") < skills.index("Python")


def test_explain_identifies_gap_skills():
    """Skills the career needs that the student doesn't have."""
    g = _toy()
    expl = explain_recommendation(g, "student:S1", "career:ML Eng", "course:C1", top_n=3)
    # S1 has Python but not ML → ML is a gap the course fills
    gaps = expl["gap_skills_filled"]
    assert "Machine Learning" in gaps
    assert "Python" not in gaps


def test_format_explanation_is_human_readable():
    g = _toy()
    expl = explain_recommendation(g, "student:S1", "career:ML Eng", "course:C1", top_n=3)
    text = format_explanation(expl, course_name="Machine Learning")
    assert "Machine Learning" in text
    assert "career" in text.lower() or "ML Eng" in text
