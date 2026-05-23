import networkx as nx
import pytest
from src.ppr.engine import recommend_courses, build_personalization_vector


def _toy_graph():
    g = nx.DiGraph()
    # 3 courses, 2 skills, 1 career, 1 student
    for n, t in [
        ("course:A", "course"), ("course:B", "course"), ("course:C", "course"),
        ("skill:Python", "skill"), ("skill:SQL", "skill"),
        ("career:DE", "career"), ("student:S1", "student"),
    ]:
        g.add_node(n, type=t)
    # A→Python, B→Python+SQL, C→SQL
    for c, skills in [("A", ["Python"]), ("B", ["Python", "SQL"]), ("C", ["SQL"])]:
        for s in skills:
            g.add_edge(f"course:{c}", f"skill:{s}", weight=1.0, edge_type="course-skill")
            g.add_edge(f"skill:{s}", f"course:{c}", weight=1.0, edge_type="skill-course")
    # Career DE needs SQL much more than Python
    g.add_edge("career:DE", "skill:SQL", weight=0.8, edge_type="career-skill")
    g.add_edge("skill:SQL", "career:DE", weight=0.8, edge_type="skill-career")
    g.add_edge("career:DE", "skill:Python", weight=0.2, edge_type="career-skill")
    g.add_edge("skill:Python", "career:DE", weight=0.2, edge_type="skill-career")
    # Student S1 has completed nothing yet
    g.add_node("student:S1", type="student", career_goal="DE")
    return g


def test_personalization_vector_seeds_student_and_career():
    g = _toy_graph()
    pv = build_personalization_vector(g, student_node="student:S1", career_node="career:DE")
    assert pv["student:S1"] == pytest.approx(0.5)
    assert pv["career:DE"] == pytest.approx(0.5)
    assert pv["course:A"] == 0.0
    assert sum(pv.values()) == pytest.approx(1.0)


def test_recommend_returns_only_course_nodes():
    g = _toy_graph()
    recs = recommend_courses(g, student_node="student:S1", career_node="career:DE", top_k=3)
    for code, score in recs:
        assert code.startswith("course:")
        assert score > 0


def test_de_career_prefers_sql_courses():
    """Student wanting DE should get SQL-heavy course C ahead of Python-only course A."""
    g = _toy_graph()
    recs = recommend_courses(g, student_node="student:S1", career_node="career:DE", top_k=3)
    rec_codes = [r[0] for r in recs]
    # C (SQL only) should rank above A (Python only) because DE weights SQL higher
    assert rec_codes.index("course:C") < rec_codes.index("course:A")


def test_completed_courses_are_excluded_from_recommendations():
    g = _toy_graph()
    g.add_edge("student:S1", "course:A", weight=1.0, edge_type="student-course")
    recs = recommend_courses(g, student_node="student:S1", career_node="career:DE", top_k=3,
                             exclude_completed=True)
    assert all(r[0] != "course:A" for r in recs)


def test_no_career_falls_back_to_student_only():
    g = _toy_graph()
    recs = recommend_courses(g, student_node="student:S1", career_node=None, top_k=2)
    assert len(recs) == 2
    assert all(r[0].startswith("course:") for r in recs)


def test_orphan_career_still_produces_recommendations():
    """A career node with no skill edges should not crash PPR — Task 8 hits this often."""
    g = _toy_graph()
    # Add a career with NO skill edges
    g.add_node("career:Orphan", type="career")
    recs = recommend_courses(g, "student:S1", "career:Orphan", top_k=2)
    assert len(recs) == 2
    assert all(r[0].startswith("course:") for r in recs)
