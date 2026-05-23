import networkx as nx
import pandas as pd
from src.demo.transient_student import add_transient_student


def _toy_graph():
    g = nx.DiGraph()
    # 2 courses, 2 skills
    g.add_node("skill:Python", type="skill", doc_frequency=10)
    g.add_node("skill:SQL", type="skill", doc_frequency=10)
    g.add_node("course:C1", type="course", name="Intro")
    g.add_node("course:C2", type="course", name="DB")
    g.add_edge("course:C1", "skill:Python", weight=0.7, edge_type="course-skill")
    g.add_edge("skill:Python", "course:C1", weight=0.7, edge_type="skill-course")
    g.add_edge("course:C2", "skill:SQL", weight=0.7, edge_type="course-skill")
    g.add_edge("skill:SQL", "course:C2", weight=0.7, edge_type="skill-course")
    return g


def test_add_transient_student_creates_node():
    g = _toy_graph()
    sid = add_transient_student(g, completed_courses=["C1"], student_id="DEMO")
    assert sid == "student:DEMO"
    assert g.nodes[sid]["type"] == "student"


def test_student_completed_edges_added():
    g = _toy_graph()
    sid = add_transient_student(g, completed_courses=["C1", "C2"], student_id="DEMO")
    assert g.has_edge(sid, "course:C1")
    assert g.has_edge(sid, "course:C2")
    assert g[sid]["course:C1"]["edge_type"] == "student-course"


def test_student_skill_edges_normalized():
    g = _toy_graph()
    sid = add_transient_student(g, completed_courses=["C1", "C2"], student_id="DEMO")
    # student-skill weights should be in [0, 1] (same normalization as Task 4)
    skill_edges = [d["weight"] for _, t, d in g.out_edges(sid, data=True)
                   if d.get("edge_type") == "student-skill"]
    assert skill_edges
    assert max(skill_edges) <= 1.0


def test_unknown_completed_courses_silently_skipped():
    g = _toy_graph()
    sid = add_transient_student(g, completed_courses=["C1", "NONEXIST"], student_id="X")
    assert g.has_edge(sid, "course:C1")
    assert not g.has_edge(sid, "course:NONEXIST")


def test_duplicate_ids_raise():
    g = _toy_graph()
    add_transient_student(g, completed_courses=["C1"], student_id="DUP")
    try:
        add_transient_student(g, completed_courses=["C2"], student_id="DUP")
        assert False, "should have raised"
    except ValueError:
        pass
