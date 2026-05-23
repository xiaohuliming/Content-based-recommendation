import networkx as nx
import pickle
import pytest

from src.ppr.engine import find_similar_careers, summarize_student_skills


def _toy_graph_with_three_careers():
    g = nx.DiGraph()
    for n, t in [
        ("skill:Python", "skill"), ("skill:SQL", "skill"),
        ("skill:ML", "skill"), ("skill:Marketing", "skill"),
    ]:
        g.add_node(n, type=t)
    # Careers carry n_postings so they pass the default min_postings filter
    for c in ["career:Data Engineer", "career:Data Scientist", "career:Marketing Manager"]:
        g.add_node(c, type="career", n_postings=5)

    def _add_career(c, edges):
        for s, w in edges:
            g.add_edge(c, s, weight=w, edge_type="career-skill")
            g.add_edge(s, c, weight=w, edge_type="skill-career")

    _add_career("career:Data Engineer", [
        ("skill:Python", 0.4), ("skill:SQL", 0.5), ("skill:ML", 0.1),
    ])
    _add_career("career:Data Scientist", [
        ("skill:Python", 0.5), ("skill:SQL", 0.3), ("skill:ML", 0.2),
    ])
    _add_career("career:Marketing Manager", [
        ("skill:Marketing", 1.0),
    ])
    return g


def test_similar_careers_ranks_overlap_highest():
    g = _toy_graph_with_three_careers()
    out = find_similar_careers(g, "career:Data Engineer", top_k=5, min_shared=2)
    titles = [c["title"] for c in out]
    # Data Scientist shares 3 skills with DE; Marketing Manager shares 0.
    assert titles[0] == "Data Scientist"
    assert "Marketing Manager" not in titles  # zero overlap → filtered


def test_similar_careers_excludes_target_itself():
    g = _toy_graph_with_three_careers()
    out = find_similar_careers(g, "career:Data Engineer", top_k=5, min_shared=1)
    assert "Data Engineer" not in [c["title"] for c in out]


def test_similar_careers_returns_skill_count_and_sample():
    g = _toy_graph_with_three_careers()
    out = find_similar_careers(g, "career:Data Engineer", top_k=1, min_shared=2)
    assert out[0]["shared_skill_count"] == 3
    assert "Python" in out[0]["shared_skills_sample"]
    assert "SQL" in out[0]["shared_skills_sample"]


def test_similar_careers_handles_target_with_no_edges():
    g = nx.DiGraph()
    g.add_node("career:Orphan", type="career")
    assert find_similar_careers(g, "career:Orphan") == []


def test_summarize_student_skills_returns_sorted_top_k():
    g = nx.DiGraph()
    g.add_node("student:S1", type="student")
    for s in ["A", "B", "C"]:
        g.add_node(f"skill:{s}", type="skill")
    g.add_edge("student:S1", "skill:A", weight=0.3, edge_type="student-skill")
    g.add_edge("student:S1", "skill:B", weight=0.7, edge_type="student-skill")
    g.add_edge("student:S1", "skill:C", weight=0.5, edge_type="student-skill")
    # Also a non-student-skill edge that should be ignored
    g.add_edge("student:S1", "skill:A", weight=0.99, edge_type="student-course")  # different type
    out = summarize_student_skills(g, "student:S1", top_k=2)
    assert len(out) == 2
    assert out[0]["skill"] == "B"
    assert out[1]["skill"] == "C"


def test_dashboard_endpoint_returns_all_sections(tmp_path):
    from src.demo.app import create_app
    g = _toy_graph_with_three_careers()
    g.add_node("course:C1", type="course", name="Intro")
    g.add_edge("course:C1", "skill:Python", weight=0.8, edge_type="course-skill")
    g.add_edge("skill:Python", "course:C1", weight=0.8, edge_type="skill-course")
    pkl = tmp_path / "g.pkl"
    pkl.write_bytes(pickle.dumps(g))

    app = create_app(graph_path=str(pkl))
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.post("/api/dashboard", json={
        "career_goal": "Data Engineer",
        "completed_courses": [],
        "top_k": 5,
    })
    assert r.status_code == 200
    body = r.get_json()
    for key in ["recommendations", "student_top_skills", "alternative_careers",
                "career_top_skills", "career_n_postings", "graph_stats"]:
        assert key in body, f"missing key: {key}"
    assert isinstance(body["graph_stats"]["n_careers"], int)
    assert body["graph_stats"]["n_careers"] >= 3


def test_dashboard_endpoint_400_on_missing_career(tmp_path):
    from src.demo.app import create_app
    g = _toy_graph_with_three_careers()
    pkl = tmp_path / "g.pkl"
    pkl.write_bytes(pickle.dumps(g))
    app = create_app(graph_path=str(pkl))
    client = app.test_client()
    r = client.post("/api/dashboard", json={"completed_courses": []})
    assert r.status_code == 400


def test_dashboard_endpoint_404_on_unknown_career(tmp_path):
    from src.demo.app import create_app
    g = _toy_graph_with_three_careers()
    pkl = tmp_path / "g.pkl"
    pkl.write_bytes(pickle.dumps(g))
    app = create_app(graph_path=str(pkl))
    client = app.test_client()
    r = client.post("/api/dashboard", json={"career_goal": "Nope", "completed_courses": []})
    assert r.status_code == 404
