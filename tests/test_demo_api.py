import json
import pickle
import pytest
import networkx as nx
from src.demo.app import create_app


@pytest.fixture
def client(tmp_path):
    # Build a tiny throwaway graph and pickle it
    g = nx.DiGraph()
    g.add_node("skill:Python", type="skill", doc_frequency=10)
    g.add_node("course:C1", type="course", name="Intro")
    g.add_edge("course:C1", "skill:Python", weight=0.7, edge_type="course-skill")
    g.add_edge("skill:Python", "course:C1", weight=0.7, edge_type="skill-course")
    g.add_node("career:Dev", type="career", n_postings=5)
    g.add_edge("career:Dev", "skill:Python", weight=1.0, edge_type="career-skill")
    g.add_edge("skill:Python", "career:Dev", weight=1.0, edge_type="skill-career")
    pkl = tmp_path / "g.pkl"
    pkl.write_bytes(pickle.dumps(g))

    app = create_app(graph_path=str(pkl))
    app.config["TESTING"] = True
    return app.test_client()


def test_index_returns_prototype_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower() or b"<!doctype" in resp.data.lower()


def test_careers_endpoint_lists_career_titles(client):
    resp = client.get("/api/careers")
    data = resp.get_json()
    assert "Dev" in data["careers"]


def test_recommend_returns_ranked_courses(client):
    payload = {
        "career_goal": "Dev",
        "completed_courses": [],
        "top_k": 3,
    }
    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert "recommendations" in body
    assert body["recommendations"][0]["code"] == "C1"
    assert "score" in body["recommendations"][0]
    assert "explanation" in body["recommendations"][0]


def test_recommend_missing_career_returns_400(client):
    resp = client.post("/api/recommend", json={"completed_courses": []})
    assert resp.status_code == 400
