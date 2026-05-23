"""Flask app serving the SkillPath demo.

Loads graph.pkl ONCE at startup. Each /api/recommend request:
  1. Shallow-copies the graph
  2. Inserts a transient student
  3. Runs PPR + explain
  4. Returns JSON

Endpoints:
  GET  /                    serve prototype HTML
  GET  /api/careers         list career titles in the graph (for frontend autocomplete)
  POST /api/recommend       body: {career_goal, completed_courses[], top_k=10}
"""
import copy
import pickle
import secrets
from pathlib import Path

import networkx as nx
from flask import Flask, jsonify, request, send_from_directory

from src.demo.transient_student import add_transient_student
from src.ppr.engine import recommend_courses
from src.ppr.explain import explain_recommendation, format_explanation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "output" / "phase2" / "graph.pkl"
PROTOTYPE_HTML = PROJECT_ROOT / "prototype" / "SkillPath-Demo.html"


def create_app(graph_path: str | None = None) -> Flask:
    app = Flask(__name__)
    g_path = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
    g: nx.DiGraph = pickle.loads(g_path.read_bytes())
    app.config["GRAPH"] = g

    @app.get("/")
    def index():
        return send_from_directory(PROTOTYPE_HTML.parent, PROTOTYPE_HTML.name)

    @app.get("/api/careers")
    def careers():
        g: nx.DiGraph = app.config["GRAPH"]
        titles = sorted(
            n.removeprefix("career:")
            for n, d in g.nodes(data=True)
            if d.get("type") == "career"
        )
        return jsonify({"careers": titles})

    @app.post("/api/recommend")
    def recommend():
        payload = request.get_json(silent=True) or {}
        career_goal = payload.get("career_goal")
        completed = payload.get("completed_courses", []) or []
        top_k = int(payload.get("top_k", 10))
        if not career_goal:
            return jsonify({"error": "career_goal required"}), 400

        g: nx.DiGraph = app.config["GRAPH"]
        career_node = f"career:{career_goal}"
        if not g.has_node(career_node):
            return jsonify({"error": f"unknown career: {career_goal}"}), 404

        g_copy = copy.copy(g)  # shallow — node/edge dicts share, modifications happen via add_*
        sid = add_transient_student(
            g_copy,
            completed_courses=completed,
            student_id=f"DEMO_{secrets.token_hex(4)}",
            career_goal=career_goal,
        )

        recs = recommend_courses(g_copy, sid, career_node, top_k=top_k)
        results = []
        for course_node, score in recs:
            code = course_node.removeprefix("course:")
            name = g_copy.nodes[course_node].get("name", "")
            expl = explain_recommendation(g_copy, sid, career_node, course_node, top_n=5)
            results.append({
                "code": code,
                "name": name,
                "score": round(score, 6),
                "bridge_skills": expl["bridge_skills"],
                "gap_skills": expl["gap_skills_filled"],
                "explanation": format_explanation(expl, course_name=name),
            })

        return jsonify({"student_id": sid, "recommendations": results})

    return app


def main() -> None:
    app = create_app()
    print(f"Loaded graph from {DEFAULT_GRAPH_PATH}")
    print(f"  serving on http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
