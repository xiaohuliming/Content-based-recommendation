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
  POST /api/transcript      multipart file upload → {major, year, gpa,
                            completed_courses[], current_courses[]}
"""
import copy
import os
import pickle
import secrets
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

from src.demo.transcript import extract_transcript_text, parse_transcript_with_llm
from src.demo.transient_student import add_transient_student
from src.ppr.engine import (
    find_similar_careers,
    recommend_courses,
    summarize_student_skills,
)
from src.ppr.explain import explain_recommendation, format_explanation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "output" / "phase2" / "graph.pkl"
PROTOTYPE_HTML = PROJECT_ROOT / "prototype" / "SkillPath-Demo.html"


def create_app(graph_path: str | None = None) -> Flask:
    load_dotenv()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB transcript cap
    g_path = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
    g: nx.DiGraph = pickle.loads(g_path.read_bytes())
    app.config["GRAPH"] = g

    # Lazy LLM client — only built if /api/transcript is hit. Avoids requiring
    # LLM_API_KEY for the basic /api/recommend flow.
    _llm_state = {"client": None, "model": None}

    def _get_llm():
        if _llm_state["client"] is None:
            api_key = os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
            model = os.getenv("LLM_MODEL", "deepseek-chat")
            if not api_key:
                return None, None
            _llm_state["client"] = OpenAI(api_key=api_key, base_url=base_url)
            _llm_state["model"] = model
        return _llm_state["client"], _llm_state["model"]

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

    @app.post("/api/dashboard")
    def dashboard():
        """One-shot bundle for the comprehensive dashboard view.

        Body: same as /api/recommend — {career_goal, completed_courses[], top_k=10}.
        Returns: recommendations + similar_careers + student_top_skills + graph_stats.
        """
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

        g_copy = copy.copy(g)
        sid = add_transient_student(
            g_copy,
            completed_courses=completed,
            student_id=f"DEMO_{secrets.token_hex(4)}",
            career_goal=career_goal,
        )

        # Recommendations + per-rec explanations (same shape as /api/recommend)
        recs = recommend_courses(g_copy, sid, career_node, top_k=top_k)
        recommendations = []
        for course_node, score in recs:
            code = course_node.removeprefix("course:")
            name = g_copy.nodes[course_node].get("name", "")
            expl = explain_recommendation(g_copy, sid, career_node, course_node, top_n=5)
            recommendations.append({
                "code": code, "name": name, "score": round(score, 6),
                "bridge_skills": expl["bridge_skills"],
                "gap_skills": expl["gap_skills_filled"],
                "explanation": format_explanation(expl, course_name=name),
            })

        # Student's top skills derived from completed courses
        student_top_skills = summarize_student_skills(g_copy, sid, top_k=12)

        # Career-graph stats (count node types)
        nodes_by_type: dict[str, int] = {}
        for _, d in g.nodes(data=True):
            t = d.get("type")
            if t:
                nodes_by_type[t] = nodes_by_type.get(t, 0) + 1

        # Alternative careers ranked by skill-vector cosine. min_postings filters
        # out the long-tail one-off recruiter titles ("000198 - W2 Only - ...") that
        # would otherwise dominate the list when the target is itself niche.
        alt_careers = find_similar_careers(
            g, career_node, top_k=5, min_shared=3, min_postings=3,
        )

        # All skills the target career needs (for the "career profile" panel)
        career_needs: list[dict] = []
        for _, skill_node, ed in g.out_edges(career_node, data=True):
            if ed.get("edge_type") == "career-skill":
                career_needs.append({
                    "skill": skill_node.removeprefix("skill:"),
                    "weight": round(ed["weight"], 4),
                })
        career_needs.sort(key=lambda r: -r["weight"])

        return jsonify({
            "student_id": sid,
            "career_goal": career_goal,
            "recommendations": recommendations,
            "student_top_skills": student_top_skills,
            "alternative_careers": alt_careers,
            "career_top_skills": career_needs[:15],
            "career_n_postings": int(g.nodes[career_node].get("n_postings", 0)),
            "graph_stats": {
                "total_nodes": g.number_of_nodes(),
                "total_edges": g.number_of_edges(),
                "n_skills": nodes_by_type.get("skill", 0),
                "n_courses": nodes_by_type.get("course", 0),
                "n_careers": nodes_by_type.get("career", 0),
                "n_students": nodes_by_type.get("student", 0),
            },
        })

    @app.post("/api/transcript")
    def transcript():
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded (use multipart field 'file')"}), 400
        upload = request.files["file"]
        if not upload.filename:
            return jsonify({"error": "empty filename"}), 400
        if not upload.filename.lower().endswith(".pdf"):
            return jsonify({"error": "only .pdf files are supported"}), 415

        try:
            text = extract_transcript_text(upload.read())
        except Exception as exc:
            return jsonify({"error": f"failed to read PDF: {exc}"}), 400

        client, model = _get_llm()
        if client is None:
            return jsonify({"error": "LLM_API_KEY not configured on server"}), 503

        parsed = parse_transcript_with_llm(text, client, model)
        parsed["raw_text_chars"] = len(text)  # debugging aid for the frontend
        return jsonify(parsed)

    return app


def main() -> None:
    app = create_app()
    print(f"Loaded graph from {DEFAULT_GRAPH_PATH}")
    print(f"  serving on http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
