"""Insert a transient student into an existing graph for live PPR recommendation.

Mirrors the student-edge logic from src.graph.builder but for one student at a time,
without rebuilding the whole graph. Used by the Flask demo's /api/recommend endpoint.
"""
from collections import defaultdict

import networkx as nx


def add_transient_student(
    g: nx.DiGraph,
    completed_courses: list[str],
    *,
    student_id: str,
    career_goal: str = "",
    year: int = 0,
) -> str:
    """Add a student node + student-course + student-skill (normalized) edges in place.

    Returns the full node id (e.g. 'student:DEMO'). Unknown courses are silently
    skipped (no error). Raises ValueError if `student_id` is already in the graph.
    """
    sid = f"student:{student_id}"
    if g.has_node(sid):
        raise ValueError(f"student node {sid!r} already in graph — pick a unique id")

    g.add_node(sid, type="student", career_goal=career_goal, year=year)

    student_skill_score: dict[str, float] = defaultdict(float)
    for code in completed_courses:
        course_node = f"course:{code}"
        if not g.has_node(course_node):
            continue
        g.add_edge(sid, course_node, weight=1.0, edge_type="student-course")
        for skill_node in g.successors(course_node):
            if g.nodes[skill_node].get("type") == "skill":
                student_skill_score[skill_node] += g[course_node][skill_node]["weight"]

    if student_skill_score:
        max_w = max(student_skill_score.values())
        if max_w > 0:
            student_skill_score = {k: v / max_w for k, v in student_skill_score.items()}

    for skill_node, w in student_skill_score.items():
        g.add_edge(sid, skill_node, weight=w, edge_type="student-skill")

    return sid
