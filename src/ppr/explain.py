"""Explanation generation for course recommendations.

A "good" explanation for "we recommend course C to student S for career goal G" is:
  - The bridge skills: skills C teaches that G needs (ranked by combined relevance)
  - The gap skills filled: bridge skills S doesn't already have

This lets a human read "Course X covers Python and Machine Learning, both needed
for ML Engineer roles, and you haven't taken Machine Learning yet" — which is the
project's pitch.
"""
import networkx as nx


def explain_recommendation(
    g: nx.DiGraph,
    student_node: str,
    career_node: str,
    course_node: str,
    *,
    top_n: int = 5,
) -> dict:
    """Trace 2-hop paths Course → Skill → Career, ranked by edge-weight product."""
    course_skills: dict[str, float] = {}
    for _, skill_node, d in g.out_edges(course_node, data=True):
        if d.get("edge_type") == "course-skill":
            course_skills[skill_node] = d["weight"]

    career_skills: dict[str, float] = {}
    for _, skill_node, d in g.out_edges(career_node, data=True):
        if d.get("edge_type") == "career-skill":
            career_skills[skill_node] = d["weight"]

    student_skills: set[str] = set()
    for _, skill_node, d in g.out_edges(student_node, data=True):
        if d.get("edge_type") == "student-skill":
            student_skills.add(skill_node)

    bridges = []
    for skill_node, course_w in course_skills.items():
        if skill_node in career_skills:
            score = course_w * career_skills[skill_node]
            label = skill_node.removeprefix("skill:")
            bridges.append({
                "skill": label,
                "course_weight": round(course_w, 4),
                "career_weight": round(career_skills[skill_node], 4),
                "score": round(score, 4),
                "is_gap": skill_node not in student_skills,
            })
    bridges.sort(key=lambda b: -b["score"])

    return {
        "bridge_skills": bridges[:top_n],
        "gap_skills_filled": [b["skill"] for b in bridges[:top_n] if b["is_gap"]],
    }


def format_explanation(expl: dict, course_name: str = "") -> str:
    """Convert the structured explanation into a 1–3 sentence human-readable string."""
    bridges = expl["bridge_skills"]
    gaps = expl["gap_skills_filled"]
    if not bridges:
        return f"{course_name} has no overlapping skills with your career goal."
    skill_phrase = ", ".join(b["skill"] for b in bridges[:3])
    lead = f"{course_name} covers {skill_phrase} — all relevant to your target career."
    if gaps:
        gap_phrase = ", ".join(gaps[:3])
        lead += f" In particular, you haven't built {gap_phrase} yet."
    return lead
