"""Personalized PageRank-based course recommendation."""
import math
from typing import Optional

import networkx as nx


def build_personalization_vector(
    g: nx.DiGraph,
    student_node: str,
    career_node: Optional[str],
) -> dict[str, float]:
    """Build a personalization vector over all graph nodes.

    Mass is split between the student node and the optional career-goal node.
    If career_node is None, all mass goes to the student.
    """
    pv = {n: 0.0 for n in g.nodes()}
    if student_node not in pv:
        raise ValueError(f"student node {student_node!r} not in graph")
    if career_node is not None and career_node not in pv:
        raise ValueError(f"career node {career_node!r} not in graph")
    if career_node is None:
        pv[student_node] = 1.0
    else:
        pv[student_node] = 0.5
        pv[career_node] = 0.5
    return pv


def recommend_courses(
    g: nx.DiGraph,
    student_node: str,
    career_node: Optional[str] = None,
    *,
    top_k: int = 10,
    alpha: float = 0.85,
    exclude_completed: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> list[tuple[str, float]]:
    """Run Personalized PageRank and return top-k course nodes by score.

    If `career_node` has no outgoing skill edges (an orphan job title whose
    skills were all pruned by Task 2), PageRank still converges — the career
    mass redistributes via dangling-node handling, but recommendations will
    closely resemble the no-career fallback. Task 7's explanations will reflect
    this (few bridge skills).

    On rare graphs where the default `max_iter=100` doesn't converge, the
    function retries once with `max_iter * 2` and `tol * 10`.
    """
    pv = build_personalization_vector(g, student_node, career_node)
    try:
        scores = nx.pagerank(
            g, alpha=alpha, personalization=pv, weight="weight",
            max_iter=max_iter, tol=tol,
        )
    except nx.PowerIterationFailedConvergence:
        # Retry once with looser convergence on rare edge-case graphs
        scores = nx.pagerank(
            g, alpha=alpha, personalization=pv, weight="weight",
            max_iter=max_iter * 2, tol=tol * 10,
        )

    # Filter to course nodes
    course_scores = [(n, s) for n, s in scores.items() if n.startswith("course:")]

    if exclude_completed:
        completed = set()
        for _, target, d in g.out_edges(student_node, data=True):
            if d.get("edge_type") == "student-course":
                completed.add(target)
        course_scores = [(n, s) for n, s in course_scores if n not in completed]

    course_scores.sort(key=lambda x: -x[1])
    return course_scores[:top_k]


def find_similar_careers(
    g: nx.DiGraph,
    target_career_node: str,
    *,
    top_k: int = 5,
    min_shared: int = 3,
    min_postings: int = 1,
) -> list[dict]:
    """Find careers whose skill-weight vector is most cosine-similar to the target.

    Each career has outgoing career-skill edges with normalized weights.
    Cosine over those weight vectors gives a "do these two jobs need similar skills"
    score. `min_postings` filters out one-off recruiter listings that have a too-
    specific title to be a meaningful alternative recommendation.
    """
    target_skills: dict[str, float] = {}
    for _, skill_node, d in g.out_edges(target_career_node, data=True):
        if d.get("edge_type") == "career-skill":
            target_skills[skill_node] = d["weight"]
    if not target_skills:
        return []
    norm_target = math.sqrt(sum(w * w for w in target_skills.values()))

    scored: list[dict] = []
    for n, data in g.nodes(data=True):
        if data.get("type") != "career" or n == target_career_node:
            continue
        if int(data.get("n_postings", 0)) < min_postings:
            continue
        other_skills: dict[str, float] = {}
        for _, sn, ed in g.out_edges(n, data=True):
            if ed.get("edge_type") == "career-skill":
                other_skills[sn] = ed["weight"]
        shared = set(target_skills) & set(other_skills)
        if len(shared) < min_shared:
            continue
        dot = sum(target_skills[s] * other_skills[s] for s in shared)
        norm_other = math.sqrt(sum(w * w for w in other_skills.values()))
        if norm_other == 0:
            continue
        scored.append({
            "title": n.removeprefix("career:"),
            "similarity": round(dot / (norm_target * norm_other), 4),
            "shared_skill_count": len(shared),
            "shared_skills_sample": sorted(s.removeprefix("skill:") for s in shared)[:6],
            "n_postings": int(data.get("n_postings", 0)),
        })
    scored.sort(key=lambda x: -x["similarity"])
    return scored[:top_k]


def summarize_student_skills(
    g: nx.DiGraph,
    student_node: str,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Return the student's top-k accumulated skills (inherited from completed courses).

    Each entry: {skill, weight}. Sorted descending.
    """
    rows: list[dict] = []
    for _, skill_node, d in g.out_edges(student_node, data=True):
        if d.get("edge_type") == "student-skill":
            rows.append({
                "skill": skill_node.removeprefix("skill:"),
                "weight": round(d["weight"], 4),
            })
    rows.sort(key=lambda r: -r["weight"])
    return rows[:top_k]
