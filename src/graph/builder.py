"""Build the heterogeneous SkillPath graph.

Node types:
  course:<CODE>     UIC courses
  career:<TITLE>    Job titles (from LinkedIn sample)
  skill:<NAME>      Canonical skills (post-LSH)
  student:<ID>      Synthetic students

Edge types (all stored with edge_type=... attribute):
  course-skill   weight = length_norm * IDF(skill); both directions
  career-skill   weighted by freq / total
  student-course 1.0 (completed)
  student-skill  derived sum of course-skill weights
  prereq         directed course→course (only direction kept)
"""
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

from src.graph.prereq_parser import parse_prerequisites


def _split_semis(cell) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [s.strip() for s in cell.split(";") if s.strip()]


def build_graph(
    courses: pd.DataFrame,
    jobs: pd.DataFrame,
    students: pd.DataFrame,
    skills: pd.DataFrame,
) -> nx.DiGraph:
    """Construct the heterogeneous graph from canonical Phase 2 inputs."""
    g = nx.DiGraph()
    valid_skills = set(skills["canonical_name"])

    # Precompute IDF for each skill (used in course-skill edge weights)
    n_docs = len(courses) + len(jobs)  # total documents skills can appear in
    skill_idf: dict[str, float] = {
        row["canonical_name"]: math.log(n_docs / max(1, row["doc_frequency"]))
        for _, row in skills.iterrows()
    }

    # Add skill nodes
    for _, row in skills.iterrows():
        g.add_node(f"skill:{row['canonical_name']}", type="skill",
                   doc_frequency=int(row["doc_frequency"]))

    # Add course nodes + course→skill (and reverse) edges
    for _, row in courses.iterrows():
        code = row["code"]
        g.add_node(f"course:{code}", type="course", name=row.get("name", ""))
        course_skills = [s for s in _split_semis(row.get("canonical_skills")) if s in valid_skills]
        if course_skills:
            length_norm = 1.0 / math.sqrt(len(course_skills))
            for s in course_skills:
                w = length_norm * skill_idf.get(s, 0.0)
                g.add_edge(f"course:{code}", f"skill:{s}", weight=w, edge_type="course-skill")
                g.add_edge(f"skill:{s}", f"course:{code}", weight=w, edge_type="skill-course")

    # Add career nodes (one per unique job title, freq-weighted skills)
    career_skill_counts: dict[str, Counter] = defaultdict(Counter)
    career_doc_counts: Counter = Counter()
    for _, row in jobs.iterrows():
        title = row["title"]
        career_doc_counts[title] += 1
        for s in _split_semis(row.get("canonical_skills")):
            if s in valid_skills:
                career_skill_counts[title][s] += 1
    for title, skill_counter in career_skill_counts.items():
        g.add_node(f"career:{title}", type="career", n_postings=int(career_doc_counts[title]))
        total = sum(skill_counter.values())
        if total == 0:
            continue
        for s, c in skill_counter.items():
            w = c / total
            g.add_edge(f"career:{title}", f"skill:{s}", weight=w, edge_type="career-skill")
            g.add_edge(f"skill:{s}", f"career:{title}", weight=w, edge_type="skill-career")

    # Ensure every observed job title has a career node, even if all its skills got pruned by Task 2.
    # Without this, Task 5's PPR raises ValueError when a student targets a "skill-less" career.
    for title, count in career_doc_counts.items():
        if f"career:{title}" not in g:
            g.add_node(f"career:{title}", type="career", n_postings=int(count))

    # Add students + student→course, student→skill (derived from completed courses)
    for _, row in students.iterrows():
        sid = row["student_id"]
        year_val = row.get("year")
        year_int = int(year_val) if not pd.isna(year_val) else 0
        g.add_node(f"student:{sid}", type="student",
                   career_goal=row.get("career_goal"),
                   year=year_int)
        completed = _split_semis(row.get("completed_courses"))
        student_skill_score: dict[str, float] = defaultdict(float)
        for c in completed:
            course_node = f"course:{c}"
            if g.has_node(course_node):
                g.add_edge(f"student:{sid}", course_node, weight=1.0, edge_type="student-course")
                for skill_node in g.successors(course_node):
                    if g.nodes[skill_node].get("type") == "skill":
                        edge_w = g[course_node][skill_node]["weight"]
                        student_skill_score[skill_node] += edge_w
        for skill_node, w in student_skill_score.items():
            g.add_edge(f"student:{sid}", skill_node, weight=w, edge_type="student-skill")

    # Prerequisite edges (course → course, directed only)
    for _, row in courses.iterrows():
        code = row["code"]
        prereqs = parse_prerequisites(str(row.get("prerequisites_text", "") or ""))
        for p in prereqs:
            if g.has_node(f"course:{p}"):
                g.add_edge(f"course:{p}", f"course:{code}", weight=0.5, edge_type="prereq")

    return g


def graph_stats(g: nx.DiGraph) -> dict:
    by_type = Counter(d.get("type") for _, d in g.nodes(data=True))
    edges_by_type = Counter(d.get("edge_type") for _, _, d in g.edges(data=True))
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "nodes_by_type": dict(by_type),
        "edges_by_type": dict(edges_by_type),
        "is_weakly_connected": nx.is_weakly_connected(g),
        "n_weakly_connected_components": nx.number_weakly_connected_components(g),
    }


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PHASE2 = PROJECT_ROOT / "output" / "phase2"

    print("Loading inputs...")
    courses = pd.read_csv(PHASE2 / "courses_canonical.csv")
    jobs = pd.read_csv(PHASE2 / "jobs_canonical.csv")
    students = pd.read_csv(PHASE2 / "students.csv")
    skills = pd.read_csv(PHASE2 / "skills_canonical.csv")

    print(f"  {len(courses)} courses, {len(jobs)} jobs, {len(students)} students, {len(skills)} skills")
    print("Building graph...")
    g = build_graph(courses, jobs, students, skills)

    stats = graph_stats(g)
    print(f"  nodes: {stats['n_nodes']:,} ({stats['nodes_by_type']})")
    print(f"  edges: {stats['n_edges']:,} ({stats['edges_by_type']})")
    print(f"  weakly connected: {stats['is_weakly_connected']} "
          f"({stats['n_weakly_connected_components']} components)")

    (PHASE2 / "graph.pkl").write_bytes(pickle.dumps(g))
    (PHASE2 / "graph_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote graph.pkl ({(PHASE2 / 'graph.pkl').stat().st_size/1024/1024:.1f} MB) + graph_stats.json")


if __name__ == "__main__":
    main()
