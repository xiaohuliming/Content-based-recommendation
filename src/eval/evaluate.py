"""Phase 2 evaluation: skill coverage, intra-list diversity, 5 case studies."""
import pickle
from pathlib import Path

import pandas as pd

from src.eval.baseline import recommend_by_cosine
from src.ppr.engine import recommend_courses
from src.ppr.explain import explain_recommendation, format_explanation


def skill_coverage_at_k(
    recommended_codes: list[str],
    course_skills_map: dict[str, set[str]],
    career_needed_skills: set[str],
) -> float:
    if not career_needed_skills:
        return 0.0
    covered: set[str] = set()
    for code in recommended_codes:
        covered.update(course_skills_map.get(code, set()))
    return len(covered & career_needed_skills) / len(career_needed_skills)


def intra_list_diversity(
    recommended_codes: list[str],
    course_skills_map: dict[str, set[str]],
) -> float:
    if len(recommended_codes) < 2:
        return 0.0
    distances = []
    for i, ci in enumerate(recommended_codes):
        for cj in recommended_codes[i + 1:]:
            si, sj = course_skills_map.get(ci, set()), course_skills_map.get(cj, set())
            union = si | sj
            if not union:
                continue
            distances.append(1 - len(si & sj) / len(union))
    return sum(distances) / len(distances) if distances else 0.0


def _has(s, *keywords):
    """True if student's career_goal contains any of the keywords (case-insensitive)."""
    goal = str(s.get("career_goal", "")).lower()
    return any(k.lower() in goal for k in keywords)


# Five case-study students chosen for narrative variety
CASE_STUDY_PROFILES = [
    {
        "profile": "CS junior aiming for an engineering / data role",
        "filter": lambda s: (
            s["major_prefix"] in {"COMP", "AI", "DS", "STAT"}
            and s["year"] == 3
            and _has(s, "Engineer", "Developer", "Data Scientist", "Programmer", "Architect")
        ),
    },
    {
        "profile": "Business sophomore aiming for analyst / consultant",
        "filter": lambda s: (
            s["major_prefix"] in {"BUS", "ECON", "ACCT", "FIN"}
            and s["year"] == 2
            and _has(s, "Analyst", "Consultant", "Associate", "Manager", "Accountant", "Generalist")
        ),
    },
    {
        "profile": "Freshman explorer (any major, any goal)",
        "filter": lambda s: s["year"] == 1,
    },
    {
        "profile": "Senior pivoting to a management role",
        "filter": lambda s: (
            s["year"] == 4
            and _has(s, "Manager", "Director", "Lead", "Supervisor")
        ),
    },
    {
        "profile": "Humanities student exploring an analyst / writer role",
        "filter": lambda s: (
            s["major_prefix"] in {"TRA", "MCOM", "ENG"}
            and _has(s, "Analyst", "Writer", "Editor", "Marketing", "Specialist", "Coordinator")
        ),
    },
]


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PHASE2 = PROJECT_ROOT / "output" / "phase2"

    print("Loading graph + data...")
    g = pickle.loads((PHASE2 / "graph.pkl").read_bytes())
    students = pd.read_csv(PHASE2 / "students_clustered.csv")
    courses = pd.read_csv(PHASE2 / "courses_canonical.csv")

    course_skills_map: dict[str, set[str]] = {}
    for _, row in courses.iterrows():
        raw = row.get("canonical_skills")
        cell = str(raw) if (raw is not None and raw == raw) else ""  # guard against NaN
        course_skills_map[row["code"]] = {s.strip() for s in cell.split(";") if s.strip()}
    course_name_map = dict(zip(courses["code"], courses["name"]))

    # Build career -> top skills map for baseline + coverage
    career_skills_map: dict[str, dict[str, float]] = {}
    for n, d in g.nodes(data=True):
        if d.get("type") != "career":
            continue
        title = n.removeprefix("career:")
        career_skills_map[title] = {}
        for _, skill_node, ed in g.out_edges(n, data=True):
            if ed.get("edge_type") == "career-skill":
                career_skills_map[title][skill_node.removeprefix("skill:")] = ed["weight"]

    # === Case studies ===
    cases = []
    for profile in CASE_STUDY_PROFILES:
        matches = students[students.apply(profile["filter"], axis=1)]
        if matches.empty:
            print(f"  no student matched profile: {profile['profile']}")
            continue
        student_row = matches.iloc[0]
        sid = student_row["student_id"]
        career = student_row["career_goal"]
        student_node = f"student:{sid}"
        career_node = f"career:{career}" if g.has_node(f"career:{career}") else None

        ppr_recs = recommend_courses(g, student_node, career_node, top_k=10)
        ppr_codes = [n.removeprefix("course:") for n, _ in ppr_recs]

        # Pull MORE candidates so we still have 10 after excluding completed
        cosine_recs = recommend_by_cosine(courses, career_skills_map.get(career, {}), top_k=50)
        # Same exclude-completed treatment as PPR for fair comparison
        completed_set = set()
        if isinstance(student_row.get("completed_courses"), str):
            completed_set = {c.strip() for c in student_row["completed_courses"].split(";") if c.strip()}
        cosine_codes = [code for code, _ in cosine_recs if code not in completed_set][:10]

        career_needed = set(career_skills_map.get(career, {}).keys())
        ppr_cov = skill_coverage_at_k(ppr_codes, course_skills_map, career_needed)
        cosine_cov = skill_coverage_at_k(cosine_codes, course_skills_map, career_needed)
        ppr_div = intra_list_diversity(ppr_codes, course_skills_map)
        cosine_div = intra_list_diversity(cosine_codes, course_skills_map)

        # Build explanation for top PPR rec
        explanation_text = ""
        if ppr_recs and career_node is not None:
            top_course_node = ppr_recs[0][0]
            expl = explain_recommendation(g, student_node, career_node, top_course_node, top_n=5)
            explanation_text = format_explanation(
                expl, course_name=course_name_map.get(top_course_node.removeprefix("course:"), "")
            )

        cases.append({
            "profile": profile["profile"],
            "student_id": sid,
            "major": student_row["major_prefix"],
            "year": int(student_row["year"]),
            "career_goal": career,
            "ppr_top10": ppr_codes,
            "cosine_top10": cosine_codes,
            "ppr_skill_coverage": round(ppr_cov, 3),
            "cosine_skill_coverage": round(cosine_cov, 3),
            "ppr_diversity": round(ppr_div, 3),
            "cosine_diversity": round(cosine_div, 3),
            "explanation_top1": explanation_text,
        })

    # Write CSV
    pd.DataFrame(cases).to_csv(PHASE2 / "case_recommendations.csv", index=False)
    print(f"wrote case_recommendations.csv ({len(cases)} cases)")

    # Write Markdown
    md_lines = ["# Phase 2 Case Studies\n"]
    for c in cases:
        md_lines.append(f"## {c['profile']}")
        md_lines.append(f"- **Student**: `{c['student_id']}` ({c['major']}, year {c['year']})")
        md_lines.append(f"- **Career goal**: {c['career_goal']}")
        md_lines.append(f"- **PPR coverage**: {c['ppr_skill_coverage']}  vs  **Cosine baseline**: {c['cosine_skill_coverage']}")
        md_lines.append(f"- **PPR diversity**: {c['ppr_diversity']}  vs  **Cosine baseline**: {c['cosine_diversity']}")
        md_lines.append(f"\n**PPR top-10**: {', '.join(c['ppr_top10'])}")
        md_lines.append(f"\n**Cosine baseline top-10**: {', '.join(c['cosine_top10'])}")
        if c["explanation_top1"]:
            md_lines.append(f"\n**Why we recommend the top PPR pick**: {c['explanation_top1']}")
        md_lines.append("")
    (PHASE2 / "case_studies.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote case_studies.md")

    # Aggregate metrics
    if cases:
        avg_ppr_cov = sum(c["ppr_skill_coverage"] for c in cases) / len(cases)
        avg_cos_cov = sum(c["cosine_skill_coverage"] for c in cases) / len(cases)
        print(f"\nAvg skill coverage @10 -- PPR: {avg_ppr_cov:.3f}  Cosine: {avg_cos_cov:.3f}")


if __name__ == "__main__":
    main()
