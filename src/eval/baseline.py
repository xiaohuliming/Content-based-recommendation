"""Content-based cosine-similarity recommender.

This is the honest baseline: 'what would we get if we just matched
skill vectors directly, without any graph structure?' Shows whether
PPR is actually doing useful work.
"""
import math
from typing import Mapping

import pandas as pd


def recommend_by_cosine(
    courses: pd.DataFrame,
    career_skills: Mapping[str, float],
    *,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Rank courses by cosine similarity between their skill set and the career's weighted skill profile."""
    career_norm = math.sqrt(sum(w * w for w in career_skills.values()))
    if career_norm == 0:
        return []

    scored: list[tuple[str, float]] = []
    for _, row in courses.iterrows():
        raw = row.get("canonical_skills")
        cell = str(raw) if (raw is not None and raw == raw) else ""  # guard against NaN float
        course_skills = [s.strip() for s in cell.split(";") if s.strip()]
        if not course_skills:
            scored.append((row["code"], 0.0))
            continue
        # Course skill weights = 1.0 each, L2-normalize
        dot = sum(career_skills.get(s, 0.0) for s in course_skills)
        course_norm = math.sqrt(len(course_skills))
        scored.append((row["code"], dot / (course_norm * career_norm)))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]
