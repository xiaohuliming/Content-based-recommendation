"""K-means clustering of students by their accumulated skill profile.

Each student → vector of skill weights derived from completed courses.
Clusters → "career archetypes" you can name (e.g., "data-leaning juniors").
Course lecture 6 alignment.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def build_skill_matrix(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    skill_vocab: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build a student × skill matrix where each row is a normalized skill vector."""
    course_skills: dict[str, list[str]] = {}
    for _, row in courses.iterrows():
        cell = row.get("canonical_skills")
        if isinstance(cell, str) and cell.strip():
            course_skills[row["code"]] = [s.strip() for s in cell.split(";") if s.strip()]
        else:
            course_skills[row["code"]] = []

    skill_idx = {s: i for i, s in enumerate(skill_vocab)}
    n = len(students)
    X = np.zeros((n, len(skill_vocab)), dtype=np.float32)
    ids = []
    for r, (_, row) in enumerate(students.iterrows()):
        ids.append(row["student_id"])
        completed_cell = row.get("completed_courses")
        completed = []
        if isinstance(completed_cell, str) and completed_cell.strip():
            completed = [c.strip() for c in completed_cell.split(";") if c.strip()]
        for c in completed:
            for s in course_skills.get(c, []):
                if s in skill_idx:
                    X[r, skill_idx[s]] += 1.0
        # L2 normalize so students with more completed courses don't dominate
        norm = np.linalg.norm(X[r])
        if norm > 0:
            X[r] /= norm
    return X, ids


def cluster_students(X: np.ndarray, k: int = 8, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(X)


def describe_cluster(
    centroid: np.ndarray,
    global_mean: np.ndarray,
    skill_vocab: list[str],
    *,
    top_n: int = 8,
) -> dict:
    """Top-N skills where this cluster's centroid is most above the global mean."""
    diff = centroid - global_mean
    top_idx = np.argsort(-diff)[:top_n]
    return {
        "top_distinguishing_skills": [skill_vocab[i] for i in top_idx],
        "centroid_strength": float(np.linalg.norm(centroid)),
    }


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PHASE2 = PROJECT_ROOT / "output" / "phase2"

    students = pd.read_csv(PHASE2 / "students.csv")
    courses = pd.read_csv(PHASE2 / "courses_canonical.csv")
    skills = pd.read_csv(PHASE2 / "skills_canonical.csv")
    skill_vocab = skills["canonical_name"].tolist()

    print(f"Building skill matrix: {len(students)} students × {len(skill_vocab)} skills...")
    X, ids = build_skill_matrix(students, courses, skill_vocab)

    k = 8
    print(f"K-means k={k}...")
    labels = cluster_students(X, k=k, seed=42)
    students_clustered = students.copy()
    students_clustered.loc[:, "cluster_id"] = labels
    students_clustered.to_csv(PHASE2 / "students_clustered.csv", index=False)

    # Profile each cluster
    global_mean = X.mean(axis=0)
    profiles: dict[int, dict] = {}
    for cid in sorted(set(labels)):
        mask = labels == cid
        if not mask.any():
            continue
        centroid = X[mask].mean(axis=0)
        profile = describe_cluster(centroid, global_mean, skill_vocab, top_n=10)
        profile["size"] = int(mask.sum())
        profiles[int(cid)] = profile
    (PHASE2 / "cluster_profiles.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2)
    )
    print(f"\nCluster sizes:")
    for cid, p in profiles.items():
        skills_str = ", ".join(p["top_distinguishing_skills"][:4])
        print(f"  {cid}: {p['size']:>3} students  ← {skills_str}")


if __name__ == "__main__":
    main()
