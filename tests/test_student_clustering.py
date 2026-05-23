import numpy as np
import pandas as pd
import pytest
from src.students.clustering import build_skill_matrix, cluster_students, describe_cluster


def test_skill_matrix_shape_matches_inputs():
    students = pd.DataFrame([
        {"student_id": "S1", "completed_courses": "C1;C2"},
        {"student_id": "S2", "completed_courses": "C2"},
    ])
    courses = pd.DataFrame([
        {"code": "C1", "canonical_skills": "Python;SQL"},
        {"code": "C2", "canonical_skills": "Python;ML"},
    ])
    skill_vocab = ["Python", "SQL", "ML"]
    X, idx = build_skill_matrix(students, courses, skill_vocab)
    assert X.shape == (2, 3)
    assert list(idx) == ["S1", "S2"]
    # S1 has both courses → all 3 skills
    assert (X[0] > 0).sum() == 3
    # S2 only has C2 → Python and ML (not SQL)
    assert X[1, skill_vocab.index("SQL")] == 0


def test_cluster_students_returns_deterministic_labels():
    rng = np.random.default_rng(0)
    X = rng.random((50, 10))
    a = cluster_students(X, k=3, seed=42)
    b = cluster_students(X, k=3, seed=42)
    np.testing.assert_array_equal(a, b)


def test_cluster_count_matches_request():
    rng = np.random.default_rng(0)
    X = rng.random((100, 8))
    labels = cluster_students(X, k=4, seed=1)
    assert len(set(labels)) <= 4
    assert all(0 <= l < 4 for l in labels)


def test_describe_cluster_picks_distinguishing_skills():
    """Cluster centroids - global mean → top features distinguish each cluster."""
    skill_vocab = ["Python", "SQL", "Excel", "ML"]
    # Cluster 0 is much higher on Python+ML than the rest
    centroid = np.array([0.8, 0.2, 0.1, 0.7])
    global_mean = np.array([0.3, 0.3, 0.3, 0.3])
    profile = describe_cluster(centroid, global_mean, skill_vocab, top_n=2)
    assert profile["top_distinguishing_skills"][:2] == ["Python", "ML"]
