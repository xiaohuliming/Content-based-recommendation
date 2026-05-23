import pytest
from src.normalize.minhash_lsh import (
    normalize_skill_name,
    char_shingles,
    cluster_skills,
)


def test_normalize_strips_punctuation_and_lowercases():
    assert normalize_skill_name("Microsoft Office (Suite)") == "microsoft office suite"
    assert normalize_skill_name("  C++  ") == "c++"
    assert normalize_skill_name("Data-Driven Decisions") == "data driven decisions"


def test_char_shingles_produces_3grams():
    s = "abcdef"
    assert char_shingles(s, k=3) == {"abc", "bcd", "cde", "def"}


def test_short_string_handled():
    """Skills shorter than k should still produce a usable signature."""
    assert char_shingles("ml", k=3) == {"ml"}


def test_cluster_merges_substring_variants():
    """LSH+exact-verify SHOULD catch substring/suffix variants where Jaccard is genuinely high."""
    skills = [
        "Microsoft Office",
        "Microsoft Office Suite",
        "Microsoft Office 365",
        "Python",
        "Machine Learning",
        "Java",
    ]
    clusters = cluster_skills(skills, threshold=0.4, verify_threshold=0.7, num_perm=128, seed=42)
    by_skill = {s: cid for cid, members in clusters.items() for s in members}
    # The three Microsoft Office variants should merge into one cluster
    office_cids = {by_skill["Microsoft Office"], by_skill["Microsoft Office Suite"], by_skill["Microsoft Office 365"]}
    assert len(office_cids) == 1, f"office variants split across clusters: {office_cids}"
    # Unrelated skills must remain separate
    assert by_skill["Python"] != by_skill["Java"]
    assert by_skill["Python"] != by_skill["Microsoft Office"]
    assert by_skill["Machine Learning"] != by_skill["Python"]


def test_acronym_variants_are_NOT_merged_known_limitation():
    """Honest test: statistical char-shingle similarity CANNOT catch acronym variants.

    'Ms Office' vs 'Microsoft Office' have Jaccard ~0.31 on char 3-grams — below
    any sensible verify_threshold. We document this as a known limitation rather
    than pretending the algorithm handles it.
    """
    skills = ["Microsoft Office", "Ms Office"]
    clusters = cluster_skills(skills, threshold=0.4, verify_threshold=0.7, seed=42)
    by_skill = {s: cid for cid, members in clusters.items() for s in members}
    assert by_skill["Microsoft Office"] != by_skill["Ms Office"], \
        "acronym variants should remain separate at sensible verify_threshold"


def test_no_transitive_runaway_via_shared_suffix():
    """Skills that share a common suffix ('Management') but are otherwise distinct
    must NOT all collapse into one giant cluster via transitive Union-Find."""
    skills = [
        "Event Management",
        "Cost Management",
        "Credit Management",
        "Risk Management",
        "Audit Management",
        "Debt Management",
        "Time Management",
        "Stress Management",
    ]
    clusters = cluster_skills(skills, threshold=0.4, verify_threshold=0.7, seed=42)
    # If runaway occurred, we'd have 1 giant cluster. Healthy result: most stay singleton.
    largest = max(len(v) for v in clusters.values())
    assert largest <= 2, f"transitive runaway: largest cluster has {largest} members"


def test_clustering_is_deterministic():
    skills = ["Python", "Java", "SQL", "Microsoft Office", "Microsoft Office Suite"]
    a = cluster_skills(skills, threshold=0.4, verify_threshold=0.7, num_perm=128, seed=42)
    b = cluster_skills(skills, threshold=0.4, verify_threshold=0.7, num_perm=128, seed=42)
    # Cluster IDs may differ between runs, but the partition must be equal
    a_part = {frozenset(v) for v in a.values()}
    b_part = {frozenset(v) for v in b.values()}
    assert a_part == b_part


def test_cluster_singletons_kept():
    """A skill matching nothing else must end up in its own cluster, not be dropped."""
    skills = ["Python", "Underwater Basket Weaving", "Java"]
    clusters = cluster_skills(skills, threshold=0.4, verify_threshold=0.7, num_perm=64, seed=1)
    all_members = [s for v in clusters.values() for s in v]
    assert set(all_members) == set(skills)


def test_empty_and_whitespace_skills_dropped():
    """Inputs that produce no shingles should not appear as clusters at all."""
    clusters = cluster_skills(["", "   ", "Python"], threshold=0.4, verify_threshold=0.7, seed=42)
    all_members = [s for v in clusters.values() for s in v]
    assert all_members == ["Python"]
