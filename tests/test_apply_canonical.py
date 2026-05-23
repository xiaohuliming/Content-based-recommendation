import json
import pandas as pd
from pathlib import Path
from src.normalize.apply_canonical import (
    pick_canonical_name,
    build_canonical_taxonomy,
    remap_document_skills,
)


def test_canonical_name_is_most_frequent_then_shortest():
    """Canonical name = the variant with highest doc_freq; tie → shortest, then alphabetical."""
    variants = ["Microsoft Office", "Ms Office", "Office 365", "Microsoft Office 365"]
    freqs = {"Microsoft Office": 50, "Ms Office": 50, "Office 365": 100, "Microsoft Office 365": 10}
    assert pick_canonical_name(variants, freqs) == "Office 365"
    # tie on count, prefer shorter
    freqs2 = {"AAA": 5, "AA": 5, "A": 5}
    assert pick_canonical_name(["AAA", "AA", "A"], freqs2) == "A"


def test_build_canonical_taxonomy_aggregates_freqs():
    clusters = {0: ["Python"], 1: ["Microsoft Office", "Ms Office"]}
    freqs = {"Python": 100, "Microsoft Office": 60, "Ms Office": 40}
    df = build_canonical_taxonomy(clusters, freqs, min_doc_freq=1)
    assert len(df) == 2
    office_row = df[df["canonical_name"].isin(["Microsoft Office", "Ms Office"])].iloc[0]
    assert office_row["doc_frequency"] == 100  # 60 + 40 merged
    assert set(office_row["variants"]) == {"Microsoft Office", "Ms Office"}


def test_frequency_prune_drops_low_freq_clusters():
    clusters = {0: ["Common"], 1: ["Rare"]}
    freqs = {"Common": 100, "Rare": 2}
    df = build_canonical_taxonomy(clusters, freqs, min_doc_freq=5)
    assert len(df) == 1
    assert df.iloc[0]["canonical_name"] == "Common"


def test_remap_document_skills_replaces_variants_with_canonical():
    docs = pd.DataFrame([
        {"doc_id": "A", "extracted_skills": "Microsoft Office,Python"},
        {"doc_id": "B", "extracted_skills": "Ms Office,Java"},
    ])
    skill_to_canonical = {
        "Microsoft Office": "Office 365",
        "Ms Office": "Office 365",
        "Python": "Python",
        "Java": "Java",
    }
    valid_canonicals = {"Office 365", "Python", "Java"}
    out = remap_document_skills(docs, "extracted_skills", skill_to_canonical, valid_canonicals)
    # Order preserved, duplicates from collapsing kept as a single occurrence per doc
    assert out.iloc[0]["canonical_skills"] == ["Office 365", "Python"]
    assert out.iloc[1]["canonical_skills"] == ["Office 365", "Java"]


def test_remap_drops_pruned_skills():
    """Skills not in valid_canonicals (because they got frequency-pruned) drop out."""
    docs = pd.DataFrame([{"doc_id": "A", "extracted_skills": "Python,Niche"}])
    out = remap_document_skills(
        docs, "extracted_skills",
        {"Python": "Python", "Niche": "Niche"},
        valid_canonicals={"Python"},
    )
    assert out.iloc[0]["canonical_skills"] == ["Python"]


def test_build_canonical_taxonomy_empty_clusters_returns_empty_df():
    df = build_canonical_taxonomy({}, {}, min_doc_freq=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
