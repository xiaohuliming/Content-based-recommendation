"""Apply LSH clustering + frequency prune to produce canonical skill universe."""
import json
from pathlib import Path

import pandas as pd

from src.normalize.minhash_lsh import cluster_skills


def pick_canonical_name(variants: list[str], freqs: dict[str, int]) -> str:
    """Canonical = most frequent; tie-break by shorter name, then alphabetical."""
    return min(variants, key=lambda v: (-freqs.get(v, 0), len(v), v))


def build_canonical_taxonomy(
    clusters: dict[int, list[str]],
    freqs: dict[str, int],
    *,
    min_doc_freq: int,
) -> pd.DataFrame:
    """Build canonical_skill_id, canonical_name, doc_frequency, variants[] table."""
    rows = []
    for cid, variants in clusters.items():
        total_freq = sum(freqs.get(v, 0) for v in variants)
        if total_freq < min_doc_freq:
            continue
        rows.append({
            "canonical_skill_id": cid,
            "canonical_name": pick_canonical_name(variants, freqs),
            "doc_frequency": total_freq,
            "variants": sorted(variants),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("doc_frequency", ascending=False).reset_index(drop=True)


def remap_document_skills(
    docs: pd.DataFrame,
    raw_skill_column: str,
    skill_to_canonical: dict[str, str],
    valid_canonicals: set[str],
) -> pd.DataFrame:
    """Replace per-document raw skill names with canonical names; drop pruned skills.

    Output column: 'canonical_skills' (list of canonical names, deduplicated, order-preserving).
    """
    def remap(cell: str) -> list[str]:
        if not isinstance(cell, str) or not cell.strip():
            return []
        seen: dict[str, None] = {}  # dict preserves insertion order
        for raw in cell.split(","):
            raw = raw.strip()
            canonical = skill_to_canonical.get(raw)
            if canonical and canonical in valid_canonicals:
                seen.setdefault(canonical, None)
        return list(seen.keys())

    out = docs.copy()
    out.loc[:, "canonical_skills"] = out[raw_skill_column].apply(remap)
    return out


def main() -> None:
    """End-to-end: load Phase 1 outputs, PRUNE FIRST, then cluster, then write outputs.

    Why prune-first: empirical testing showed LSH on all 21K skills produces
    transitive Union-Find runaway (largest cluster of 3,641 at threshold 0.6
    even with exact-Jaccard verification). Pruning to doc_freq >= 5 first
    drops the input to ~2,300 skills with much less common-suffix noise,
    after which LSH+verify produces clean clusters (max size 4-6).
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PHASE1 = PROJECT_ROOT / "output"
    PHASE2 = PROJECT_ROOT / "output" / "phase2"
    PHASE2.mkdir(parents=True, exist_ok=True)
    MIN_DOC_FREQ = 5

    tax = pd.read_csv(PHASE1 / "skill_taxonomy.csv")
    freqs = dict(zip(tax["skill"], tax["doc_frequency"]))

    print(f"[1/4] Pruning by doc_frequency >= {MIN_DOC_FREQ}...")
    high_freq_skills = tax[tax["doc_frequency"] >= MIN_DOC_FREQ]["skill"].tolist()
    print(f"  → {len(high_freq_skills):,} survive (from {len(tax):,} raw)")

    print(f"[2/4] Clustering {len(high_freq_skills):,} skills via LSH"
          f" (threshold=0.4, verify_threshold=0.7)...")
    clusters = cluster_skills(
        high_freq_skills,
        threshold=0.4, verify_threshold=0.7, num_perm=128, seed=42,
    )
    print(f"  → {len(clusters):,} clusters ({len(high_freq_skills) - len(clusters):,} merges)")

    (PHASE2 / "skill_clusters.json").write_text(
        json.dumps({str(k): v for k, v in clusters.items()}, ensure_ascii=False, indent=2)
    )

    print(f"[3/4] Building canonical taxonomy...")
    # min_doc_freq is now redundant (we already pruned) but kept as a safety net at 1
    canon = build_canonical_taxonomy(clusters, freqs, min_doc_freq=1)
    print(f"  → {len(canon):,} canonical skills")
    canon_for_csv = canon.copy()
    canon_for_csv.loc[:, "variants"] = canon["variants"].apply(lambda v: ";".join(v))
    canon_for_csv.to_csv(PHASE2 / "skills_canonical.csv", index=False)

    # Build variant→canonical lookup
    skill_to_canonical: dict[str, str] = {}
    for _, row in canon.iterrows():
        for variant in row["variants"]:
            skill_to_canonical[variant] = row["canonical_name"]
    valid = set(canon["canonical_name"])

    print(f"[4/4] Remapping {len(skill_to_canonical):,} variants → canonical in courses + jobs...")
    courses = pd.read_csv(PHASE1 / "courses_skills.csv")
    courses_canon = remap_document_skills(courses, "extracted_skills", skill_to_canonical, valid)
    courses_canon.loc[:, "canonical_skills"] = courses_canon["canonical_skills"].apply(lambda v: ";".join(v))
    courses_canon.drop(columns=["extracted_skills"]).to_csv(
        PHASE2 / "courses_canonical.csv", index=False
    )

    jobs = pd.read_csv(PHASE1 / "jobs_sample_skills.csv")
    jobs_canon = remap_document_skills(jobs, "extracted_skills", skill_to_canonical, valid)
    jobs_canon.loc[:, "canonical_skills"] = jobs_canon["canonical_skills"].apply(lambda v: ";".join(v))
    jobs_canon.drop(columns=["extracted_skills"]).to_csv(
        PHASE2 / "jobs_canonical.csv", index=False
    )

    print("\nDone. Outputs in output/phase2/:")
    for name in ["skill_clusters.json", "skills_canonical.csv",
                 "courses_canonical.csv", "jobs_canonical.csv"]:
        size = (PHASE2 / name).stat().st_size / 1024
        print(f"  {name:30s} {size:>7.1f} KB")


if __name__ == "__main__":
    main()
