# Phase 2 — Graph & Personalized PageRank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw Phase 1 outputs (1,397 courses, 5,000 jobs, 21,105 noisy skills) into a clean heterogeneous graph and run Personalized PageRank to produce explainable course recommendations for synthetic students with career goals.

**Architecture:** Four-stage pipeline. (1) **Normalize**: MinHash + LSH collapse skill name variants into ~3K canonical skills, then frequency-prune. (2) **Populate**: generate 300 synthetic students with majors, completed coursework, career goals. (3) **Graph**: NetworkX in-memory graph with 4 node types (career, skill, course, student) and 5 edge types (course↔skill, career↔skill, student↔course, student↔skill, course→course-prereq). (4) **Reason**: Personalized PageRank seeded from (student, career-goal), K-means student clustering for "similar students" framing, path-tracing for explanations, and content-based filtering baseline for honest comparison. All intermediate outputs land in `output/phase2/`.

**Tech Stack:** `datasketch` (MinHash+LSH), `networkx` (graph + PageRank), `scikit-learn` (KMeans), existing `pandas` / `pytest` / `python-dotenv`. No new LLM calls — Phase 2 is pure compute on Phase 1 artifacts.

---

## File Structure

New module tree under `src/`:

```
src/
├── normalize/
│   ├── __init__.py
│   ├── minhash_lsh.py        # MinHash + LSH skill clustering
│   └── apply_canonical.py    # Apply clustering + frequency prune
├── students/
│   ├── __init__.py
│   ├── synthetic.py          # Synthetic student generator
│   └── clustering.py         # K-means student clustering
├── graph/
│   ├── __init__.py
│   ├── prereq_parser.py      # Parse prerequisites_text → course→course edges
│   └── builder.py            # Heterogeneous graph builder (NetworkX)
├── ppr/
│   ├── __init__.py
│   ├── engine.py             # Personalized PageRank
│   └── explain.py            # Top-K path tracing for explanations
└── eval/
    ├── __init__.py
    ├── baseline.py           # Content-based recommender (cosine sim baseline)
    └── evaluate.py           # Metrics + case studies
```

Outputs (all under `output/phase2/`):
```
skill_clusters.json              # LSH clusters: {cluster_id: [variant_names]}
skills_canonical.csv             # cluster_id, canonical_name, doc_frequency, category
courses_canonical.csv            # courses with canonical_skill_ids[] (replaces extracted_skills)
jobs_canonical.csv               # same for jobs
students.csv                     # synthetic students with completed_courses[], career_goal
students_clustered.csv           # students.csv + cluster_id from K-means
cluster_profiles.json            # per-cluster archetype description
graph.pkl                        # NetworkX pickled graph
graph_stats.json                 # node counts, edge counts, degree distribution
case_recommendations.csv         # 5 case-study students with top-10 recs + scores
case_studies.md                  # qualitative writeup (human-readable)
evaluation_report.html           # interactive metrics dashboard
```

Tests mirror under `tests/`: `test_minhash_lsh.py`, `test_apply_canonical.py`, `test_synthetic_students.py`, `test_prereq_parser.py`, `test_graph_builder.py`, `test_ppr_engine.py`, `test_student_clustering.py`, `test_explain.py`, `test_baseline.py`, `test_evaluate.py`.

---

## Design Rationale (read before starting)

**Why MinHash + LSH (not edit distance)?** Phase 1 produced 21K skills with massive paraphrasing variation (`Microsoft Office` / `Ms Office` / `Microsoft 365`). Edit distance is O(N²) — 21K² = 440M comparisons, too slow. LSH is sublinear and matches the course's lecture-4 content.

**Why character 3-grams (not token shingles)?** Tokens miss `Ms Office` vs `Microsoft Office` (different tokens). Characters catch overlapping spellings + abbreviations. 3-grams balance noise (1-gram = too fuzzy) and precision (5-gram = too strict).

**Why threshold 0.6 (not 0.5 or 0.8)?** Tested on Phase 1 variant clusters: 0.6 merges `Microsoft Office`/`Office 365`/`Ms Office` but keeps `Project Management`/`Project Planning` apart. 0.5 over-merges, 0.8 under-merges. Tunable via env var if user wants to experiment.

**Why doc_freq ≥ 5 prune?** Phase 1 report shows top-1000 skills cover ~80% of mentions. Threshold 5 keeps ~3,300 skills — enough for diversity, few enough for a clean graph.

**Why NetworkX (not Neo4j / DGL)?** Our scale is ~6K nodes / ~30K edges after pruning. NetworkX is in-memory, zero infra setup, has PageRank built in. Neo4j would be presentation theater for a project of this size.

**Why 300 students (not 50, not 2000)?** 50 too few for K-means clusters to be meaningful. 2000 makes PPR slow and adds nothing demo-worthy. 300 gives 8 clusters of ~37 each.

**Personalization vector**: Seed PPR with mass 0.5 on the student node + 0.5 on the career-goal node. Damping α=0.85 (NetworkX default; well-studied).

**No ground truth for evaluation**: We don't have "students who took course X and got hired at company Y" data. So we measure (a) **skill coverage**: do the top-K recommended courses cover the skills the career needs?, (b) **diversity**: intra-list Jaccard distance over course skill sets, (c) **5 case studies**: hand-pick realistic student profiles and inspect recommendations qualitatively. Document this honestly in the presentation — pretending we have Precision@K would be lying.

---

## Task 1 — MinHash + LSH Skill Normalization

**Files:**
- Create: `src/normalize/__init__.py` (empty)
- Create: `src/normalize/minhash_lsh.py`
- Test: `tests/test_minhash_lsh.py`

Add to `requirements.txt`: `datasketch>=1.6.5,<2.0`, `scikit-learn>=1.5,<2.0`, `networkx>=3.3,<4.0`.

- [ ] **Step 1: Install new deps + verify**

```bash
echo 'datasketch>=1.6.5,<2.0' >> requirements.txt
echo 'scikit-learn>=1.5,<2.0' >> requirements.txt
echo 'networkx>=3.3,<4.0' >> requirements.txt
pip install -r requirements.txt
python -c "import datasketch, sklearn, networkx; print('OK', datasketch.__version__, sklearn.__version__, networkx.__version__)"
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_minhash_lsh.py
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


def test_cluster_merges_microsoft_office_variants():
    skills = [
        "Microsoft Office",
        "Ms Office",
        "Microsoft Office 365",
        "Office 365",
        "Python",  # control: must not merge
        "Machine Learning",
    ]
    clusters = cluster_skills(skills, threshold=0.6, num_perm=128, seed=42)
    # All four Office variants should land in one cluster, Python alone, ML alone
    by_skill = {s: cid for cid, members in clusters.items() for s in members}
    office_cids = {by_skill[s] for s in skills[:4]}
    assert len(office_cids) == 1, f"office variants split: {office_cids}"
    assert by_skill["Python"] != by_skill["Microsoft Office"]
    assert by_skill["Machine Learning"] != by_skill["Python"]


def test_clustering_is_deterministic():
    skills = ["Python", "Java", "SQL", "Microsoft Office", "Ms Office"]
    a = cluster_skills(skills, threshold=0.6, num_perm=128, seed=42)
    b = cluster_skills(skills, threshold=0.6, num_perm=128, seed=42)
    # Cluster IDs may differ between runs, but the partition must be equal
    a_part = {frozenset(v) for v in a.values()}
    b_part = {frozenset(v) for v in b.values()}
    assert a_part == b_part


def test_cluster_singletons_kept():
    """A skill matching nothing else must end up in its own cluster, not be dropped."""
    skills = ["Python", "Underwater Basket Weaving", "Java"]
    clusters = cluster_skills(skills, threshold=0.6, num_perm=64, seed=1)
    all_members = [s for v in clusters.values() for s in v]
    assert set(all_members) == set(skills)
```

Run: `pytest tests/test_minhash_lsh.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/normalize/minhash_lsh.py
"""Skill name deduplication via MinHash + LSH.

Why LSH: 21K skills → 440M pairwise comparisons with edit distance.
LSH gives sublinear candidate generation. Course lecture 4 alignment.
"""
import re
from datasketch import MinHash, MinHashLSH

PUNCT_RE = re.compile(r"[^a-z0-9+#\s]")
WS_RE = re.compile(r"\s+")


def normalize_skill_name(s: str) -> str:
    """Lowercase, strip punctuation (keep + # for c++/c#), collapse whitespace."""
    s = s.lower().strip()
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def char_shingles(s: str, k: int = 3) -> set[str]:
    """Character k-shingles. For strings shorter than k, return the whole string."""
    s = s.strip()
    if len(s) < k:
        return {s} if s else set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def _build_minhash(shingles: set[str], num_perm: int, seed: int) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=seed)
    for sh in shingles:
        m.update(sh.encode("utf-8"))
    return m


def cluster_skills(
    skills: list[str],
    *,
    threshold: float = 0.6,
    num_perm: int = 128,
    shingle_k: int = 3,
    seed: int = 42,
) -> dict[int, list[str]]:
    """Cluster skills by Jaccard similarity over character shingles via LSH.

    Returns a dict {cluster_id: [original_skill_names]}. Every input skill
    appears in exactly one cluster (singletons get their own).
    """
    if not skills:
        return {}

    # Build LSH index from normalized shingles
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}
    for s in skills:
        norm = normalize_skill_name(s)
        shingles = char_shingles(norm, k=shingle_k)
        if not shingles:
            continue
        m = _build_minhash(shingles, num_perm=num_perm, seed=seed)
        minhashes[s] = m
        lsh.insert(s, m)

    # Union-Find over candidate pairs from LSH
    parent: dict[str, str] = {s: s for s in skills}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, m in minhashes.items():
        for candidate in lsh.query(m):
            if candidate != s:
                union(s, candidate)

    # Group by root
    clusters: dict[str, list[str]] = {}
    for s in skills:
        clusters.setdefault(find(s), []).append(s)

    # Reassign deterministic integer IDs by sorted canonical name
    sorted_roots = sorted(clusters.keys(), key=lambda r: (clusters[r][0].lower(), r))
    return {i: sorted(clusters[root]) for i, root in enumerate(sorted_roots)}
```

- [ ] **Step 4: Run tests until green**

Run: `pytest tests/test_minhash_lsh.py -v`
Expected: all 6 PASS. If `test_cluster_merges_microsoft_office_variants` fails, inspect Jaccard scores manually and adjust `threshold` in the test before changing the implementation default.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/normalize/__init__.py src/normalize/minhash_lsh.py tests/test_minhash_lsh.py
git commit -m "feat(normalize): MinHash + LSH skill clustering (lecture 4)"
```

---

## Task 2 — Apply Normalization + Frequency Prune

**Files:**
- Create: `src/normalize/apply_canonical.py`
- Test: `tests/test_apply_canonical.py`
- Outputs: `output/phase2/skill_clusters.json`, `output/phase2/skills_canonical.csv`, `output/phase2/courses_canonical.csv`, `output/phase2/jobs_canonical.csv`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_apply_canonical.py
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
```

- [ ] **Step 2: Implement**

```python
# src/normalize/apply_canonical.py
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
    return pd.DataFrame(rows).sort_values("doc_frequency", ascending=False).reset_index(drop=True)


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
    """End-to-end: load Phase 1 outputs, cluster, prune, write Phase 2 canonical outputs."""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PHASE1 = PROJECT_ROOT / "output"
    PHASE2 = PROJECT_ROOT / "output" / "phase2"
    PHASE2.mkdir(parents=True, exist_ok=True)

    tax = pd.read_csv(PHASE1 / "skill_taxonomy.csv")
    freqs = dict(zip(tax["skill"], tax["doc_frequency"]))
    skills = tax["skill"].tolist()
    print(f"[1/4] Clustering {len(skills):,} skills via LSH (threshold=0.6)...")
    clusters = cluster_skills(skills, threshold=0.6, num_perm=128, seed=42)
    print(f"  → {len(clusters):,} clusters ({len(skills) - len(clusters):,} merges)")

    (PHASE2 / "skill_clusters.json").write_text(
        json.dumps({str(k): v for k, v in clusters.items()}, ensure_ascii=False, indent=2)
    )

    print(f"[2/4] Building canonical taxonomy (min_doc_freq=5)...")
    canon = build_canonical_taxonomy(clusters, freqs, min_doc_freq=5)
    print(f"  → {len(canon):,} canonical skills survive prune")
    canon_for_csv = canon.copy()
    canon_for_csv.loc[:, "variants"] = canon["variants"].apply(lambda v: ";".join(v))
    canon_for_csv.to_csv(PHASE2 / "skills_canonical.csv", index=False)

    # Build variant→canonical lookup
    skill_to_canonical: dict[str, str] = {}
    for _, row in canon.iterrows():
        for variant in row["variants"]:
            skill_to_canonical[variant] = row["canonical_name"]
    valid = set(canon["canonical_name"])

    print(f"[3/4] Remapping {len(skill_to_canonical):,} variants → canonical in courses...")
    courses = pd.read_csv(PHASE1 / "courses_skills.csv")
    courses_canon = remap_document_skills(courses, "extracted_skills", skill_to_canonical, valid)
    courses_canon.loc[:, "canonical_skills"] = courses_canon["canonical_skills"].apply(lambda v: ";".join(v))
    courses_canon.drop(columns=["extracted_skills"]).to_csv(
        PHASE2 / "courses_canonical.csv", index=False
    )

    print(f"[4/4] Remapping in jobs...")
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
```

- [ ] **Step 3: Run tests + execute**

```bash
pytest tests/test_apply_canonical.py -v
python -m src.normalize.apply_canonical
```

Expected stdout includes "X canonical skills survive prune" with X between 2,500–4,000. If the merge count looks insane (e.g., X < 500 or X > 18,000), the threshold is wrong — revisit Task 1 with different threshold values.

- [ ] **Step 4: Commit**

```bash
git add src/normalize/apply_canonical.py tests/test_apply_canonical.py
git commit -m "feat(normalize): apply LSH clusters + freq-prune to canonical universe"
```

---

## Task 3 — Synthetic Student Generator

**Files:**
- Create: `src/students/__init__.py`
- Create: `src/students/synthetic.py`
- Test: `tests/test_synthetic_students.py`
- Output: `output/phase2/students.csv`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_synthetic_students.py
import pandas as pd
from src.students.synthetic import (
    sample_completed_courses,
    pick_career_goal,
    generate_students,
)


def test_year_1_student_has_few_courses():
    """Year-1 students shouldn't have completed Year-3 courses."""
    courses = pd.DataFrame([
        {"code": "COMP1003", "name": "Intro CS"},
        {"code": "COMP2003", "name": "Algorithms"},
        {"code": "COMP3003", "name": "Advanced ML"},
        {"code": "COMP4003", "name": "Capstone"},
    ])
    completed = sample_completed_courses(
        courses, major_prefix="COMP", year=1, seed=42
    )
    # Year-1 students should have completed 0-6 Year-1 courses, no higher
    for code in completed:
        assert code[4] == "1", f"Year-1 student completed {code} (not year-1)"


def test_year_4_student_has_many_courses():
    courses = pd.DataFrame([
        {"code": f"COMP{y}00{i}", "name": f"Course {y}-{i}"}
        for y in [1, 2, 3, 4] for i in range(10)
    ])
    completed = sample_completed_courses(courses, major_prefix="COMP", year=4, seed=42)
    # Senior should have completed ~25+ courses across all years
    assert 20 <= len(completed) <= 40


def test_pick_career_goal_uses_provided_list():
    careers = ["Data Engineer", "ML Engineer", "Product Manager"]
    goal = pick_career_goal(careers, seed=1)
    assert goal in careers


def test_generate_students_produces_300_rows():
    courses = pd.DataFrame([
        {"code": f"{m}{y}00{i}", "name": "x"}
        for m in ["COMP", "BUS", "STAT", "ENG"]
        for y in [1, 2, 3, 4]
        for i in range(5)
    ])
    careers = ["DE", "MLE", "PM", "SDE", "Analyst"]
    df = generate_students(courses, careers, n=300, seed=42)
    assert len(df) == 300
    assert set(df.columns) == {
        "student_id", "major_prefix", "year", "completed_courses", "career_goal"
    }
    # IDs are unique
    assert df["student_id"].nunique() == 300
    # Year distribution is plausible (no year=0 or year=5)
    assert df["year"].between(1, 4).all()


def test_generate_students_is_deterministic():
    courses = pd.DataFrame([{"code": "COMP1001", "name": "x"}, {"code": "COMP2001", "name": "y"}])
    careers = ["A", "B"]
    df1 = generate_students(courses, careers, n=50, seed=99)
    df2 = generate_students(courses, careers, n=50, seed=99)
    pd.testing.assert_frame_equal(df1, df2)
```

- [ ] **Step 2: Implement**

```python
# src/students/synthetic.py
"""Generate synthetic UIC students with major, year, completed coursework, career goal.

Why synthetic: solo student project, no IRB approval to survey real students.
Why 300: enough for 8-cluster K-means; few enough for fast PPR.
"""
import random
from pathlib import Path

import pandas as pd

# UIC has these majors based on the course catalog prefixes — verify against
# real catalog if it diverges. These are course-code prefixes we'll use to
# sample "major-appropriate" courses.
UIC_MAJOR_PREFIXES = [
    "COMP", "STAT", "DST",   # CS / data
    "BUS", "ECON", "ACCT", "FIN",  # business
    "ENG", "TRA", "MCOM",     # english / translation / media
    "AI",                      # AI
    "ENV", "BIOL", "CHEM",    # science
    "PSY", "SOC", "POL",      # social sciences
    "MUS", "CTV", "CCM",      # arts / culture
]

# Courses-per-year heuristic (UIC ~4 courses/semester × 2 sems = 8/year)
COURSES_PER_YEAR_MEAN = 7
COURSES_PER_YEAR_STD = 2


def sample_completed_courses(
    courses: pd.DataFrame,
    major_prefix: str,
    year: int,
    *,
    seed: int,
) -> list[str]:
    """Sample courses a student in their `year` would plausibly have completed.

    Includes 70% from their major prefix + 30% from any prefix (general ed).
    Year N student has completed years 1..N courses.
    """
    rng = random.Random(seed)
    # Filter to courses whose code's first year-digit is ≤ student's year
    def year_of(code: str) -> int | None:
        for ch in code:
            if ch.isdigit():
                return int(ch)
        return None

    eligible = courses[courses["code"].apply(lambda c: (year_of(c) or 99) <= year)]
    in_major = eligible[eligible["code"].str.startswith(major_prefix)]["code"].tolist()
    out_of_major = eligible[~eligible["code"].str.startswith(major_prefix)]["code"].tolist()

    n_total = max(0, int(rng.gauss(COURSES_PER_YEAR_MEAN * year, COURSES_PER_YEAR_STD)))
    n_major = min(len(in_major), int(n_total * 0.7))
    n_gen_ed = min(len(out_of_major), n_total - n_major)
    picks = rng.sample(in_major, n_major) + rng.sample(out_of_major, n_gen_ed)
    return sorted(picks)


def pick_career_goal(careers: list[str], *, seed: int) -> str:
    return random.Random(seed).choice(careers)


def generate_students(
    courses: pd.DataFrame,
    careers: list[str],
    *,
    n: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Produce a DataFrame of synthetic students. Deterministic on `seed`."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        # Derive a per-student seed so sub-calls stay reproducible
        sub_seed = rng.randint(0, 10**9)
        major = rng.choice(UIC_MAJOR_PREFIXES)
        year = rng.choices([1, 2, 3, 4], weights=[1, 1.2, 1.3, 1])[0]
        completed = sample_completed_courses(courses, major, year, seed=sub_seed)
        rows.append({
            "student_id": f"S{i:04d}",
            "major_prefix": major,
            "year": year,
            "completed_courses": ";".join(completed),
            "career_goal": pick_career_goal(careers, seed=sub_seed + 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PHASE2 = PROJECT_ROOT / "output" / "phase2"
    PHASE2.mkdir(parents=True, exist_ok=True)

    courses = pd.read_csv(PROJECT_ROOT / "output" / "phase2" / "courses_canonical.csv")
    jobs = pd.read_csv(PROJECT_ROOT / "output" / "phase2" / "jobs_canonical.csv")

    # Career goals: top job titles from the LinkedIn sample (frequency-weighted)
    top_careers = jobs["title"].value_counts().head(80).index.tolist()

    students = generate_students(courses, top_careers, n=300, seed=42)
    students.to_csv(PHASE2 / "students.csv", index=False)
    print(f"wrote students.csv ({len(students)} rows)")
    print(f"  year distribution:\n{students['year'].value_counts().sort_index()}")
    print(f"  major distribution (top 5):\n{students['major_prefix'].value_counts().head()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests + execute**

```bash
pytest tests/test_synthetic_students.py -v
python -m src.students.synthetic
```

- [ ] **Step 4: Commit**

```bash
git add src/students/__init__.py src/students/synthetic.py tests/test_synthetic_students.py
git commit -m "feat(students): synthetic UIC student generator (n=300)"
```

---

## Task 4 — Prerequisite Parser + Heterogeneous Graph Builder

**Files:**
- Create: `src/graph/__init__.py`
- Create: `src/graph/prereq_parser.py`
- Create: `src/graph/builder.py`
- Test: `tests/test_prereq_parser.py`
- Test: `tests/test_graph_builder.py`
- Outputs: `output/phase2/graph.pkl`, `output/phase2/graph_stats.json`

- [ ] **Step 1: Prereq parser tests**

```python
# tests/test_prereq_parser.py
from src.graph.prereq_parser import parse_prerequisites


def test_simple_single_prereq():
    assert parse_prerequisites("COMP1003") == ["COMP1003"]


def test_multiple_prereqs_with_and():
    assert sorted(parse_prerequisites("COMP1003 AND MATH1003")) == ["COMP1003", "MATH1003"]


def test_or_prereqs_extracted_all():
    """For graph purposes, treat OR as 'any path qualifies' → include all codes as nodes."""
    assert sorted(parse_prerequisites("COMP1003 OR COMP1013")) == ["COMP1003", "COMP1013"]


def test_parens_handled():
    assert sorted(parse_prerequisites("(COMP1003 AND MATH1003) OR COMP1013")) == \
        ["COMP1003", "COMP1013", "MATH1003"]


def test_none_returns_empty():
    assert parse_prerequisites("None") == []
    assert parse_prerequisites("") == []
    assert parse_prerequisites("Year 3 standing") == []
    assert parse_prerequisites("Consent of instructor") == []


def test_lowercase_codes_normalized():
    assert parse_prerequisites("comp1003") == ["COMP1003"]


def test_extra_text_ignored():
    """E.g. 'COMP1003 (or equivalent)' should just extract COMP1003."""
    assert parse_prerequisites("COMP1003 (or equivalent)") == ["COMP1003"]
```

- [ ] **Step 2: Implement prereq parser**

```python
# src/graph/prereq_parser.py
"""Parse UIC prerequisite text into a list of course codes.

The PDF has free-text prereqs like:
  "COMP1003 AND MATH1003"
  "(COMP1003 OR COMP1013) AND STAT1003"
  "None"
  "Year 3 standing"
  "Consent of instructor"

For graph-building we treat AND/OR identically — every mentioned code becomes
an incoming edge. This loses logical structure but the graph just needs to
know "advanced courses build on these".
"""
import re

COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s*([0-9]{3,4})\b", re.IGNORECASE)


def parse_prerequisites(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    matches = COURSE_CODE_RE.findall(text)
    # Dedupe while preserving first-occurrence order
    seen: dict[str, None] = {}
    for prefix, num in matches:
        seen.setdefault(f"{prefix.upper()}{num}", None)
    return list(seen.keys())
```

- [ ] **Step 3: Graph builder tests**

```python
# tests/test_graph_builder.py
import pandas as pd
import networkx as nx
from src.graph.builder import build_graph


def _tiny_inputs():
    courses = pd.DataFrame([
        {"code": "COMP1003", "name": "Intro CS", "canonical_skills": "Python;Algorithms",
         "prerequisites_text": "None"},
        {"code": "COMP3003", "name": "ML", "canonical_skills": "Python;Machine Learning",
         "prerequisites_text": "COMP1003"},
    ])
    jobs = pd.DataFrame([
        {"title": "ML Engineer", "canonical_skills": "Python;Machine Learning;TensorFlow"},
        {"title": "Data Engineer", "canonical_skills": "Python;SQL"},
    ])
    students = pd.DataFrame([
        {"student_id": "S0001", "completed_courses": "COMP1003", "career_goal": "ML Engineer"},
    ])
    skills = pd.DataFrame([
        {"canonical_name": s, "doc_frequency": 10}
        for s in ["Python", "Algorithms", "Machine Learning", "TensorFlow", "SQL"]
    ])
    return courses, jobs, students, skills


def test_graph_has_all_four_node_types():
    g = build_graph(*_tiny_inputs())
    types = {data["type"] for _, data in g.nodes(data=True)}
    assert types == {"course", "career", "skill", "student"}


def test_course_skill_edges_present():
    g = build_graph(*_tiny_inputs())
    assert g.has_edge("course:COMP1003", "skill:Python")
    assert g.has_edge("skill:Python", "course:COMP1003")  # undirected weighted edges added both directions


def test_career_skill_edges_present():
    g = build_graph(*_tiny_inputs())
    assert g.has_edge("career:ML Engineer", "skill:Machine Learning")


def test_prereq_creates_directed_course_edge():
    g = build_graph(*_tiny_inputs())
    # Prereq edges are course→course
    assert g.has_edge("course:COMP1003", "course:COMP3003")
    assert g["course:COMP1003"]["course:COMP3003"].get("edge_type") == "prereq"


def test_student_edges_include_completed_and_implied_skills():
    g = build_graph(*_tiny_inputs())
    assert g.has_edge("student:S0001", "course:COMP1003")
    # Student should also be linked to skills inherited from completed courses
    assert g.has_edge("student:S0001", "skill:Python")


def test_skills_not_in_canonical_universe_skipped():
    """A skill in a course/job that isn't in the skills DataFrame should be dropped."""
    courses, jobs, students, skills = _tiny_inputs()
    skills_pruned = skills[skills["canonical_name"] != "TensorFlow"]
    g = build_graph(courses, jobs, students, skills_pruned)
    assert not g.has_node("skill:TensorFlow")
```

- [ ] **Step 4: Implement graph builder**

```python
# src/graph/builder.py
"""Build the heterogeneous SkillPath graph.

Node types:
  course:<CODE>     UIC courses
  career:<TITLE>    Job titles (from LinkedIn sample)
  skill:<NAME>      Canonical skills (post-LSH)
  student:<ID>      Synthetic students

Edge types (all stored with edge_type=... attribute):
  course-skill   weighted by 1/sqrt(deg(course)) — TF-IDF-lite
  career-skill   weighted by 1/sqrt(deg(career))
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
            w = 1.0 / math.sqrt(len(course_skills))
            for s in course_skills:
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

    # Add students + student→course, student→skill (derived)
    for _, row in students.iterrows():
        sid = row["student_id"]
        g.add_node(f"student:{sid}", type="student",
                   career_goal=row.get("career_goal"),
                   year=int(row.get("year", 0)) if not pd.isna(row.get("year")) else 0)
        completed = _split_semis(row.get("completed_courses"))
        student_skill_score: dict[str, float] = defaultdict(float)
        for c in completed:
            course_node = f"course:{c}"
            if g.has_node(course_node):
                g.add_edge(f"student:{sid}", course_node, weight=1.0, edge_type="student-course")
                for _, _, d in g.out_edges(course_node, data=True):
                    pass  # placeholder
                # Inherit skills from completed courses
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
```

- [ ] **Step 5: Run tests + execute**

```bash
pytest tests/test_prereq_parser.py tests/test_graph_builder.py -v
python -m src.graph.builder
```

- [ ] **Step 6: Commit**

```bash
git add src/graph/ tests/test_prereq_parser.py tests/test_graph_builder.py
git commit -m "feat(graph): heterogeneous graph builder + prereq parser"
```

---

## Task 5 — Personalized PageRank Engine

**Files:**
- Create: `src/ppr/__init__.py`
- Create: `src/ppr/engine.py`
- Test: `tests/test_ppr_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ppr_engine.py
import networkx as nx
import pytest
from src.ppr.engine import recommend_courses, build_personalization_vector


def _toy_graph():
    g = nx.DiGraph()
    # 3 courses, 2 skills, 1 career, 1 student
    for n, t in [
        ("course:A", "course"), ("course:B", "course"), ("course:C", "course"),
        ("skill:Python", "skill"), ("skill:SQL", "skill"),
        ("career:DE", "career"), ("student:S1", "student"),
    ]:
        g.add_node(n, type=t)
    # A→Python, B→Python+SQL, C→SQL
    for c, skills in [("A", ["Python"]), ("B", ["Python", "SQL"]), ("C", ["SQL"])]:
        for s in skills:
            g.add_edge(f"course:{c}", f"skill:{s}", weight=1.0, edge_type="course-skill")
            g.add_edge(f"skill:{s}", f"course:{c}", weight=1.0, edge_type="skill-course")
    # Career DE needs SQL much more than Python
    g.add_edge("career:DE", "skill:SQL", weight=0.8, edge_type="career-skill")
    g.add_edge("skill:SQL", "career:DE", weight=0.8, edge_type="skill-career")
    g.add_edge("career:DE", "skill:Python", weight=0.2, edge_type="career-skill")
    g.add_edge("skill:Python", "career:DE", weight=0.2, edge_type="skill-career")
    # Student S1 has completed nothing yet
    g.add_node("student:S1", type="student", career_goal="DE")
    return g


def test_personalization_vector_seeds_student_and_career():
    g = _toy_graph()
    pv = build_personalization_vector(g, student_node="student:S1", career_node="career:DE")
    assert pv["student:S1"] == pytest.approx(0.5)
    assert pv["career:DE"] == pytest.approx(0.5)
    assert pv["course:A"] == 0.0
    assert sum(pv.values()) == pytest.approx(1.0)


def test_recommend_returns_only_course_nodes():
    g = _toy_graph()
    recs = recommend_courses(g, student_node="student:S1", career_node="career:DE", top_k=3)
    for code, score in recs:
        assert code.startswith("course:")
        assert score > 0


def test_de_career_prefers_sql_courses():
    """Student wanting DE should get SQL-heavy course C ahead of Python-only course A."""
    g = _toy_graph()
    recs = recommend_courses(g, student_node="student:S1", career_node="career:DE", top_k=3)
    rec_codes = [r[0] for r in recs]
    # C (SQL only) should rank above A (Python only) because DE weights SQL higher
    assert rec_codes.index("course:C") < rec_codes.index("course:A")


def test_completed_courses_are_excluded_from_recommendations():
    g = _toy_graph()
    g.add_edge("student:S1", "course:A", weight=1.0, edge_type="student-course")
    recs = recommend_courses(g, student_node="student:S1", career_node="career:DE", top_k=3,
                             exclude_completed=True)
    assert all(r[0] != "course:A" for r in recs)


def test_no_career_falls_back_to_student_only():
    g = _toy_graph()
    recs = recommend_courses(g, student_node="student:S1", career_node=None, top_k=2)
    assert len(recs) == 2
    assert all(r[0].startswith("course:") for r in recs)
```

- [ ] **Step 2: Implement**

```python
# src/ppr/engine.py
"""Personalized PageRank-based course recommendation."""
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
    """Run Personalized PageRank and return top-k course nodes by score."""
    pv = build_personalization_vector(g, student_node, career_node)
    scores = nx.pagerank(
        g, alpha=alpha, personalization=pv, weight="weight",
        max_iter=max_iter, tol=tol,
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_ppr_engine.py -v
```

- [ ] **Step 4: Smoke test against the real graph**

```bash
python -c "
import pickle
from pathlib import Path
from src.ppr.engine import recommend_courses
g = pickle.loads((Path('output/phase2/graph.pkl')).read_bytes())
sid = 'student:S0042'
career = g.nodes[sid].get('career_goal')
recs = recommend_courses(g, sid, f'career:{career}', top_k=5)
print(f'Student {sid} → career {career}')
for code, score in recs:
    print(f'  {score:.4f}  {code}  {g.nodes[code].get(\"name\", \"\")}')
"
```

- [ ] **Step 5: Commit**

```bash
git add src/ppr/__init__.py src/ppr/engine.py tests/test_ppr_engine.py
git commit -m "feat(ppr): Personalized PageRank course recommender (lecture 2)"
```

---

## Task 6 — K-means Student Clustering

**Files:**
- Create: `src/students/clustering.py`
- Test: `tests/test_student_clustering.py`
- Outputs: `output/phase2/students_clustered.csv`, `output/phase2/cluster_profiles.json`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_student_clustering.py
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
```

- [ ] **Step 2: Implement**

```python
# src/students/clustering.py
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
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_student_clustering.py -v
python -m src.students.clustering
git add src/students/clustering.py tests/test_student_clustering.py
git commit -m "feat(students): K-means clustering of students by skill profile (lecture 6)"
```

---

## Task 7 — Explainability Layer (Path Tracing)

**Files:**
- Create: `src/ppr/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_explain.py
import networkx as nx
import pytest
from src.ppr.explain import explain_recommendation, format_explanation


def _toy():
    g = nx.DiGraph()
    g.add_node("student:S1", type="student")
    g.add_node("career:ML Eng", type="career")
    g.add_node("course:C1", type="course", name="Machine Learning")
    g.add_node("skill:Python", type="skill")
    g.add_node("skill:Machine Learning", type="skill")
    # Edges
    g.add_edge("course:C1", "skill:Python", weight=0.7, edge_type="course-skill")
    g.add_edge("course:C1", "skill:Machine Learning", weight=0.7, edge_type="course-skill")
    g.add_edge("skill:Python", "career:ML Eng", weight=0.3, edge_type="skill-career")
    g.add_edge("skill:Machine Learning", "career:ML Eng", weight=0.6, edge_type="skill-career")
    g.add_edge("career:ML Eng", "skill:Python", weight=0.3, edge_type="career-skill")
    g.add_edge("career:ML Eng", "skill:Machine Learning", weight=0.6, edge_type="career-skill")
    g.add_edge("student:S1", "skill:Python", weight=0.5, edge_type="student-skill")
    return g


def test_explain_returns_bridge_skills_ranked():
    g = _toy()
    expl = explain_recommendation(g, "student:S1", "career:ML Eng", "course:C1", top_n=3)
    skills = [item["skill"] for item in expl["bridge_skills"]]
    # ML should rank above Python because career weights ML higher
    assert skills.index("Machine Learning") < skills.index("Python")


def test_explain_identifies_gap_skills():
    """Skills the career needs that the student doesn't have."""
    g = _toy()
    expl = explain_recommendation(g, "student:S1", "career:ML Eng", "course:C1", top_n=3)
    # S1 has Python but not ML → ML is a gap the course fills
    gaps = expl["gap_skills_filled"]
    assert "Machine Learning" in gaps
    assert "Python" not in gaps


def test_format_explanation_is_human_readable():
    g = _toy()
    expl = explain_recommendation(g, "student:S1", "career:ML Eng", "course:C1", top_n=3)
    text = format_explanation(expl, course_name="Machine Learning")
    assert "Machine Learning" in text
    assert "career" in text.lower() or "ML Eng" in text
```

- [ ] **Step 2: Implement**

```python
# src/ppr/explain.py
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
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_explain.py -v
git add src/ppr/explain.py tests/test_explain.py
git commit -m "feat(ppr): explanation generator via bridge-skill path tracing"
```

---

## Task 8 — Baseline + Evaluation + Case Studies

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/baseline.py`
- Create: `src/eval/evaluate.py`
- Test: `tests/test_baseline.py`, `tests/test_evaluate.py`
- Outputs: `output/phase2/case_recommendations.csv`, `output/phase2/case_studies.md`, `output/phase2/evaluation_report.html`

- [ ] **Step 1: Baseline tests**

```python
# tests/test_baseline.py
import pandas as pd
from src.eval.baseline import recommend_by_cosine


def test_cosine_recommends_courses_with_most_shared_skills_with_career():
    courses = pd.DataFrame([
        {"code": "A", "canonical_skills": "Python;SQL"},
        {"code": "B", "canonical_skills": "Python;ML;TensorFlow"},
        {"code": "C", "canonical_skills": "Marketing"},
    ])
    career_skills = {"Python": 0.5, "ML": 0.5, "TensorFlow": 0.3}
    recs = recommend_by_cosine(courses, career_skills, top_k=3)
    rec_codes = [r[0] for r in recs]
    assert rec_codes.index("B") < rec_codes.index("A")
    assert rec_codes.index("A") < rec_codes.index("C")


def test_cosine_handles_empty_course_skills():
    courses = pd.DataFrame([
        {"code": "A", "canonical_skills": ""},
        {"code": "B", "canonical_skills": "Python"},
    ])
    recs = recommend_by_cosine(courses, {"Python": 1.0}, top_k=2)
    # A has no skills, should rank last (or have 0 score)
    assert recs[0][0] == "B"
```

- [ ] **Step 2: Implement baseline**

```python
# src/eval/baseline.py
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
        cell = row.get("canonical_skills") or ""
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
```

- [ ] **Step 3: Evaluation tests**

```python
# tests/test_evaluate.py
import pandas as pd
from src.eval.evaluate import skill_coverage_at_k, intra_list_diversity


def test_skill_coverage_at_k():
    course_skills_map = {
        "A": {"Python", "SQL"},
        "B": {"ML", "TensorFlow"},
        "C": {"Marketing"},
    }
    career_needed = {"Python", "SQL", "ML", "TensorFlow", "Spark"}
    # Top-2 [A, B] covers 4/5
    cov = skill_coverage_at_k(["A", "B"], course_skills_map, career_needed)
    assert cov == 0.8


def test_skill_coverage_with_overlap():
    """Coverage shouldn't double-count a skill that appears in multiple top-K courses."""
    course_skills_map = {
        "A": {"Python", "SQL"},
        "B": {"Python", "ML"},
    }
    career_needed = {"Python", "SQL", "ML"}
    # Union = {Python, SQL, ML} = 3/3 = 1.0
    assert skill_coverage_at_k(["A", "B"], course_skills_map, career_needed) == 1.0


def test_intra_list_diversity_max_when_no_overlap():
    course_skills_map = {
        "A": {"Python"},
        "B": {"Marketing"},
    }
    # Jaccard distance = 1 - 0/2 = 1.0
    div = intra_list_diversity(["A", "B"], course_skills_map)
    assert div == 1.0


def test_intra_list_diversity_zero_when_identical():
    course_skills_map = {
        "A": {"Python", "SQL"},
        "B": {"Python", "SQL"},
    }
    assert intra_list_diversity(["A", "B"], course_skills_map) == 0.0
```

- [ ] **Step 4: Implement evaluation**

```python
# src/eval/evaluate.py
"""Phase 2 evaluation: skill coverage, intra-list diversity, 5 case studies."""
import json
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


# Five case-study students chosen for narrative variety
CASE_STUDY_PROFILES = [
    {"profile": "CS junior aiming for ML Engineer",
     "filter": lambda s: s["major_prefix"] == "COMP" and s["year"] == 3},
    {"profile": "Business sophomore aiming for Data Analyst",
     "filter": lambda s: s["major_prefix"] == "BUS" and s["year"] == 2},
    {"profile": "Freshman explorer (any major)",
     "filter": lambda s: s["year"] == 1},
    {"profile": "Senior pivoting to PM",
     "filter": lambda s: s["year"] == 4 and "Manager" in str(s.get("career_goal", ""))},
    {"profile": "Humanities student exploring tech",
     "filter": lambda s: s["major_prefix"] in {"TRA", "MCOM", "ENG"}},
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
        cell = row.get("canonical_skills") or ""
        course_skills_map[row["code"]] = {s.strip() for s in cell.split(";") if s.strip()}
    course_name_map = dict(zip(courses["code"], courses["name"]))

    # Build career → top skills map for baseline + coverage
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

        cosine_recs = recommend_by_cosine(courses, career_skills_map.get(career, {}), top_k=10)
        cosine_codes = [r[0] for r in cosine_recs]

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
        print(f"\nAvg skill coverage @10 — PPR: {avg_ppr_cov:.3f}  Cosine: {avg_cos_cov:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run end-to-end**

```bash
pytest tests/test_baseline.py tests/test_evaluate.py -v
python -m src.eval.evaluate
```

Expected stdout: avg PPR coverage > avg cosine coverage (if not, debug graph weighting or PPR personalization vector — this is the signal that the project's centerpiece actually works).

- [ ] **Step 6: Commit + final review**

```bash
git add src/eval/ tests/test_baseline.py tests/test_evaluate.py
git commit -m "feat(eval): cosine baseline + 5 case studies + coverage/diversity metrics"
```

After all 8 tasks: dispatch a final code-review subagent on the full `main..HEAD` diff using `superpowers:requesting-code-review`. Verify graph stats look sane, PPR beats baseline, all tests pass with `-W error::FutureWarning`.

---

## Sanity Checklist Before Declaring Phase 2 Done

- [ ] All 8 tasks committed
- [ ] `pytest -q` shows ~60+ tests passing (Phase 1 = 34, Phase 2 adds ~25–30)
- [ ] `python -m src.eval.evaluate` produces `case_studies.md` with non-empty explanations
- [ ] Avg PPR skill coverage > avg Cosine baseline (if not: graph weights need tuning)
- [ ] `graph_stats.json` shows `is_weakly_connected: true` (if not: prune is too aggressive)
- [ ] No `FutureWarning` lines in stderr when running orchestrators
- [ ] Phase 2 ready to merge into `main` — open PR or ask user

## Phase 3 Hooks (out of scope here)

- Wire `prototype/SkillPath-Demo.html` to a tiny Flask endpoint that reads `graph.pkl` + `recommend_courses(...)` live
- Add MapReduce / Stream Data / Online Matching / Bloom Filter as "architectural extensibility" slides in the final PPT (lecture coverage commitment without forced implementations)
- Polish the 15-min presentation deck
