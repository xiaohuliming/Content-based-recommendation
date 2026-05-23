"""Skill name deduplication via MinHash + LSH.

Why LSH: 21K skills -> 440M pairwise comparisons with edit distance.
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
    threshold: float = 0.4,           # LSH candidate-gen threshold (cast wide net)
    verify_threshold: float = 0.7,    # exact Jaccard threshold for actual union
    num_perm: int = 128,
    shingle_k: int = 3,
    seed: int = 42,
) -> dict[int, list[str]]:
    """Cluster skills by Jaccard similarity via LSH + exact verification.

    LSH at `threshold` gives candidate pairs (sublinear, may have false positives
    from common suffix shingles). Each candidate pair is then verified by computing
    the EXACT Jaccard over the cached shingle sets — only pairs with exact Jaccard
    >= `verify_threshold` are merged. This two-stage design prevents transitive
    Union-Find runaway where unrelated skills get blobbed together via shared 3-grams.

    Returns {cluster_id: [original_skill_names]}. Every input skill appears in
    exactly one cluster (singletons get their own ID).
    """
    if not skills:
        return {}

    skills = list(dict.fromkeys(skills))  # dedupe input, preserve order

    # Build LSH index AND keep shingle sets for exact verification
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}
    shingle_sets: dict[str, set[str]] = {}
    for s in skills:
        norm = normalize_skill_name(s)
        shingles = char_shingles(norm, k=shingle_k)
        if not shingles:
            continue
        shingle_sets[s] = shingles
        m = _build_minhash(shingles, num_perm=num_perm, seed=seed)
        minhashes[s] = m
        lsh.insert(s, m)

    # Union-Find — only union pairs whose EXACT Jaccard clears verify_threshold
    parent: dict[str, str] = {s: s for s in minhashes}

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
            if candidate == s:
                continue
            sa, sb = shingle_sets[s], shingle_sets[candidate]
            exact_jaccard = len(sa & sb) / len(sa | sb)
            if exact_jaccard >= verify_threshold:
                union(s, candidate)

    # Group by root
    clusters: dict[str, list[str]] = {}
    for s in minhashes:
        clusters.setdefault(find(s), []).append(s)

    # Reassign deterministic integer IDs by sorted canonical name
    sorted_roots = sorted(clusters.keys(), key=lambda r: (clusters[r][0].lower(), r))
    return {i: sorted(clusters[root]) for i, root in enumerate(sorted_roots)}
