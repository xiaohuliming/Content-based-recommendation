"""Build an interactive HTML report exploring Phase 1 outputs.

Usage (from project root, with .venv activated):
    python -m src.analysis.phase1_report

Writes: output/phase1_analysis.html
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "phase1_analysis.html"


def _explode(series: pd.Series) -> pd.Series:
    """Split comma-joined skill cells into one skill per row, stripping whitespace."""
    return (
        series.fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .pipe(lambda s: s[s.str.len() > 0])
    )


def _variant_clusters(skills: list[str], top_n: int = 15) -> list[dict]:
    """Heuristic variant clustering — demo of what LSH would do at scale.

    Pairs skills that share a 2-gram of tokens (e.g., "warehouse"+"management").
    Drops giant blobs caused by transitive runaway. Real Phase 2 should use
    MinHash + LSH on character shingles for better precision.
    """
    STOP = {"and", "of", "the", "for", "to", "in", "on", "a", "an", "with",
            "management", "system", "systems", "analysis", "service", "services"}

    def tokens(s: str) -> frozenset[str]:
        return frozenset(re.findall(r"\b[a-z0-9]+\b", s.lower())) - STOP

    # Index skills by 2-grams of their tokens
    from itertools import combinations
    bigram_to_skills: dict[frozenset[str], list[int]] = {}
    skill_tokens = [tokens(s) for s in skills]
    for i, toks in enumerate(skill_tokens):
        if len(toks) < 2:
            continue
        for pair in combinations(sorted(toks), 2):
            bigram_to_skills.setdefault(frozenset(pair), []).append(i)

    parent = list(range(len(skills)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for bigram, members in bigram_to_skills.items():
        if len(members) > 30:  # bigram too common — not a strong variant signal
            continue
        for s in members[1:]:
            union(members[0], s)

    clusters: dict[int, list[str]] = {}
    for i, s in enumerate(skills):
        clusters.setdefault(find(i), []).append(s)

    # Drop tiny and giant clusters — only meaningful "variant groups" stay
    candidates = [
        {"size": len(v), "members": v}
        for v in clusters.values()
        if 3 <= len(v) <= 30
    ]
    return sorted(candidates, key=lambda x: -x["size"])[:top_n]


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    courses = pd.read_csv(OUTPUT_DIR / "courses_skills.csv")
    jobs = pd.read_csv(OUTPUT_DIR / "jobs_sample_skills.csv")
    tax = pd.read_csv(OUTPUT_DIR / "skill_taxonomy.csv")

    # --- per-skill stats ---
    courses_skills = _explode(courses["extracted_skills"])
    jobs_skills = _explode(jobs["extracted_skills"])
    courses_set = set(courses_skills.unique())
    jobs_set = set(jobs_skills.unique())

    only_courses = courses_set - jobs_set
    only_jobs = jobs_set - courses_set
    both = courses_set & jobs_set

    # --- per-document stats ---
    n_skills_per_course = courses["extracted_skills"].fillna("").apply(
        lambda s: 0 if s == "" else len(s.split(","))
    )
    n_skills_per_job = jobs["extracted_skills"].fillna("").apply(
        lambda s: 0 if s == "" else len(s.split(","))
    )

    # --- cumulative coverage curve (top-N skills cover X% of mentions) ---
    tax_sorted = tax.sort_values("doc_frequency", ascending=False).reset_index(drop=True)
    total_mentions = tax_sorted["doc_frequency"].sum()
    tax_sorted.loc[:, "cum_pct"] = tax_sorted["doc_frequency"].cumsum() / total_mentions * 100

    # --- variant clusters (top 15 by size) ---
    clusters = _variant_clusters(tax_sorted["skill"].tolist(), top_n=15)

    # ============== build the report ==============
    fig_freq = go.Figure()
    bins = [1, 2, 3, 6, 11, 26, 51, 101, 501, 1001]
    labels = ["1x", "2x", "3-5x", "6-10x", "11-25x", "26-50x", "51-100x", "101-500x", "501+"]
    counts = []
    for i in range(len(bins) - 1):
        c = ((tax["doc_frequency"] >= bins[i]) & (tax["doc_frequency"] < bins[i + 1])).sum()
        counts.append(int(c))
    counts.append(int((tax["doc_frequency"] >= bins[-1]).sum()))
    fig_freq.add_trace(go.Bar(
        x=labels + ["1000+"],
        y=counts + [int((tax["doc_frequency"] >= 1000).sum())],
        text=counts + [int((tax["doc_frequency"] >= 1000).sum())],
        textposition="outside",
        marker_color=["#ef4444", "#f59e0b", "#f59e0b",
                      "#84cc16", "#84cc16", "#22c55e",
                      "#10b981", "#06b6d4", "#3b82f6", "#6366f1"],
    ))
    fig_freq.update_layout(
        title="Skill frequency distribution — 67% of skills appear only once (long-tail noise)",
        xaxis_title="Document frequency bucket",
        yaxis_title="Number of unique skills",
        yaxis_type="log",
        height=400,
    )

    # --- coverage curve ---
    fig_cov = go.Figure()
    fig_cov.add_trace(go.Scatter(
        x=tax_sorted.index + 1,
        y=tax_sorted["cum_pct"],
        mode="lines",
        line=dict(color="#3b82f6", width=2),
        name="cumulative %",
    ))
    # annotate key thresholds
    for n in [100, 500, 1000, 3000, 5000]:
        if n <= len(tax_sorted):
            pct = tax_sorted.iloc[n - 1]["cum_pct"]
            fig_cov.add_annotation(
                x=n, y=pct,
                text=f"top {n} = {pct:.0f}%",
                showarrow=True,
                arrowhead=2,
                ax=20, ay=-30,
            )
    fig_cov.update_layout(
        title="Cumulative coverage — how many skills do we actually need?",
        xaxis_title="Top-N skills (log scale)",
        yaxis_title="% of all skill mentions covered",
        xaxis_type="log",
        height=400,
    )

    # --- category breakdown ---
    cat_counts = tax["category"].value_counts()
    fig_cat = go.Figure(go.Bar(
        x=cat_counts.values,
        y=cat_counts.index,
        orientation="h",
        marker_color="#6366f1",
        text=cat_counts.values,
        textposition="outside",
    ))
    fig_cat.update_layout(
        title=f"Category breakdown — 94% of skills landed in 'Other' (rule-based mapper too narrow)",
        xaxis_title="Number of unique skills",
        height=350,
    )

    # --- skills per doc (jobs vs courses, side by side) ---
    fig_per_doc = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Skills per course", "Skills per job"),
    )
    fig_per_doc.add_trace(
        go.Histogram(x=n_skills_per_course, nbinsx=20, marker_color="#10b981", name="courses"),
        row=1, col=1,
    )
    fig_per_doc.add_trace(
        go.Histogram(x=n_skills_per_job, nbinsx=20, marker_color="#f59e0b", name="jobs"),
        row=1, col=2,
    )
    fig_per_doc.update_layout(
        title="Skills per document (courses tend to be terse; jobs are richer)",
        showlegend=False,
        height=350,
    )

    # --- overlap venn-as-bar ---
    fig_overlap = go.Figure(go.Bar(
        x=["Courses ONLY", "BOTH (the bridge)", "Jobs ONLY"],
        y=[len(only_courses), len(both), len(only_jobs)],
        text=[len(only_courses), len(both), len(only_jobs)],
        textposition="outside",
        marker_color=["#10b981", "#8b5cf6", "#f59e0b"],
    ))
    fig_overlap.update_layout(
        title=f"Skill overlap: courses vs jobs — the 'bridge' is what the recommendation depends on",
        yaxis_title="Number of unique skills",
        height=350,
    )

    # --- top skills tables ---
    top30 = tax_sorted.head(30)[["skill", "doc_frequency", "category"]]

    # categorize where each top skill came from
    top30 = top30.copy()
    top30.loc[:, "in_courses"] = top30["skill"].isin(courses_set)
    top30.loc[:, "in_jobs"] = top30["skill"].isin(jobs_set)
    top30.loc[:, "source"] = top30.apply(
        lambda r: "both" if r["in_courses"] and r["in_jobs"]
        else ("courses only" if r["in_courses"] else "jobs only"),
        axis=1,
    )
    top30_html = top30[["skill", "doc_frequency", "category", "source"]].to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=False,
    )

    # cluster examples — pick the most striking 8
    clusters_top = clusters[:8]
    cluster_html_rows = []
    for c in clusters_top:
        # show first 6 members
        sample = c["members"][:6]
        more = f" ... +{len(c['members']) - 6} more" if len(c["members"]) > 6 else ""
        cluster_html_rows.append(
            f'<tr><td><b>{c["size"]}</b></td><td>{", ".join(sample)}{more}</td></tr>'
        )
    cluster_html = (
        "<table class='data-table'><thead><tr><th>Group size</th>"
        "<th>Variants the LLM produced (would collapse to 1 skill node after dedup)</th></tr></thead>"
        "<tbody>" + "".join(cluster_html_rows) + "</tbody></table>"
    )

    # ============== render HTML ==============
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SkillPath — Phase 1 Data Analysis</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 1100px; margin: 24px auto; padding: 0 24px;
    color: #1f2937; line-height: 1.5;
  }}
  h1 {{ border-bottom: 2px solid #6366f1; padding-bottom: 8px; }}
  h2 {{ margin-top: 32px; color: #4338ca; }}
  .stats {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    margin: 16px 0;
  }}
  .stat {{
    background: #f3f4f6; padding: 12px 16px; border-radius: 8px;
    border-left: 4px solid #6366f1;
  }}
  .stat .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
  .stat .value {{ font-size: 22px; font-weight: 700; color: #111827; }}
  .data-table {{
    width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px;
  }}
  .data-table th {{ background: #6366f1; color: white; padding: 8px; text-align: left; }}
  .data-table td {{ padding: 6px 8px; border-bottom: 1px solid #e5e7eb; }}
  .data-table tr:nth-child(even) {{ background: #f9fafb; }}
  .callout {{
    background: #fef3c7; border-left: 4px solid #f59e0b;
    padding: 12px 16px; margin: 16px 0; border-radius: 4px;
  }}
  .callout-good {{
    background: #d1fae5; border-left-color: #10b981;
  }}
  .callout-bad {{
    background: #fee2e2; border-left-color: #ef4444;
  }}
  .plot {{ margin: 20px 0; }}
</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>

<h1>SkillPath — Phase 1 Data Foundation Analysis</h1>
<p>Built from <code>courses_skills.csv</code>, <code>jobs_sample_skills.csv</code>,
<code>skill_taxonomy.csv</code>. Run <code>python -m src.analysis.phase1_report</code> to regenerate.</p>

<div class="stats">
  <div class="stat"><div class="label">UIC Courses</div><div class="value">{len(courses):,}</div></div>
  <div class="stat"><div class="label">Job Postings</div><div class="value">{len(jobs):,}</div></div>
  <div class="stat"><div class="label">Unique Skills</div><div class="value">{len(tax):,}</div></div>
  <div class="stat"><div class="label">Empty courses</div><div class="value">{(n_skills_per_course==0).sum()}</div></div>
</div>

<h2>1. The long-tail problem</h2>
<div class="callout callout-bad">
  <b>67% of skills appear in exactly one document.</b> These are mostly
  paraphrasing variants the LLM produced ("Resort Management" vs
  "Hotel Operations" vs "Hospitality Management") — counted as 3 distinct nodes
  in any graph built from this taxonomy.
</div>
<div class="plot" id="freq"></div>

<h2>2. How many skills do we actually need?</h2>
<div class="callout">
  Even the top-1000 skills cover most of what shows up in the data.
  Below shows the cumulative coverage curve — annotated at key thresholds.
</div>
<div class="plot" id="cov"></div>

<h2>3. Top 30 skills — the "core" we'd keep after pruning</h2>
{top30_html}

<h2>4. Variant clusters that should merge (LSH/dedup target)</h2>
<div class="callout-good callout">
  Below are 8 clusters of skills that share enough tokens to suggest they're
  variants of the same underlying concept. This is exactly what
  <b>MinHash + LSH</b> (course lecture 4) can collapse automatically.
  Going from 21,105 → ~3,000 nodes would make Phase 2's PageRank
  graph far cleaner.
</div>
{cluster_html}

<h2>5. Category breakdown</h2>
<div class="callout callout-bad">
  The hardcoded keyword-based categorizer only matched 6% of skills.
  The "Other" bucket (19,911 skills) is too big to be useful for any
  category-level analysis. Either expand the keyword list, or drop
  categorization entirely and let clustering/PageRank define topics implicitly.
</div>
<div class="plot" id="cat"></div>

<h2>6. Skills per document</h2>
<div class="plot" id="per_doc"></div>

<h2>7. The skill-gap bridge</h2>
<div class="callout">
  <b>This is the load-bearing slide of the project.</b>
  Skills that appear in BOTH courses and jobs are what makes
  "career → course" recommendations possible. Skills only on one side
  are either:
  <ul>
    <li><b>Jobs-only</b> = career demand the curriculum doesn't currently teach (gap to flag)</li>
    <li><b>Courses-only</b> = academic content with no obvious job-market signal (skills students learn that don't show up in JDs)</li>
  </ul>
</div>
<div class="plot" id="overlap"></div>

<h2>Phase 2 implications</h2>
<ol>
  <li><b>Dedup first, graph after.</b> Run MinHash + LSH on the
      21K skill names; collapse near-duplicates. Target: ~3,000 canonical skills.</li>
  <li><b>Frequency-prune aggressively.</b> Drop skills with doc_frequency &lt; 5.
      Cuts another ~50% of nodes with minimal information loss.</li>
  <li><b>Re-extract empty courses</b> (116 humanities/communications courses
      went to []). Likely a prompt issue — soft skills are valid for those.</li>
  <li><b>Skip the rule-based categorizer.</b> 94% "Other" means it's noise.
      Use PageRank community detection or K-means on the skill co-occurrence
      matrix instead.</li>
</ol>

<script>
const plots = {{
  freq:    {fig_freq.to_json()},
  cov:     {fig_cov.to_json()},
  cat:     {fig_cat.to_json()},
  per_doc: {fig_per_doc.to_json()},
  overlap: {fig_overlap.to_json()},
}};
for (const [id, fig] of Object.entries(plots)) {{
  Plotly.newPlot(id, fig.data, fig.layout, {{responsive: true}});
}}
</script>

</body>
</html>
"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {REPORT_PATH}  ({REPORT_PATH.stat().st_size/1024:.0f} KB)")
    print(f"\nopen with: open {REPORT_PATH}")


if __name__ == "__main__":
    build()
