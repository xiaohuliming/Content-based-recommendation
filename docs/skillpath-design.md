# SkillPath — Design Document

**Date:** 2026-05-22
**Course:** Big Data Analytics, UIC (BNU-HKBU United International College)
**Execution:** Solo
**Status:** Draft v1 — pending user review

---

## 1. Project Positioning

**Title:** SkillPath: A Career-Oriented Personalized Course Recommendation System Using Big Data Analytics

**One-line pitch:** Recommend UIC courses to students based on their career goals and current skill profile, with explainable skill-gap analysis.

**What makes this different from generic course recommendation:**

- **Skill-Gap Driven** — the system models *what jobs require*, *what courses teach*, and *what students know*, then recommends courses that close the gap. Not "people similar to you also took..."
- **Heterogeneous graph + Personalized PageRank** is the algorithmic centerpiece. Direct mapping to Lecture 3.
- **Explainable** — every recommendation comes with the career-relevant skills it unlocks and the reasoning chain through the graph.

**Audience for demo/presentation:** UIC students and faculty. All deliverables (slides, demo UI, code comments) are in English to match UIC's language environment.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Data Layer                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ LinkedIn     │  │ UIC Courses  │  │ Synthetic        │   │
│  │ 124K jobs    │  │ (Excel+PDF)  │  │ Students         │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│  Processing Layer                                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │ Skill Extraction│ ─▶ │ Unified Skill Taxonomy        │    │
│  │ (LLM-assisted)  │    │ (~500 skills, 6 categories)   │    │
│  └─────────────────┘    └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│  Algorithm Layer                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Heterogeneous Graph                                     │  │
│  │   Career   ──needs──▶  Skill                            │  │
│  │   Course   ──teaches─▶ Skill                            │  │
│  │   Course   ──prereq──▶ Course                           │  │
│  │   Student  ──has────▶  Skill                            │  │
│  │   Student  ──took───▶  Course                           │  │
│  │                                                          │  │
│  │ Personalized PageRank                                   │  │
│  │   teleport set = {target career} ∪ {missing skills}     │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│  Presentation Layer                                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Demo UI (extends prototype/SkillPath-Demo.html)         │  │
│  │  • Input: career + completed courses + skill self-rating│  │
│  │  • Output: ranked course list with per-course reasons   │  │
│  │  • Visualization: skill-gap chart + PageRank explanation│  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Data Sources

### 3.1 Jobs — LinkedIn Postings 2023–2024

- **Location:** `data/archive.zip` (Kaggle dataset)
- **Scale:** 124K real postings, full English JD text
- **Key fields used:** `job_id, title, description, normalized_salary, location, formatted_experience_level, listed_time, skills_desc`
- **Filtering:** None — keep all 124K per the "no filter" decision. Story is "international career market across all functions."
- **Caveat:** LinkedIn's `skills` mapping field is only ~30 high-level *job functions* (FIN, MRKT, ENG…), not technical skills. Our technical skills come from extracting `description` text directly.

### 3.2 Courses — UIC (two-source join)

**Course descriptions PDF** — `data/Course Descriptions_20260421.pdf`

- 172 pages, ~1300 courses (full UIC catalog)
- Per-course fields: code, name, units, prerequisites, full description text
- Format is highly regular — parseable with regex

**Course timetable Excel** — `data/Course List and Timetable_Semester 2 of AY2025-26_20260224.xls`

- 2443 session rows, ~600–800 unique courses currently offered (Sem 2 AY2025-26)
- Fields: code, title, programme, units, schedule, classroom, teacher, requirements

**Join:** PDF is the canonical course catalog. Excel adds *current-semester* metadata (timetable, instructor) where available. Join key: course code.

**Master course record schema:**

```
course_id, code, name, units, programme, department,
description,                  # from PDF
prerequisites_text,           # raw string from PDF
prerequisite_ids[],           # parsed course codes
schedule, classroom, teacher, # from Excel if offered this sem
is_offered_current_sem (bool),
extracted_skills[]            # from skill extraction pipeline
```

### 3.3 Students — Synthetic

- **Seed:** 1–2 real profiles authored by hand (self + imagined classmate)
- **Generation:** LLM (Claude/GPT-4) synthesizes 500–1000 students from 6–8 personas: Data Analyst track / ML Engineer / Backend / NLP / Quant / Product / Design / Business Analytics
- **Schema:**

```
student_id, major, year, target_career,
completed_courses[],
skill_ratings { skill_id : 1-5 },
interest_prefs { theoretical|applied, research|career, ... },
max_credits, constraints[]
```

### 3.4 Skill Taxonomy — the cross-cutting backbone

This is the critical glue. Without a clean skill taxonomy, the graph cannot be built.

**Construction pipeline:**

1. LLM extracts candidate technical skills from LinkedIn `description` field (sample ~5K representative postings to avoid runaway cost).
2. LLM extracts candidate technical skills from PDF course descriptions.
3. Merge, lowercase-normalize, deduplicate.
4. Synonym consolidation (Python / PYTHON / py → "Python"). Manual review of top ~200.
5. Categorize into 6 buckets: Programming Languages, ML/AI, Data Systems, Math/Stats, Domain Knowledge, Soft Skills.

**Target size:** 500–800 unique skills.

**Schema:**

```
skill_id, name, category, aliases[], parent_skill (optional)
```

---

## 4. Graph Edges (derived from data)

| Edge | Source | Computation |
|---|---|---|
| career → skill (weighted) | LinkedIn | Cluster jobs by normalized title, count skill frequencies in JD text, normalize → weight |
| course → skill (weighted) | PDF descriptions | LLM extraction + TF-IDF cross-check; weight = confidence |
| course → course (prereq) | PDF prerequisites field | Regex parse of "Pre-requisite(s): X" |
| student → skill (mastery) | Self-rating + completed-course inference | rating ∈ {0..5} |
| student → course (took) | Student profile | direct |

---

## 5. Core Algorithm: Personalized PageRank

**Why PageRank is the right centerpiece:**

- Maps directly to Lecture 3 — the "PageRank, teleport, personalization" trio is exactly the course's headline concept.
- Naturally handles heterogeneous nodes (career / skill / course / student all live in the same graph).
- The teleport mechanism is the perfect knob for personalization to a student's career goal.
- Explainable — high PageRank on a course can be traced backward through skill nodes to the teleport set.

**Formulation:**

- `r = β · M·r + (1−β) · p`
- `M` = column-normalized adjacency matrix of the heterogeneous graph
- `p` = teleport distribution, mass concentrated on `{target_career_node} ∪ {student's missing_skill_nodes}`
- `β = 0.85` (standard)
- Iterate ~50 times to convergence

**Output:** rank course nodes by PageRank score. Then apply constraint filters:

- Drop courses already taken.
- Drop courses with unmet prerequisites.
- Drop courses with schedule conflicts (using Excel timetable).
- Cap to `max_credits` budget.

**Explanation generation:**

For each recommended course, traverse the graph backward and surface the top-3 paths from the course to the teleport set. Render as natural language:

> "Recommended because:
> 1. Teaches **Python** — Data Analyst careers weight this skill 0.42
> 2. Bridges to **Machine Learning**, which 4 of your target-career's top-10 skills depend on
> 3. Prerequisite for AI3013 (Machine Learning), which you've flagged as desired"

---

## 6. Other Lecture Concepts — Coverage Strategy

The course has 8 lectures. Trying to genuinely implement all 8 in solo execution would result in shallow coverage of each. Strategy: **build 2 deeply, frame the rest as architectural extensions** in the presentation.

| Lecture | Coverage strategy | Built? |
|---|---|---|
| L1 KDD framing | Project's overall framing — "valid, useful, unexpected, understandable" patterns | Built into narrative |
| L2 MapReduce | Skill frequency aggregation across 124K jobs is the natural use case. In demo: "we run this in-memory on 124K which fits; the same logic scales to Spark/Hadoop on 100M+ jobs." | Frame only — show the map/reduce pseudocode, run single-machine |
| L3 PageRank | **Centerpiece. Built.** | ✅ Built |
| L4 Online Matching / AdWords | One slide: "course-capacity allocation as online bipartite matching, with BALANCE algorithm." | Frame only |
| L5 MinHash / LSH | Course-description similarity for "related courses" feature | Optional add-on (1–2 days) |
| L6 Stream Data | Use LinkedIn `listed_time` to simulate stream; sliding-window skill trend | Optional add-on (2 days) |
| L7 Clustering | K-means on student skill vectors → student archetypes | ✅ Built (cheap, ~1 day) |
| L8 Recommendation | This is what the whole system *is*. Content-based + graph-based. | ✅ Built |

**Net coverage:** L3, L7, L8 fully built; L1 framing throughout; L2/L4/L5/L6 acknowledged. Final presentation should be honest about which are built vs framed.

---

## 7. Demo UI

- Extend existing `prototype/SkillPath-Demo.html`
- **Input form:** career dropdown, completed-courses multi-select, skill self-rating sliders
- **Output:**
  - Top 10 recommended courses with PageRank score
  - Skill-gap radar chart (target-career skill profile vs current mastery)
  - Per-course: 3-bullet explanation
  - "Why not these?" panel — show 3 high-PR courses that were *filtered out* and why (prereq missing, schedule conflict, off-topic)
- **Implementation note:** keep UI static HTML + JS, with backend computation **pre-computed offline** and loaded as JSON. No live backend needed for a demo.

---

## 8. Evaluation Strategy

**No ground truth → no fake Precision@K.** Instead:

1. **Skill coverage Δ (hard metric):**
   - Before recs: fraction of target-career's top-20 skills the student has at rating ≥ 3
   - After taking top-5 recs (project the skills those courses cover): projected fraction
   - Report Δ averaged over the 500–1000 synthetic students
2. **Baseline comparison:** rank courses by simple popularity (enrollment count, or PageRank with uniform teleport) vs Personalized PageRank. Compare skill-coverage Δ.
3. **Qualitative narrative:** walk through 2–3 hand-crafted student personas in the presentation. Narrate why the recs make sense.
4. **Explainability check:** every recommendation has reasons. Count "no-explanation" failures (should be 0).

**Skipped:** user satisfaction surveys — no time, no audience, would be ceremonial.

---

## 9. Scope: In and Out

**In (built and demoed):**
- Full data pipeline (PDF parser, Excel parser, LinkedIn loader, skill extractor, graph builder)
- Personalized PageRank with constraint filtering and explanation generation
- K-means student clustering (second algorithmic technique)
- Demo UI with 2–3 student personas walked through live
- Skill-coverage Δ evaluation + baseline comparison

**Out (mentioned in presentation as architectural extension, not built):**
- True distributed MapReduce execution (single-machine version stands in)
- Online matching algorithm for course allocation
- Real user studies
- Collaborative filtering (no historical student-choice data exists)

**Optional add-ons (if Phase 1–4 finish ahead of pace):**
- LSH for course similarity (~1–2 days)
- Stream data with DGIM-style window on LinkedIn time-stamps (~2 days)

**Honesty principle:** in the presentation, when discussing un-built modules, explicitly label them as "architecture extension point" not "implemented." This is *more* impressive than pretending, because it shows you understand the techniques' scope.

---

## 10. Implementation Phases

Time-flexible. Natural execution order:

| Phase | Deliverable | Est. solo-days |
|---|---|---|
| 1 | Data foundation: PDF/Excel/LinkedIn parsers, skill extraction, skill taxonomy | 3–5 |
| 2 | Heterogeneous graph + Personalized PageRank + constraint filters + explanation generator | 3–5 |
| 3 | Synthetic students, batch recommendations, persona crafting | 2–3 |
| 4 | Demo UI extension, pre-computed JSON, end-to-end demo for 3 personas | 2–3 |
| 5 | Evaluation: skill-coverage Δ, baseline comparison, K-means clustering add-on | 2–3 |
| 6 | (Optional) LSH and/or Stream modules | 2–4 |
| 7 | Slides, dry run, screen recording backup | 2–3 |

**Total core (Phases 1–5, 7):** 14–22 solo-days at sustainable pace.

---

## 11. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Skill extraction produces noisy/duplicate skills | High | Budget explicit cleanup time; manual review of top-200; iterate LLM prompts |
| PDF parser fails on irregular formatting in some courses | Medium | Inspect 50 random courses' parsed output; fallback to title-only mode for failures |
| LinkedIn 124K dominated by non-tech jobs distorts career-skill weights | Medium | For weight derivation, group by job-title cluster so each career's weights come from its own posting subset, not the global mix |
| Synthetic students feel artificial in demo | Medium | Hand-craft the 2–3 demo personas in detail; use LLM-generated bulk only for aggregate evaluation |
| PageRank doesn't converge or gives weird ranks | Low | Standard algorithm; verify on a 10-node toy graph first; check stochastic matrix is column-stochastic with dead-end fixup |
| Scope creep — adding L4/L5/L6 modules and missing core | Medium | Strict gate: don't start optional add-ons until Phases 1–5 are demo-ready |

---

## 12. Repository Structure (proposed)

```
小组项目/
├── data/
│   ├── archive.zip
│   ├── Course List and Timetable_Semester 2 of AY2025-26_20260224.xls
│   └── Course Descriptions_20260421.pdf
├── docs/
│   └── skillpath-design.md          # this file
├── prototype/
│   └── SkillPath-Demo.html          # existing UI mockup
├── src/
│   ├── parsers/                     # PDF + Excel + LinkedIn loaders
│   ├── skills/                      # skill extraction + taxonomy
│   ├── graph/                       # heterogeneous graph + PageRank
│   ├── recommend/                   # recommendation + constraints + explanations
│   ├── students/                    # synthetic student generation
│   └── eval/                        # skill-coverage metric + baseline
├── notebooks/                       # exploration + figures
└── output/
    ├── courses_master.csv
    ├── jobs_processed.csv
    ├── skill_taxonomy.csv
    ├── students_synthetic.csv
    └── recommendations/             # JSON per persona
```

---

## 13. Open Questions

None blocking. Items deferred to implementation plan:

- Concrete LLM choice for skill extraction (Claude vs GPT-4) and cost budget
- Whether to use NetworkX (simpler, in-memory) or Neo4j (richer, more impressive in demo) for the graph
- Visualization library for the graph and skill-gap chart (D3 vs Plotly vs static matplotlib export)

These will be decided in the implementation plan based on what's fastest for solo execution.

---

## Next Step

After user review and approval of this design, transition to the **implementation plan**: each phase broken into concrete tasks with file paths, function signatures, and verification checkpoints.
