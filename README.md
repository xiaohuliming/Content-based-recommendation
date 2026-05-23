# SkillPath — Career-Oriented Course Recommender

A knowledge-graph + Personalized PageRank (PPR) system that takes a student profile
(major / year / completed courses / career goal) and returns ranked course
recommendations with bridge-skill explanations.

Built for the *Big Data* group project — Phase 1 (data foundation) → Phase 2 (PPR
engine) → Phase 3 (live Flask demo).

---

## Quickstart — Run the Demo

Prereqs: Python 3.14+, `.venv` activated, Phase 1 & 2 outputs in `output/phase2/`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m src.demo.app
# Open http://127.0.0.1:5000/
```

Enter a career goal (autocomplete is populated from the LinkedIn job titles) and a
comma-separated list of completed course codes (e.g., `COMP1003, MATH1003`). The page
shows top-10 PPR-ranked course recommendations with bridge-skill explanations.

To regenerate Phase 2 outputs from scratch:

```bash
python -m src.normalize.apply_canonical
python -m src.students.synthetic
python -m src.graph.builder
python -m src.students.clustering
python -m src.eval.evaluate
```

---

## Project Structure

```
src/
├── normalize/       # canonical skill normalisation
├── graph/           # heterogeneous graph builder
├── ppr/             # PPR engine + explanation generator
├── students/        # synthetic student profiles + K-means clustering
├── eval/            # cosine baseline, case studies, coverage/diversity metrics
└── demo/            # Flask app + transient student insertion
data/                # raw course / job-posting CSVs
output/phase2/       # pickled graph + evaluation artefacts
prototype/           # static HTML mockup → now live via Flask
tests/               # pytest suite (97 tests)
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest -W error::FutureWarning -q
```
