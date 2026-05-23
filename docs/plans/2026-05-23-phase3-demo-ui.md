# Phase 3 — Demo UI Wiring Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` in 精简模式 — short subagent reports (<150 words), combined spec+quality review, no double-verification by controller.

**Goal:** Take Phase 2's pickled graph + PPR engine and serve them through a Flask backend so the existing `prototype/SkillPath-Demo.html` mockup becomes a live, interactive demo where a user can enter a new student profile (major / year / completed courses / career goal) and see real PPR recommendations + bridge-skill explanations.

**Architecture:** Single Flask app. `GET /` returns the prototype HTML (now wired). `POST /api/recommend` accepts a JSON student profile, transiently inserts the student into a copy of the loaded graph, runs PPR + explain, returns ranked recommendations with explanations. No persistence — every request is stateless.

**Tech Stack:** Flask (new dep), existing Phase 2 modules (`src.ppr.engine`, `src.ppr.explain`, `src.graph.builder` for the transient-insert helper). Browser-only frontend (no React/build tools — keep it shippable for the 15-min presentation).

---

## File Structure

```
src/
├── demo/
│   ├── __init__.py
│   ├── transient_student.py   # add_transient_student(g, profile) → new_student_node
│   └── app.py                 # Flask app with 2 routes
prototype/
└── SkillPath-Demo.html         # MODIFIED: form → fetch /api/recommend → render results
tests/
├── test_transient_student.py
└── test_demo_api.py            # Flask test client
```

No new outputs — runtime serves recommendations on demand.

---

## Design Rationale (read first)

**Why Flask, not FastAPI?** One-file, stable, no async coordination needed. Phase 1/2 are sync — keep Phase 3 sync.

**Why transient student insertion (not pre-baked)?** The demo's value is "anyone can enter their profile and see real recommendations". Pre-baked synthetic students can't do that. Insertion is O(N courses × ~5 skill edges) ≈ a few ms on the 8K-node graph.

**Why not React?** Build pipeline = friction. Demo lives in one HTML file → fetch → JSON → DOM. Done.

**State**: graph.pkl loaded ONCE at server startup. Per-request, we `copy.copy(g)` (shallow — node attrs are immutable, edge dict is copy-on-modify) then add student edges. About 50ms per request total.

**Form schema**: `{major: str, year: int, completed_courses: list[str], career_goal: str}`. The career_goal must match a known career node (HTML provides autocomplete from the loaded careers).

---

## Task 1 — Transient Student Insertion

**Files:** `src/demo/__init__.py` (empty), `src/demo/transient_student.py`, `tests/test_transient_student.py`

- [ ] **Step 1: Tests**

```python
# tests/test_transient_student.py
import networkx as nx
import pandas as pd
from src.demo.transient_student import add_transient_student


def _toy_graph():
    g = nx.DiGraph()
    # 2 courses, 2 skills
    g.add_node("skill:Python", type="skill", doc_frequency=10)
    g.add_node("skill:SQL", type="skill", doc_frequency=10)
    g.add_node("course:C1", type="course", name="Intro")
    g.add_node("course:C2", type="course", name="DB")
    g.add_edge("course:C1", "skill:Python", weight=0.7, edge_type="course-skill")
    g.add_edge("skill:Python", "course:C1", weight=0.7, edge_type="skill-course")
    g.add_edge("course:C2", "skill:SQL", weight=0.7, edge_type="course-skill")
    g.add_edge("skill:SQL", "course:C2", weight=0.7, edge_type="skill-course")
    return g


def test_add_transient_student_creates_node():
    g = _toy_graph()
    sid = add_transient_student(g, completed_courses=["C1"], student_id="DEMO")
    assert sid == "student:DEMO"
    assert g.nodes[sid]["type"] == "student"


def test_student_completed_edges_added():
    g = _toy_graph()
    sid = add_transient_student(g, completed_courses=["C1", "C2"], student_id="DEMO")
    assert g.has_edge(sid, "course:C1")
    assert g.has_edge(sid, "course:C2")
    assert g[sid]["course:C1"]["edge_type"] == "student-course"


def test_student_skill_edges_normalized():
    g = _toy_graph()
    sid = add_transient_student(g, completed_courses=["C1", "C2"], student_id="DEMO")
    # student-skill weights should be in [0, 1] (same normalization as Task 4)
    skill_edges = [d["weight"] for _, t, d in g.out_edges(sid, data=True)
                   if d.get("edge_type") == "student-skill"]
    assert skill_edges
    assert max(skill_edges) <= 1.0


def test_unknown_completed_courses_silently_skipped():
    g = _toy_graph()
    sid = add_transient_student(g, completed_courses=["C1", "NONEXIST"], student_id="X")
    assert g.has_edge(sid, "course:C1")
    assert not g.has_edge(sid, "course:NONEXIST")


def test_duplicate_ids_raise():
    g = _toy_graph()
    add_transient_student(g, completed_courses=["C1"], student_id="DUP")
    try:
        add_transient_student(g, completed_courses=["C2"], student_id="DUP")
        assert False, "should have raised"
    except ValueError:
        pass
```

- [ ] **Step 2: Implement**

```python
# src/demo/transient_student.py
"""Insert a transient student into an existing graph for live PPR recommendation.

Mirrors the student-edge logic from src.graph.builder but for one student at a time,
without rebuilding the whole graph. Used by the Flask demo's /api/recommend endpoint.
"""
from collections import defaultdict

import networkx as nx


def add_transient_student(
    g: nx.DiGraph,
    completed_courses: list[str],
    *,
    student_id: str,
    career_goal: str = "",
    year: int = 0,
) -> str:
    """Add a student node + student-course + student-skill (normalized) edges in place.

    Returns the full node id (e.g. 'student:DEMO'). Unknown courses are silently
    skipped (no error). Raises ValueError if `student_id` is already in the graph.
    """
    sid = f"student:{student_id}"
    if g.has_node(sid):
        raise ValueError(f"student node {sid!r} already in graph — pick a unique id")

    g.add_node(sid, type="student", career_goal=career_goal, year=year)

    student_skill_score: dict[str, float] = defaultdict(float)
    for code in completed_courses:
        course_node = f"course:{code}"
        if not g.has_node(course_node):
            continue
        g.add_edge(sid, course_node, weight=1.0, edge_type="student-course")
        for skill_node in g.successors(course_node):
            if g.nodes[skill_node].get("type") == "skill":
                student_skill_score[skill_node] += g[course_node][skill_node]["weight"]

    if student_skill_score:
        max_w = max(student_skill_score.values())
        if max_w > 0:
            student_skill_score = {k: v / max_w for k, v in student_skill_score.items()}

    for skill_node, w in student_skill_score.items():
        g.add_edge(sid, skill_node, weight=w, edge_type="student-skill")

    return sid
```

- [ ] **Step 3: Run + commit**

```bash
source .venv/bin/activate
pytest tests/test_transient_student.py -v
pytest -W error::FutureWarning -q
git add src/demo/__init__.py src/demo/transient_student.py tests/test_transient_student.py
git commit -m "feat(demo): transient student insertion for live PPR"
```

Expected: +5 tests, full suite 93.

---

## Task 2 — Flask Backend (`src/demo/app.py`)

**Files:** `src/demo/app.py`, `tests/test_demo_api.py`

Add to `requirements.txt`: `flask>=3.0,<4.0`.

- [ ] **Step 1: Install + verify**

```bash
echo 'flask>=3.0,<4.0' >> requirements.txt
pip install -r requirements.txt
python -c "import flask; print('flask', flask.__version__)"
```

- [ ] **Step 2: Tests**

```python
# tests/test_demo_api.py
import json
import pickle
import pytest
import networkx as nx
from src.demo.app import create_app


@pytest.fixture
def client(tmp_path):
    # Build a tiny throwaway graph and pickle it
    g = nx.DiGraph()
    g.add_node("skill:Python", type="skill", doc_frequency=10)
    g.add_node("course:C1", type="course", name="Intro")
    g.add_edge("course:C1", "skill:Python", weight=0.7, edge_type="course-skill")
    g.add_edge("skill:Python", "course:C1", weight=0.7, edge_type="skill-course")
    g.add_node("career:Dev", type="career", n_postings=5)
    g.add_edge("career:Dev", "skill:Python", weight=1.0, edge_type="career-skill")
    g.add_edge("skill:Python", "career:Dev", weight=1.0, edge_type="skill-career")
    pkl = tmp_path / "g.pkl"
    pkl.write_bytes(pickle.dumps(g))

    app = create_app(graph_path=str(pkl))
    app.config["TESTING"] = True
    return app.test_client()


def test_index_returns_prototype_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower() or b"<!doctype" in resp.data.lower()


def test_careers_endpoint_lists_career_titles(client):
    resp = client.get("/api/careers")
    data = resp.get_json()
    assert "Dev" in data["careers"]


def test_recommend_returns_ranked_courses(client):
    payload = {
        "career_goal": "Dev",
        "completed_courses": [],
        "top_k": 3,
    }
    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert "recommendations" in body
    assert body["recommendations"][0]["code"] == "C1"
    assert "score" in body["recommendations"][0]
    assert "explanation" in body["recommendations"][0]


def test_recommend_missing_career_returns_400(client):
    resp = client.post("/api/recommend", json={"completed_courses": []})
    assert resp.status_code == 400
```

- [ ] **Step 3: Implement**

```python
# src/demo/app.py
"""Flask app serving the SkillPath demo.

Loads graph.pkl ONCE at startup. Each /api/recommend request:
  1. Shallow-copies the graph
  2. Inserts a transient student
  3. Runs PPR + explain
  4. Returns JSON

Endpoints:
  GET  /                    serve prototype HTML
  GET  /api/careers         list career titles in the graph (for frontend autocomplete)
  POST /api/recommend       body: {career_goal, completed_courses[], top_k=10}
"""
import copy
import pickle
import secrets
from pathlib import Path

import networkx as nx
from flask import Flask, jsonify, request, send_from_directory

from src.demo.transient_student import add_transient_student
from src.ppr.engine import recommend_courses
from src.ppr.explain import explain_recommendation, format_explanation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "output" / "phase2" / "graph.pkl"
PROTOTYPE_HTML = PROJECT_ROOT / "prototype" / "SkillPath-Demo.html"


def create_app(graph_path: str | None = None) -> Flask:
    app = Flask(__name__)
    g_path = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
    g: nx.DiGraph = pickle.loads(g_path.read_bytes())
    app.config["GRAPH"] = g

    @app.get("/")
    def index():
        return send_from_directory(PROTOTYPE_HTML.parent, PROTOTYPE_HTML.name)

    @app.get("/api/careers")
    def careers():
        g: nx.DiGraph = app.config["GRAPH"]
        titles = sorted(
            n.removeprefix("career:")
            for n, d in g.nodes(data=True)
            if d.get("type") == "career"
        )
        return jsonify({"careers": titles})

    @app.post("/api/recommend")
    def recommend():
        payload = request.get_json(silent=True) or {}
        career_goal = payload.get("career_goal")
        completed = payload.get("completed_courses", []) or []
        top_k = int(payload.get("top_k", 10))
        if not career_goal:
            return jsonify({"error": "career_goal required"}), 400

        g: nx.DiGraph = app.config["GRAPH"]
        career_node = f"career:{career_goal}"
        if not g.has_node(career_node):
            return jsonify({"error": f"unknown career: {career_goal}"}), 404

        g_copy = copy.copy(g)  # shallow — node/edge dicts share, modifications happen via add_*
        sid = add_transient_student(
            g_copy,
            completed_courses=completed,
            student_id=f"DEMO_{secrets.token_hex(4)}",
            career_goal=career_goal,
        )

        recs = recommend_courses(g_copy, sid, career_node, top_k=top_k)
        results = []
        for course_node, score in recs:
            code = course_node.removeprefix("course:")
            name = g_copy.nodes[course_node].get("name", "")
            expl = explain_recommendation(g_copy, sid, career_node, course_node, top_n=5)
            results.append({
                "code": code,
                "name": name,
                "score": round(score, 6),
                "bridge_skills": expl["bridge_skills"],
                "gap_skills": expl["gap_skills_filled"],
                "explanation": format_explanation(expl, course_name=name),
            })

        return jsonify({"student_id": sid, "recommendations": results})

    return app


def main() -> None:
    app = create_app()
    print(f"Loaded graph from {DEFAULT_GRAPH_PATH}")
    print(f"  serving on http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run + commit**

```bash
pytest tests/test_demo_api.py -v
pytest -W error::FutureWarning -q
# Manual sanity: python -m src.demo.app & sleep 2 && curl localhost:5000/api/careers | head -c 200 && kill %1
git add requirements.txt src/demo/app.py tests/test_demo_api.py
git commit -m "feat(demo): Flask backend with /api/careers and /api/recommend"
```

Expected: +4 tests, full suite 97.

---

## Task 3 — Wire the Prototype HTML Frontend

**Files:** MODIFY `prototype/SkillPath-Demo.html`

The prototype is a 2,643-line static mockup with hardcoded data. We surgically wire its input form to `POST /api/recommend` and replace the hardcoded recommendation panel with rendered API results.

- [ ] **Step 1: Locate the existing form + recommendation sections**

```bash
grep -n "form-input\|recommend\|recommendation\|<button" prototype/SkillPath-Demo.html | head -30
```

Identify (a) the input fields for student profile, (b) the recommendation results container, (c) the submit button. Note their CSS selectors / element IDs (may need to ADD IDs if absent).

- [ ] **Step 2: Add IDs to the relevant elements + inject the wiring `<script>`**

At the end of the `<body>` (before `</body>`), insert:

```html
<script>
(async function wireSkillPath() {
  // Populate career-goal datalist on load
  try {
    const r = await fetch('/api/careers');
    const data = await r.json();
    const dl = document.getElementById('career-goal-options');
    if (dl) {
      dl.innerHTML = '';
      for (const c of data.careers.slice(0, 200)) {
        const opt = document.createElement('option');
        opt.value = c;
        dl.appendChild(opt);
      }
    }
  } catch (e) { console.warn('careers fetch failed:', e); }

  // Hook submit button
  const submitBtn = document.getElementById('submit-profile');
  const resultsEl = document.getElementById('recommendation-results');
  if (!submitBtn || !resultsEl) return;

  submitBtn.addEventListener('click', async (ev) => {
    ev.preventDefault();
    const career_goal = document.getElementById('input-career-goal').value.trim();
    const completed_raw = document.getElementById('input-completed-courses').value.trim();
    const completed_courses = completed_raw
      ? completed_raw.split(/[,;\s]+/).filter(Boolean)
      : [];
    resultsEl.innerHTML = '<p>Computing...</p>';
    try {
      const r = await fetch('/api/recommend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({career_goal, completed_courses, top_k: 10}),
      });
      const data = await r.json();
      if (!r.ok) {
        resultsEl.innerHTML = `<p style="color:red">${data.error || 'error'}</p>`;
        return;
      }
      resultsEl.innerHTML = data.recommendations.map(rec => `
        <div class="rec-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px">
          <strong>${rec.code}</strong> — ${rec.name}
          <div style="color:#666; font-size:0.85em">PPR score: ${rec.score.toFixed(5)}</div>
          <p>${rec.explanation || '(no explanation)'}</p>
          ${rec.gap_skills.length ? `<div style="color:#0a7"><b>Gaps you'd fill:</b> ${rec.gap_skills.join(', ')}</div>` : ''}
        </div>
      `).join('');
    } catch (e) {
      resultsEl.innerHTML = `<p style="color:red">${e.message}</p>`;
    }
  });
})();
</script>
```

Also add (or modify existing) form elements to have the expected IDs:
- Career goal `<input>`: `id="input-career-goal"` with `list="career-goal-options"`
- After that input, add: `<datalist id="career-goal-options"></datalist>`
- Completed courses `<input>` or `<textarea>`: `id="input-completed-courses"` (placeholder: "COMP1003, MATH1003, ...")
- Submit button: `id="submit-profile"`
- Results container `<div>`: `id="recommendation-results"`

If the prototype already has form sections, REUSE them (just add IDs). If sections are entirely cosmetic without functional form, add a minimal new section near the top.

- [ ] **Step 3: Manual end-to-end test**

```bash
source .venv/bin/activate
python -m src.demo.app &
sleep 2
# Open http://127.0.0.1:5000/ in a browser
# Enter career goal: "Data Engineer"
# Enter completed: "COMP1003, COMP2003"
# Click submit → should see ranked recommendations with explanations
# Kill server: kill %1
```

Document any UI quirks in the commit message.

- [ ] **Step 4: Commit**

```bash
git add prototype/SkillPath-Demo.html
git commit -m "feat(demo): wire prototype form to /api/recommend"
```

---

## Task 4 — Quickstart Documentation

**File:** MODIFY (or create) `README.md` at project root.

Add a Quickstart section:

```markdown
## Quickstart — Run the Demo

Prereqs: Python 3.14+, `.venv` activated, Phase 1 & 2 outputs in `output/phase2/`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m src.demo.app
# Open http://127.0.0.1:5000/
```

Enter a career goal (autocomplete is populated from the LinkedIn job titles) and a comma-separated list of completed course codes (e.g., `COMP1003, MATH1003`). The page shows top-10 PPR-ranked course recommendations with bridge-skill explanations.

To regenerate Phase 2 outputs from scratch:
```bash
python -m src.normalize.apply_canonical
python -m src.students.synthetic
python -m src.graph.builder
python -m src.students.clustering
python -m src.eval.evaluate
```
```

- [ ] **Commit**: `git add README.md && git commit -m "docs: quickstart for demo"`

---

## Task 5 — Final Smoke + Phase 3 Wrap

- [ ] Run full suite: `pytest -W error::FutureWarning -q` → expect 97 passing
- [ ] Boot the server and manually verify the end-to-end flow with 2-3 different career goals
- [ ] Verify per-request latency (`time curl` against `/api/recommend`) is under 1 second
- [ ] No new commits — this task is verification only

If anything fails: open a new task at that point and resolve before declaring Phase 3 done.

---

## Out of Phase 3 Scope

- 15-min presentation slide deck → Phase 4 (uses `~/.claude/templates/beautiful-html-templates/`)
- Authentication / multi-user
- Dockerization
- Persistence of user-entered profiles
- Real-time graph updates

These are deferred deliberately.
