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
    "COMP", "STAT", "DS",     # CS / data
    "BUS", "ECON", "ACCT", "FIN",  # business
    "ENG", "TRA", "MCOM",     # english / translation / media
    "AI",                      # AI
    "ENV", "BIOL", "CHEM",    # science
    "PSY", "POLS", "SWSA",    # social sciences
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
    # Filter to courses whose code's first year-digit is <= student's year
    def year_of(code: str) -> int | None:
        for ch in code:
            if ch.isdigit():
                return int(ch)
        return None

    eligible = courses[
        courses["code"].notna() &
        courses["code"].apply(lambda c: (year_of(c) or 99) <= year if isinstance(c, str) else False)
    ]
    in_major = eligible[eligible["code"].str.startswith(major_prefix)]["code"].tolist()
    out_of_major = eligible[~eligible["code"].str.startswith(major_prefix)]["code"].tolist()

    n_total = max(0, int(rng.gauss(COURSES_PER_YEAR_MEAN * year, COURSES_PER_YEAR_STD)))
    n_major = min(len(in_major), int(n_total * 0.7))
    n_gen_ed = min(len(out_of_major), n_total - n_major)
    # Backfill any gen-ed shortfall with extra in-major picks so we reach n_total
    gen_ed_shortfall = (n_total - n_major) - n_gen_ed
    n_major_backfill = min(len(in_major) - n_major, gen_ed_shortfall)
    major_pool = rng.sample(in_major, n_major + n_major_backfill)
    picks = major_pool + rng.sample(out_of_major, n_gen_ed)
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
