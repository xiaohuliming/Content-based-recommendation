"""Phase 1 orchestrator: produce all data-foundation outputs.

Run from project root:
    source .venv/bin/activate
    python -m src.pipeline.build_phase1

Outputs (in output/):
    - courses_master.csv         all UIC courses with descriptions + timetable
    - jobs_clean.csv             all 124K LinkedIn jobs, essential cols only
    - jobs_sample_skills.csv     5K random sample with extracted skills
    - courses_skills.csv         courses with extracted skills column
    - skill_taxonomy.csv         canonical skill list with categories

Environment:
    ANTHROPIC_API_KEY must be set (e.g., in .env or shell).

Reruns are resumable: LLM calls are cached on disk in output/cache/.
"""
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from src.parsers.excel_timetable import collapse_sessions, load_timetable
from src.parsers.linkedin_jobs import ensure_extracted, load_postings
from src.parsers.pdf_courses import parse_pdf
from src.pipeline.build_courses import build_courses_master
from src.skills.cache import SkillCache
from src.skills.extractor import SkillExtractor
from src.skills.taxonomy import build_taxonomy_with_categories

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

PDF_PATH = DATA_DIR / "Course Descriptions_20260421.pdf"
EXCEL_PATH = DATA_DIR / "Course List and Timetable_Semester 2 of AY2025-26_20260224.xls"
LINKEDIN_ARCHIVE = DATA_DIR / "archive.zip"

JOB_SAMPLE_SIZE = 5000
RANDOM_SEED = 42
CACHE_FLUSH_INTERVAL = 100


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. Add it to .env or export it in your shell.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step A: courses (PDF + Excel) ----
    print("[1/5] Parsing PDF course descriptions...")
    pdf_courses = parse_pdf(PDF_PATH)
    print(f"  parsed {len(pdf_courses)} courses")

    print("[2/5] Loading Excel timetable...")
    excel_df = collapse_sessions(load_timetable(EXCEL_PATH))
    print(f"  collapsed to {len(excel_df)} unique offerings")

    courses_master = build_courses_master(pdf_courses, excel_df)
    courses_master.to_csv(OUTPUT_DIR / "courses_master.csv", index=False)
    print(f"  wrote courses_master.csv ({len(courses_master)} rows)")

    # ---- Step B: LinkedIn jobs ----
    print("[3/5] Loading LinkedIn postings...")
    postings_csv = ensure_extracted(LINKEDIN_ARCHIVE, OUTPUT_DIR / "linkedin_unpacked")
    jobs = load_postings(postings_csv)
    jobs_clean = jobs.drop(columns=["description"]).copy()
    jobs_clean.to_csv(OUTPUT_DIR / "jobs_clean.csv", index=False)
    print(f"  wrote jobs_clean.csv ({len(jobs_clean)} rows)")

    # ---- Step C: skill extraction (sample of jobs + all courses) ----
    n_jobs = min(JOB_SAMPLE_SIZE, len(jobs))
    n_courses = len(courses_master)
    total_calls = n_jobs + n_courses
    print(
        f"[4/5] Extracting skills via LLM: ~{total_calls} calls "
        f"({n_jobs} jobs + {n_courses} courses).\n"
        f"      First run takes ~30 min and costs ~$9 with Claude Haiku 4.5.\n"
        f"      Reruns are resumable: results cached in output/cache/.\n"
        f"      Progress flushes every {CACHE_FLUSH_INTERVAL} items — Ctrl-C is safe between batches.",
        flush=True,
    )
    client = Anthropic()
    cache = SkillCache(OUTPUT_DIR / "cache" / "skill_extraction.json")
    extractor = SkillExtractor(client=client, cache=cache)

    job_sample = jobs.sample(n=n_jobs, random_state=RANDOM_SEED)
    job_skills = []
    for i, desc in enumerate(tqdm(job_sample["description"], desc="  jobs"), 1):
        job_skills.append(extractor.extract(desc))
        if i % CACHE_FLUSH_INTERVAL == 0:
            cache.flush()
    cache.flush()
    job_sample.drop(columns=["description"]).assign(
        extracted_skills=[",".join(s) for s in job_skills]
    ).to_csv(OUTPUT_DIR / "jobs_sample_skills.csv", index=False)
    print(f"  wrote jobs_sample_skills.csv ({len(job_sample)} rows)")

    course_skills = []
    for i, desc in enumerate(tqdm(courses_master["description"].fillna(""), desc="  courses"), 1):
        course_skills.append(extractor.extract(desc))
        if i % CACHE_FLUSH_INTERVAL == 0:
            cache.flush()
    cache.flush()
    courses_master["extracted_skills"] = [",".join(s) for s in course_skills]
    courses_master.to_csv(OUTPUT_DIR / "courses_skills.csv", index=False)
    print(f"  wrote courses_skills.csv ({len(courses_master)} rows)")

    # ---- Step D: taxonomy ----
    print("[5/5] Building skill taxonomy...")
    all_per_doc = job_skills + course_skills
    taxonomy = build_taxonomy_with_categories(all_per_doc)
    taxonomy.to_csv(OUTPUT_DIR / "skill_taxonomy.csv", index=False)
    print(f"  wrote skill_taxonomy.csv ({len(taxonomy)} unique skills)")

    # ---- Summary ----
    print("\nPhase 1 complete. Outputs in output/:")
    for name in ["courses_master.csv", "jobs_clean.csv", "jobs_sample_skills.csv",
                 "courses_skills.csv", "skill_taxonomy.csv"]:
        path = OUTPUT_DIR / name
        size_kb = path.stat().st_size / 1024
        print(f"  {name:30s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
