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

Environment (OpenAI-compatible chat-completions endpoint):
    LLM_API_KEY   required
    LLM_BASE_URL  optional, default https://api.deepseek.com/v1
    LLM_MODEL     optional, default deepseek-chat

Reruns are resumable: LLM calls are cached on disk in output/cache/.
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
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
CONCURRENCY = 30
LIST_COLUMNS_TO_FLATTEN = ("schedules", "classrooms", "teachers")

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


def _flatten_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Join list-valued cells into ';'-separated strings so CSV roundtrip is lossless."""
    df = df.copy()
    for col in LIST_COLUMNS_TO_FLATTEN:
        if col in df.columns:
            df.loc[:, col] = df[col].apply(
                lambda v: ";".join(str(x) for x in v) if isinstance(v, list) else ""
            )
    return df


def _extract_with_progress(extractor: SkillExtractor, texts: list[str], *, desc: str) -> list[list[str]]:
    """Wrap extractor.extract_many with a tqdm progress bar."""
    bar = tqdm(total=len(texts), desc=desc)
    def on_progress(done: int, _total: int) -> None:
        bar.update(1)
    try:
        return extractor.extract_many(
            texts,
            concurrency=CONCURRENCY,
            flush_every=CACHE_FLUSH_INTERVAL,
            on_progress=on_progress,
        )
    finally:
        bar.close()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise SystemExit("LLM_API_KEY not set. Add it to .env or export it in your shell.")
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step A: courses (PDF + Excel) ----
    print("[1/5] Parsing PDF course descriptions...")
    pdf_courses = parse_pdf(PDF_PATH)
    print(f"  parsed {len(pdf_courses)} courses")

    print("[2/5] Loading Excel timetable...")
    excel_df = collapse_sessions(load_timetable(EXCEL_PATH))
    print(f"  collapsed to {len(excel_df)} unique offerings")

    courses_master = build_courses_master(pdf_courses, excel_df)
    _flatten_list_columns(courses_master).to_csv(
        OUTPUT_DIR / "courses_master.csv", index=False
    )
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
        f"[4/5] Extracting skills via LLM ({model} @ {base_url}): ~{total_calls} calls "
        f"({n_jobs} jobs + {n_courses} courses) at concurrency={CONCURRENCY}.\n"
        f"      Cost / time depend on the model. Reruns are resumable (output/cache/).\n"
        f"      Progress flushes every {CACHE_FLUSH_INTERVAL} completed items — Ctrl-C is safe between flushes.",
        flush=True,
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = SkillCache(OUTPUT_DIR / "cache" / "skill_extraction.json")
    extractor = SkillExtractor(client=client, cache=cache, model=model)

    job_sample = jobs.sample(n=n_jobs, random_state=RANDOM_SEED)
    job_skills = _extract_with_progress(
        extractor, job_sample["description"].tolist(), desc="  jobs"
    )
    job_sample.drop(columns=["description"]).assign(
        extracted_skills=[",".join(s) for s in job_skills]
    ).to_csv(OUTPUT_DIR / "jobs_sample_skills.csv", index=False)
    print(f"  wrote jobs_sample_skills.csv ({len(job_sample)} rows)")

    course_skills = _extract_with_progress(
        extractor, courses_master["description"].fillna("").tolist(), desc="  courses"
    )
    courses_master["extracted_skills"] = [",".join(s) for s in course_skills]
    _flatten_list_columns(courses_master).to_csv(
        OUTPUT_DIR / "courses_skills.csv", index=False
    )
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
