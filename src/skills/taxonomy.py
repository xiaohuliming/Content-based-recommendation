"""Skill taxonomy: deduplicate, canonicalize, categorize raw extracted skills."""
from collections import Counter
import re
import pandas as pd


# Acronyms that should stay uppercase
ACRONYMS = {
    "SQL", "NLP", "AI", "API", "REST", "ETL", "AWS", "GCP", "TCP", "CSS",
    "HTML", "CRUD", "JSON", "XML", "CI/CD", "GUI", "UI", "UX", "OS", "PHP",
    "CSV", "PDF", "URL", "HTTP", "HTTPS", "VPN", "RPC", "RAG", "LLM", "GAN",
    "CNN", "RNN", "GPU", "CPU", "SVM", "PCA", "KNN",
}

# Map common aliases / variants -> canonical form
ALIASES = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "ai": "Artificial Intelligence",
    "py": "Python",
    "python programming": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "ts": "TypeScript",
    "k8s": "Kubernetes",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "powerbi": "Power BI",
    "power bi": "Power BI",
}


def normalize_skill(raw: str) -> str:
    """Canonicalize a single raw skill string. Returns "" for junk."""
    s = (raw or "").strip()
    if len(s) < 2:
        return ""
    low = s.lower()
    if low in ALIASES:
        return ALIASES[low]
    if s.upper() in ACRONYMS:
        return s.upper()
    # Title-case multi-word skills, preserving acronyms within
    parts = []
    for word in s.split():
        if word.upper() in ACRONYMS:
            parts.append(word.upper())
        else:
            parts.append(word.capitalize())
    return " ".join(parts)


def build_taxonomy(extracted_per_doc: list[list[str]]) -> pd.DataFrame:
    """Aggregate per-document skill lists into a canonical taxonomy table.

    Returns DataFrame with columns: skill_id, skill, doc_frequency.
    `doc_frequency` = number of documents where the skill appears (after canonicalization).
    """
    counter: Counter[str] = Counter()
    for doc_skills in extracted_per_doc:
        canonical = {normalize_skill(s) for s in doc_skills}
        canonical.discard("")
        for s in canonical:
            counter[s] += 1

    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    df = pd.DataFrame(
        [{"skill": s, "doc_frequency": c} for s, c in items]
    )
    df.insert(0, "skill_id", [f"skill_{i:04d}" for i in range(len(df))])
    return df


CATEGORY_KEYWORDS = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "php", "scala", "kotlin", "swift", "r", "matlab", "sas",
    ],
    "ML & AI": [
        "machine learning", "deep learning", "artificial intelligence", "neural",
        "tensorflow", "pytorch", "scikit-learn", "computer vision", "nlp",
        "natural language", "reinforcement learning", "generative", "llm",
        "transformer", "bert", "gpt", "cnn", "rnn", "gan",
    ],
    "Data Systems": [
        "sql", "nosql", "mongodb", "postgresql", "mysql", "spark", "hadoop",
        "kafka", "etl", "data warehouse", "snowflake", "bigquery", "redshift",
        "airflow", "dbt", "data pipeline", "data engineering",
    ],
    "Math & Stats": [
        "statistics", "statistical", "probability", "regression", "linear algebra",
        "calculus", "optimization", "bayesian", "hypothesis testing", "anova",
        "time series", "econometrics",
    ],
    "Tools & Platforms": [
        "git", "docker", "kubernetes", "aws", "gcp", "azure", "jenkins", "ci/cd",
        "tableau", "power bi", "excel", "jupyter", "vscode", "linux",
    ],
    "Domain Knowledge": [
        "marketing", "finance", "accounting", "supply chain", "healthcare",
        "law", "education", "biology", "chemistry", "physics", "economics",
    ],
}


def _keyword_matches(kw: str, text: str) -> bool:
    """Return True if kw matches text. Short keywords require word boundaries."""
    if len(kw) <= 2:
        pattern = rf"\b{re.escape(kw)}"
        if kw[-1].isalnum():
            pattern += r"\b"
        return bool(re.search(pattern, text))
    return kw in text


def categorize_skill(skill: str) -> str:
    """Assign a skill to a category bucket based on keyword matching."""
    low = skill.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(_keyword_matches(kw, low) for kw in keywords):
            return category
    return "Other"


def build_taxonomy_with_categories(extracted_per_doc: list[list[str]]) -> pd.DataFrame:
    """Build taxonomy and attach category column."""
    df = build_taxonomy(extracted_per_doc)
    df.loc[:, "category"] = df["skill"].apply(categorize_skill)
    return df
