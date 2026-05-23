def test_normalize_strips_and_titlecases():
    from src.skills.taxonomy import normalize_skill
    assert normalize_skill("  python  ") == "Python"
    assert normalize_skill("PYTHON") == "Python"
    assert normalize_skill("machine learning") == "Machine Learning"
    # Acronyms preserved
    assert normalize_skill("sql") == "SQL"
    assert normalize_skill("ML") == "Machine Learning"  # via alias map


def test_normalize_returns_empty_for_junk():
    from src.skills.taxonomy import normalize_skill
    assert normalize_skill("") == ""
    assert normalize_skill("   ") == ""
    assert normalize_skill("a") == ""  # single char


def test_build_taxonomy_dedups_and_counts():
    from src.skills.taxonomy import build_taxonomy

    raw_skills_per_doc = [
        ["python", "SQL", "ML"],
        ["Python", "machine learning"],
        ["PYTHON", "Excel"],
        [],
    ]
    tax = build_taxonomy(raw_skills_per_doc)
    # Three unique canonical skills: Python, SQL, Machine Learning, Excel
    assert set(tax["skill"]) == {"Python", "SQL", "Machine Learning", "Excel"}
    py_count = tax.loc[tax["skill"] == "Python", "doc_frequency"].iloc[0]
    assert py_count == 3


def test_categorize_assigns_buckets():
    from src.skills.taxonomy import categorize_skill
    assert categorize_skill("Python") == "Programming Languages"
    assert categorize_skill("Machine Learning") == "ML & AI"
    assert categorize_skill("SQL") == "Data Systems"
    assert categorize_skill("Statistics") == "Math & Stats"
    assert categorize_skill("Foo Bar Unknown") == "Other"


def test_categorize_csharp_with_special_char():
    """C# (with # non-word char) must categorize as Programming Languages."""
    from src.skills.taxonomy import categorize_skill
    assert categorize_skill("C#") == "Programming Languages"


def test_categorize_word_boundary_prevents_false_positives():
    """Short keywords like 'r', 'go' must not substring-match inside longer words."""
    from src.skills.taxonomy import categorize_skill
    # 'r' from R language must not match inside 'machine learning'
    assert categorize_skill("Machine Learning") == "ML & AI"
    # 'go' from Go language must not match inside 'Golang' (no \b)
    assert categorize_skill("Golang") == "Other"


def test_normalize_scikit_learn_aliases():
    """All casings of scikit-learn / sklearn map to canonical form."""
    from src.skills.taxonomy import normalize_skill
    assert normalize_skill("sklearn") == "scikit-learn"
    assert normalize_skill("SCIKIT-LEARN") == "scikit-learn"
    assert normalize_skill("Scikit-Learn") == "scikit-learn"


def test_build_taxonomy_skill_ids_are_deterministic():
    """Skill IDs must be the same across runs for the same input (alphabetical tiebreak on equal frequency)."""
    from src.skills.taxonomy import build_taxonomy
    docs = [["Java"], ["Python"], ["Ruby"]]  # All freq=1
    df1 = build_taxonomy(docs)
    df2 = build_taxonomy(docs)
    # Same skill -> same skill_id across runs
    for skill in ("Java", "Python", "Ruby"):
        id1 = df1.loc[df1["skill"] == skill, "skill_id"].iloc[0]
        id2 = df2.loc[df2["skill"] == skill, "skill_id"].iloc[0]
        assert id1 == id2, f"{skill} got different skill_ids: {id1} vs {id2}"
    # Skills should be in alphabetical order when all tied
    assert list(df1["skill"]) == ["Java", "Python", "Ruby"]
