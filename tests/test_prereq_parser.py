from src.graph.prereq_parser import parse_prerequisites


def test_simple_single_prereq():
    assert parse_prerequisites("COMP1003") == ["COMP1003"]


def test_multiple_prereqs_with_and():
    assert sorted(parse_prerequisites("COMP1003 AND MATH1003")) == ["COMP1003", "MATH1003"]


def test_or_prereqs_extracted_all():
    """For graph purposes, treat OR as 'any path qualifies' → include all codes as nodes."""
    assert sorted(parse_prerequisites("COMP1003 OR COMP1013")) == ["COMP1003", "COMP1013"]


def test_parens_handled():
    assert sorted(parse_prerequisites("(COMP1003 AND MATH1003) OR COMP1013")) == \
        ["COMP1003", "COMP1013", "MATH1003"]


def test_none_returns_empty():
    assert parse_prerequisites("None") == []
    assert parse_prerequisites("") == []
    assert parse_prerequisites("Year 3 standing") == []
    assert parse_prerequisites("Consent of instructor") == []


def test_lowercase_codes_normalized():
    assert parse_prerequisites("comp1003") == ["COMP1003"]


def test_extra_text_ignored():
    """E.g. 'COMP1003 (or equivalent)' should just extract COMP1003."""
    assert parse_prerequisites("COMP1003 (or equivalent)") == ["COMP1003"]
