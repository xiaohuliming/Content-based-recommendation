from src.parsers.pdf_courses import parse_course_block

SAMPLE_BLOCK = """ACCT2003 PRINCIPLES OF ACCOUNTING I
(3 units)
Pre-requisite(s): None
Course Description: The objective of this course is to provide
students with a general understanding of basic accounting concepts."""

def test_parse_single_course_block():
    course = parse_course_block(SAMPLE_BLOCK)
    assert course["code"] == "ACCT2003"
    assert course["name"] == "PRINCIPLES OF ACCOUNTING I"
    assert course["units"] == 3
    assert course["prerequisites_text"] == "None"
    assert "general understanding of basic accounting" in course["description"]


SAMPLE_BLOCK_MULTILINE = """AI2003 DATA STRUCTURES AND ALGORITHM
ANALYSIS
(3 units)
Pre-requisite(s): AI1013 OBJECT-ORIENTED PROGRAMMING
Course Description: This course aims to develop the students'
knowledge in data structures and the associated algorithms.
This course introduces the concepts and techniques."""

def test_parse_multiline_block():
    course = parse_course_block(SAMPLE_BLOCK_MULTILINE)
    assert course["code"] == "AI2003"
    # Multi-line name should be fully concatenated with single spaces
    assert course["name"] == "DATA STRUCTURES AND ALGORITHM ANALYSIS"
    assert "AI1013 OBJECT-ORIENTED PROGRAMMING" in course["prerequisites_text"]
    assert "data structures and the associated algorithms" in course["description"]
    assert "concepts and techniques" in course["description"]


TWO_COURSES = SAMPLE_BLOCK + "\n\n" + SAMPLE_BLOCK_MULTILINE

def test_extract_course_blocks_splits_correctly():
    from src.parsers.pdf_courses import extract_course_blocks
    blocks = list(extract_course_blocks(TWO_COURSES))
    assert len(blocks) == 2
    assert blocks[0].startswith("ACCT2003")
    assert blocks[1].startswith("AI2003")


def test_parse_pdf_against_real_file(data_dir):
    from src.parsers.pdf_courses import parse_pdf
    pdf_path = data_dir / "Course Descriptions_20260421.pdf"
    courses = parse_pdf(pdf_path)
    # The PDF has ~1300 courses; allow generous bounds for parser quirks
    assert 1000 < len(courses) < 1600

    courses_by_code = {c["code"]: c for c in courses}

    # Spot-check known courses are present AND have substantive descriptions
    for code in ["ACCT2003", "AI3013", "COMP4263"]:
        assert code in courses_by_code, f"{code} missing from parsed output"
        desc = courses_by_code[code]["description"]
        assert len(desc) > 50, f"{code} has truncated description: {desc!r}"

    # Overall data quality: all courses must have non-empty descriptions.
    # The fullwidth-colon fix ([:：] in regexes) recovers PSY4073 & PSY4083,
    # bringing the empty count to exactly 0.
    empty = sum(1 for c in courses if not c["description"])
    assert empty == 0, \
        f"{empty} courses have empty descriptions (expected 0 after fullwidth-colon fix)"

    # Spot-check the two PSY courses that previously had fullwidth colons
    for code in ["PSY4073", "PSY4083"]:
        assert code in courses_by_code, f"{code} missing from parsed output"
        desc = courses_by_code[code]["description"]
        assert len(desc) > 50, f"{code} has short/empty description: {desc!r}"
