from unittest.mock import MagicMock

from src.demo.insight import (
    build_insight_prompt,
    generate_insight,
    parse_insight_response,
)


def _mock_client(content: str):
    client = MagicMock()
    msg = MagicMock(); msg.content = content
    choice = MagicMock(); choice.message = msg
    r = MagicMock(); r.choices = [choice]
    client.chat.completions.create.return_value = r
    return client


def test_parse_insight_extracts_well_formed_json():
    raw = '{"headline":"H","strengths":"S","gaps":"G","strategy":"X"}'
    out = parse_insight_response(raw)
    assert out == {"headline": "H", "strengths": "S", "gaps": "G", "strategy": "X"}


def test_parse_insight_strips_markdown_fences():
    raw = '```json\n{"headline":"H","strengths":"S","gaps":"G","strategy":"X"}\n```'
    out = parse_insight_response(raw)
    assert out["headline"] == "H"


def test_parse_insight_blank_on_empty():
    out = parse_insight_response("")
    assert out == {"headline": "", "strengths": "", "gaps": "", "strategy": ""}


def test_parse_insight_blank_on_bad_json():
    out = parse_insight_response("not json at all")
    assert out["headline"] == ""


def test_parse_insight_handles_missing_keys():
    raw = '{"headline":"H","strengths":"S"}'
    out = parse_insight_response(raw)
    assert out["headline"] == "H"
    assert out["strengths"] == "S"
    assert out["gaps"] == ""
    assert out["strategy"] == ""


def test_build_insight_prompt_uses_english_template():
    p = build_insight_prompt(
        profile={"major": "AI", "year": 3, "gpa": 3.2, "completed_courses": ["AI2003"], "current_courses": ["AI3013"]},
        career="AI Engineer",
        n_postings=5,
        recommendations=[{"code": "AI3133", "name": "NLP", "bridge_skills": [{"skill": "Python"}], "gap_skills": []}],
        student_skills=[{"skill": "Python", "weight": 0.5}],
        career_skills=[{"skill": "Machine Learning", "weight": 0.2}],
        alt_careers=[{"title": "Data Scientist", "similarity": 0.4, "shared_skill_count": 3}],
        language="en",
    )
    assert "AI Engineer" in p
    assert "Year 3" in p
    assert "AI3133" in p
    assert "English" in p
    assert "DO NOT mention PageRank" in p


def test_build_insight_prompt_uses_chinese_template():
    p = build_insight_prompt(
        profile={"major": "人工智能", "year": 4, "gpa": 3.5, "completed_courses": [], "current_courses": []},
        career="AI Engineer",
        n_postings=1,
        recommendations=[],
        student_skills=[],
        career_skills=[],
        alt_careers=[],
        language="zh",
    )
    assert "中文" in p
    assert "大4" in p
    assert "不要" in p


def test_generate_insight_returns_parsed_dict():
    client = _mock_client('{"headline":"Strong AI base","strengths":"Python","gaps":"DL","strategy":"Take AI3133"}')
    out = generate_insight(client, "test-model", "prompt")
    assert out["headline"] == "Strong AI base"
    assert out["strategy"] == "Take AI3133"


def test_generate_insight_returns_empty_on_llm_exception():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network")
    out = generate_insight(client, "test", "prompt")
    assert out == {"headline": "", "strengths": "", "gaps": "", "strategy": ""}
