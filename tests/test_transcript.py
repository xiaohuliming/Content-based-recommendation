from unittest.mock import MagicMock

import pytest

from src.demo.transcript import parse_transcript_with_llm


def _mock_client(content: str):
    client = MagicMock()
    msg = MagicMock(); msg.content = content
    choice = MagicMock(); choice.message = msg
    r = MagicMock(); r.choices = [choice]
    client.chat.completions.create.return_value = r
    return client


def test_parse_returns_empty_on_blank_text():
    out = parse_transcript_with_llm("", client=MagicMock(), model="x")
    assert out == {"major": "", "year": 0, "gpa": None,
                   "completed_courses": [], "current_courses": []}


def test_parse_extracts_well_formed_json():
    client = _mock_client('{"major":"Computer Science","year":3,"gpa":3.6,'
                          '"completed_courses":["COMP1003","COMP2003"],'
                          '"current_courses":["COMP3203"]}')
    out = parse_transcript_with_llm("dummy text", client=client, model="x")
    assert out["major"] == "Computer Science"
    assert out["year"] == 3
    assert out["gpa"] == 3.6
    assert out["completed_courses"] == ["COMP1003", "COMP2003"]
    assert out["current_courses"] == ["COMP3203"]


def test_parse_strips_markdown_fences_and_prose():
    client = _mock_client('Here you go:\n```json\n{"major":"BUS","year":2,"gpa":null,'
                          '"completed_courses":["BUS1003"],"current_courses":[]}\n```\nDone.')
    out = parse_transcript_with_llm("dummy text", client=client, model="x")
    assert out["major"] == "BUS"
    assert out["year"] == 2
    assert out["gpa"] is None
    assert out["completed_courses"] == ["BUS1003"]


def test_parse_returns_empty_on_bad_json():
    client = _mock_client("not json at all, sorry")
    out = parse_transcript_with_llm("dummy text", client=client, model="x")
    assert out["completed_courses"] == []
    assert out["year"] == 0


def test_parse_filters_non_string_course_entries():
    client = _mock_client('{"major":"X","year":1,"gpa":3.0,'
                          '"completed_courses":["COMP1003",null,42,"COMP2003"," ",""],'
                          '"current_courses":[]}')
    out = parse_transcript_with_llm("text", client=client, model="x")
    assert out["completed_courses"] == ["COMP1003", "COMP2003"]


def test_parse_handles_llm_exception():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network")
    out = parse_transcript_with_llm("text", client=client, model="x")
    assert out["completed_courses"] == []


def test_parse_handles_empty_response_choices():
    client = MagicMock()
    r = MagicMock(); r.choices = []
    client.chat.completions.create.return_value = r
    out = parse_transcript_with_llm("text", client=client, model="x")
    assert out["completed_courses"] == []
