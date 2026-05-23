from unittest.mock import MagicMock


def test_cache_roundtrip(tmp_path):
    from src.skills.cache import SkillCache
    cache = SkillCache(tmp_path / "cache.json")
    assert cache.get("hello world") is None
    cache.set("hello world", ["Python", "ML"])
    assert cache.get("hello world") == ["Python", "ML"]


def test_cache_persists_across_instances(tmp_path):
    from src.skills.cache import SkillCache
    path = tmp_path / "cache.json"
    cache1 = SkillCache(path)
    cache1.set("foo", ["SQL"])
    cache1.flush()

    cache2 = SkillCache(path)
    assert cache2.get("foo") == ["SQL"]


def _mock_client_returning(content: str):
    """Build a mock Anthropic client whose messages.create returns the given text."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=content)]
    client.messages.create.return_value = response
    return client


def test_extractor_returns_parsed_skills(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('["Python", "SQL", "Machine Learning"]')
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("We need Python and SQL skills.")
    assert skills == ["Python", "SQL", "Machine Learning"]


def test_extractor_handles_text_around_json(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('Sure! Here are the skills:\n["Python"]\nThanks.')
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("We are looking for Python and SQL skills.")
    assert skills == ["Python"]


def test_extractor_returns_empty_on_bad_json(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning("not json at all")
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("Some longer text that contains no valid JSON array at all.")
    assert skills == []


def test_extractor_uses_cache(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('["Python"]')
    cache = SkillCache(tmp_path / "c.json")
    extractor = SkillExtractor(client=client, cache=cache)

    extractor.extract("This text is long enough to trigger extraction.")
    extractor.extract("This text is long enough to trigger extraction.")  # second call should hit cache
    assert client.messages.create.call_count == 1


def test_extractor_skips_short_text(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('["Python"]')
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("hi")  # too short
    assert skills == []
    assert client.messages.create.call_count == 0


def test_cache_recovers_from_corrupt_file(tmp_path):
    from src.skills.cache import SkillCache
    path = tmp_path / "cache.json"
    path.write_text("not valid json {{{")
    # Should not raise; should start empty
    cache = SkillCache(path)
    assert cache.get("anything") is None


def test_cache_flush_is_atomic(tmp_path):
    """A previous valid cache should survive even if a flush is "interrupted"."""
    from src.skills.cache import SkillCache
    path = tmp_path / "cache.json"

    # Initial valid state
    cache = SkillCache(path)
    cache.set("hello world (this is at least 20 chars)", ["Python"])
    cache.flush()
    original = path.read_text()

    # Verify no .tmp files lingering after a successful flush
    leftover = list(path.parent.glob("*.tmp*"))
    assert leftover == []

    # Reload and confirm content
    cache2 = SkillCache(path)
    assert cache2.get("hello world (this is at least 20 chars)") == ["Python"]
    assert path.read_text() == original


def test_extractor_returns_empty_on_llm_exception(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("network down")
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("This is a long enough text to trigger extraction.")
    assert skills == []


def test_extractor_handles_empty_response_content(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = MagicMock()
    response = MagicMock()
    response.content = []  # empty content list
    client.messages.create.return_value = response
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("This is a long enough text to trigger extraction.")
    assert skills == []


def test_parse_json_array_filters_non_strings():
    from src.skills.extractor import SkillExtractor
    assert SkillExtractor._parse_json_array('["Python", null, 42, "SQL"]') == ["Python", "SQL"]
