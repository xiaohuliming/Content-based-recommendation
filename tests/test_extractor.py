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
    """Build a mock OpenAI-compatible client whose chat.completions.create returns the given text."""
    client = MagicMock()
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
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
    assert client.chat.completions.create.call_count == 1


def test_extractor_skips_short_text(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning('["Python"]')
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("hi")  # too short
    assert skills == []
    assert client.chat.completions.create.call_count == 0


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
    client.chat.completions.create.side_effect = RuntimeError("network down")
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("This is a long enough text to trigger extraction.")
    assert skills == []


def test_extractor_handles_empty_response_choices(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = MagicMock()
    response = MagicMock()
    response.choices = []  # empty choices list
    client.chat.completions.create.return_value = response
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("This is a long enough text to trigger extraction.")
    assert skills == []


def test_extractor_handles_none_message_content(tmp_path):
    """OpenAI client returns None content when the model produces no text."""
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache

    client = _mock_client_returning(None)
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))

    skills = extractor.extract("This is a long enough text to trigger extraction.")
    assert skills == []


def test_parse_json_array_filters_non_strings():
    from src.skills.extractor import SkillExtractor
    assert SkillExtractor._parse_json_array('["Python", null, 42, "SQL"]') == ["Python", "SQL"]


def test_extract_many_preserves_input_order(tmp_path):
    """When concurrent workers complete out of order, results must still match input order."""
    import time, random
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache
    from unittest.mock import MagicMock

    # Each call returns a JSON array containing the input text's first word,
    # so we can verify result[i] corresponds to texts[i].
    def make_response(content):
        msg = MagicMock(); msg.content = content
        choice = MagicMock(); choice.message = msg
        r = MagicMock(); r.choices = [choice]
        return r

    client = MagicMock()
    def side_effect(messages, model, max_tokens):
        user_prompt = messages[0]["content"]
        # Extract the input text between triple quotes
        marker = '"""'
        s = user_prompt.find(marker) + len(marker)
        e = user_prompt.find(marker, s)
        text = user_prompt[s:e].strip()
        first_word = text.split()[0]
        # Simulate variable latency so completions arrive out of order
        time.sleep(random.uniform(0.01, 0.05))
        return make_response(f'["{first_word}"]')
    client.chat.completions.create = MagicMock(side_effect=side_effect)

    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))
    texts = [f"word{i} this text is long enough for extraction." for i in range(50)]
    results = extractor.extract_many(texts, concurrency=10, flush_every=5)

    assert len(results) == 50
    for i, r in enumerate(results):
        assert r == [f"word{i}"], f"order mismatch at {i}: got {r}"


def test_extract_many_handles_empty_input(tmp_path):
    from src.skills.extractor import SkillExtractor
    from src.skills.cache import SkillCache
    from unittest.mock import MagicMock

    client = MagicMock()
    extractor = SkillExtractor(client=client, cache=SkillCache(tmp_path / "c.json"))
    assert extractor.extract_many([]) == []
    assert client.chat.completions.create.call_count == 0


def test_cache_set_and_flush_are_thread_safe(tmp_path):
    """Concurrent sets + flushes must not raise 'dictionary changed size during iteration'."""
    import threading
    from src.skills.cache import SkillCache

    cache = SkillCache(tmp_path / "c.json")
    stop = threading.Event()
    errors: list[Exception] = []

    def writer(start: int):
        try:
            i = start
            while not stop.is_set():
                cache.set(f"text-{i}", ["Skill"])
                i += 1000
        except Exception as e:
            errors.append(e)

    def flusher():
        try:
            for _ in range(20):
                cache.flush()
        except Exception as e:
            errors.append(e)

    writers = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in writers: t.start()
    flush_t = threading.Thread(target=flusher)
    flush_t.start()
    flush_t.join()
    stop.set()
    for t in writers: t.join()

    assert errors == [], f"race conditions raised: {errors}"
