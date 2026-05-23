"""LLM-based skill extraction with caching.

Provider: any OpenAI-compatible chat-completions endpoint.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from src.skills.cache import SkillCache
from src.skills.prompts import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class SkillExtractor:
    """Extract skills from text using an LLM, with on-disk caching.

    The `client` parameter accepts any object with a
    `chat.completions.create(...)` method returning an object whose
    `.choices[0].message.content` is the model's text output (OpenAI-style).
    Passing a mock client is the supported test strategy.
    """

    MIN_LENGTH = 20

    def __init__(
        self,
        client,
        cache: SkillCache,
        model: str = "deepseek-v4-pro",
        max_tokens: int = 1024,
    ):
        self.client = client
        self.cache = cache
        self.model = model
        self.max_tokens = max_tokens

    def extract(self, text: str) -> list[str]:
        text = (text or "").strip()
        if len(text) < self.MIN_LENGTH:
            return []

        cached = self.cache.get(text)
        if cached is not None:
            return cached

        prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logger.warning("LLM call failed, returning []: %s", exc)
            skills = []
            self.cache.set(text, skills)
            return skills

        raw = ""
        if response.choices and response.choices[0].message:
            raw = response.choices[0].message.content or ""
        skills = self._parse_json_array(raw)
        self.cache.set(text, skills)
        return skills

    def extract_many(
        self,
        texts: list[str],
        *,
        concurrency: int = 30,
        flush_every: int = 100,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[list[str]]:
        """Extract skills for many texts concurrently. Results preserve input order.

        Periodically flushes the cache so a Ctrl-C loses at most `flush_every`
        in-memory results. Workers reuse the underlying client's connection pool.
        """
        results: list[list[str]] = [[] for _ in texts]
        if not texts:
            return results

        total = len(texts)
        completed = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {pool.submit(self.extract, t): i for i, t in enumerate(texts)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total)
                if completed % flush_every == 0:
                    self.cache.flush()
        self.cache.flush()
        return results

    @staticmethod
    def _parse_json_array(raw: str) -> list[str]:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            value = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return []
        if not isinstance(value, list):
            return []
        return [s.strip() for s in value if isinstance(s, str) and s.strip()]
