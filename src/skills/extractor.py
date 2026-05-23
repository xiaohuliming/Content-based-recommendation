"""LLM-based skill extraction with caching."""
import json
import logging
from typing import Optional

from src.skills.cache import SkillCache
from src.skills.prompts import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class SkillExtractor:
    """Extract skills from text using an LLM, with on-disk caching.

    The `client` parameter accepts any object with a `messages.create(...)`
    method returning an object with `.content[0].text` (Anthropic-style).
    Passing a mock client is the supported test strategy.
    """

    MIN_LENGTH = 20

    def __init__(
        self,
        client,
        cache: SkillCache,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 512,
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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logger.warning("LLM call failed, returning []: %s", exc)
            skills = []
            self.cache.set(text, skills)
            return skills

        raw = (response.content[0].text if response.content else "") or ""
        skills = self._parse_json_array(raw)
        self.cache.set(text, skills)
        return skills

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
