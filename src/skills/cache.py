"""On-disk cache for skill extraction results, keyed by SHA-1 of input text."""
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillCache:
    """JSON-backed on-disk cache for skill extraction.

    Use `get`/`set` for individual entries. Call `flush()` to write to disk.
    Pass `autoflush=True` to write on every `set` (slower, safer).

    Safety: with `autoflush=False` (the default), callers MUST call `flush()`
    explicitly to persist results. If the process exits without a final flush,
    all changes since the last flush are lost. Long-running batch callers should
    flush periodically (e.g., every 100 calls).
    """

    def __init__(self, path: Path, autoflush: bool = False):
        self.path = Path(path)
        self.autoflush = autoflush
        self._data: dict[str, list[str]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                logger.warning("Cache file %s is corrupt; starting empty.", self.path)
                self._data = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[str] | None:
        return self._data.get(self._key(text))

    def set(self, text: str, skills: list[str]) -> None:
        self._data[self._key(text)] = skills
        if self.autoflush:
            self.flush()

    def flush(self) -> None:
        """Write cache to disk atomically.

        Writes to a temp file in the same directory then renames into place.
        This is crash-safe — a partial write cannot corrupt the existing file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self._data, ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp", prefix=self.path.name + ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
