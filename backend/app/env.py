from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Load simple KEY=VALUE entries from .env without overriding real env vars."""
    env_path = Path(path) if path else _find_dotenv()
    if not env_path or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _parse_env_value(value.strip())


def _find_dotenv() -> Path | None:
    candidates: list[Path] = []
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        candidates.extend(parent / ".env" for parent in (base, *base.parents))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _parse_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            return value.encode("utf-8").decode("unicode_escape")
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value
