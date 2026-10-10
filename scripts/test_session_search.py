from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from session_search import rank_sessions  # noqa: E402


def lexical_embed(text: str) -> list[float]:
    vector = [0.0] * 256
    for token in re.findall(r"[0-9a-z]+", text.casefold()):
        bucket = 0
        for char in token:
            bucket = (bucket * 33 + ord(char)) % 256
        vector[bucket] += 1.0
    return vector


def test_image_question_finds_the_summary(tmp_path: Path) -> None:
    (tmp_path / "session-summary.md").write_text(
        "llama.cpp returns the text of the image marker, so images stay on sentence-transformers.\n",
        encoding="utf-8",
    )
    (tmp_path / "other.md").write_text("The warehouse closes at six.\n", encoding="utf-8")
    hits = rank_sessions("where did we leave image support", tmp_path, lexical_embed)
    assert hits[0]["path"] == "session-summary.md"
