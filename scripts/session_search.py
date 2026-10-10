"""Rank markdown session summaries in a directory.

Does not start a model by itself. Set LLAMA_EMBED_HOST to a running llama-server.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from embedding_rank import EmbedError, embedder_from_env, rank_documents


def rank_sessions(
    question: str,
    directory: Path,
    embed: Callable[[str], Sequence[float]],
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    documents: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.md")):
        documents.append(
            {
                "path": path.name,
                "text": path.read_text(encoding="utf-8", errors="replace")[:4000],
            }
        )
    ranked = rank_documents(question, documents, embed, text_of=lambda doc: doc["text"])
    return [{"path": doc["path"], "score": score} for score, doc in ranked[:limit]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank session summary markdown files.")
    parser.add_argument("question")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        hits = rank_sessions(args.question, args.directory, embedder_from_env())
    except EmbedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"hits": hits}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
