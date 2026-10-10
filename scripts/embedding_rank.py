"""Rank text with EmbeddingGemma 2 task prefixes.

The embedder is injected. Nothing here loads a model or invents a vector.
``llama_embed`` posts to llama-server ``/v1/embeddings`` and refuses a
non-finite or wrong-sized vector. The request omits ``dimensions`` so
Matryoshka truncation stays with the caller.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import TypeVar

PREFIXES: dict[str, str] = {
    "SearchQuery": "task: search result | query: ",
    "Document": "title: none | text: ",
    "CodeRetrieval": "task: code retrieval | query: ",
}

NATIVE_DIM = 768
T = TypeVar("T")
Embedder = Callable[[str], Sequence[float]]


class EmbedError(RuntimeError):
    """The embedding server is unset, unreachable, or returned a bad vector."""


def apply_prompt(text: str, prompt_name: str) -> str:
    try:
        prefix = PREFIXES[prompt_name]
    except KeyError:
        known = ", ".join(sorted(PREFIXES))
        raise KeyError(f"prompt {prompt_name!r} is not registered; known prompts: {known}") from None
    return prefix + text


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) == 0:
        raise EmbedError(f"vector length mismatch {len(left)} vs {len(right)}")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        if not math.isfinite(a) or not math.isfinite(b):
            raise EmbedError("non-finite embedding")
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def is_ignored_image_vector(
    image_vector: Sequence[float],
    marker_vector: Sequence[float],
    *,
    threshold: float = 0.999,
) -> bool:
    """True when an image call returned the text embedding of ``<|image|>``."""
    score = cosine_similarity(image_vector, marker_vector)
    return math.isfinite(score) and score >= threshold


def rank_documents(
    query: str,
    documents: Sequence[T],
    embed: Embedder,
    text_of: Callable[[T], str] | None = None,
    *,
    query_prompt: str = "SearchQuery",
    document_prompt: str = "Document",
) -> list[tuple[float, T]]:
    """Return (score, document) pairs, highest cosine first.

    Empty document text is skipped. Ties keep the original order.
    """
    if query.strip() == "":
        raise ValueError("query is empty")
    read = text_of if text_of is not None else _text_of_str
    query_vector = embed(apply_prompt(query, query_prompt))
    scored: list[tuple[float, T, int]] = []
    for index, document in enumerate(documents):
        text = read(document)
        if text.strip() == "":
            continue
        vector = embed(apply_prompt(text, document_prompt))
        scored.append((cosine_similarity(query_vector, vector), document, index))
    scored.sort(key=lambda row: (-row[0], row[2]))
    return [(score, document) for score, document, _index in scored]


def llama_embed(
    text: str,
    *,
    base_url: str,
    alias: str = "embeddinggemma2",
    timeout: float = 30.0,
    urlopen: Callable[..., object] | None = None,
) -> list[float]:
    """Embed one already-prefixed string. Does not start a server."""
    opener = urllib.request.urlopen if urlopen is None else urlopen
    url = base_url.rstrip("/") + "/v1/embeddings"
    body = {"model": alias, "input": [text], "encoding_format": "float"}
    if "dimensions" in body:
        raise EmbedError("refusing to send a dimensions field")
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        raise EmbedError(f"embedding server unreachable: {exc}") from exc
    except TimeoutError as exc:
        raise EmbedError(f"embedding server unreachable: {exc}") from exc
    return vector_from_payload(payload)


def vector_from_payload(payload: object) -> list[float]:
    raw = _extract_embedding(payload)
    if len(raw) != NATIVE_DIM:
        raise EmbedError(f"expected native dim {NATIVE_DIM}, got {len(raw)}")
    vector = [float(value) for value in raw]
    if any(not math.isfinite(value) for value in vector):
        raise EmbedError("non-finite embedding")
    return vector


def embedder_from_env(env: os._Environ[str] | None = None) -> Embedder:
    source = os.environ if env is None else env
    base = source.get("LLAMA_EMBED_HOST") or source.get("EMBED_URL") or ""
    if base.strip() == "":
        raise EmbedError("LLAMA_EMBED_HOST is unset")
    alias = source.get("LLAMA_EMBED_ALIAS") or "embeddinggemma2"

    def embed(text: str) -> list[float]:
        return llama_embed(text, base_url=base, alias=alias)

    return embed


def _text_of_str(document: T) -> str:
    if not isinstance(document, str):
        raise TypeError("text_of is required when documents are not strings")
    return document


def _extract_embedding(payload: object) -> list[object]:
    if isinstance(payload, dict) and isinstance(payload.get("embedding"), list):
        return payload["embedding"]
    rows: object
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("data")
    else:
        rows = None
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        embedding = rows[0].get("embedding")
        if isinstance(embedding, list):
            return embedding
    raise EmbedError("embeddings response did not contain one vector")
