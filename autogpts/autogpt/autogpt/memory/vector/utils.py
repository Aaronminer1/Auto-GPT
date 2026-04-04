import logging
from contextlib import suppress
from typing import Any, Sequence, overload

import numpy as np

from autogpt.config import Config

logger = logging.getLogger(__name__)

Embedding = list[float] | list[np.float32] | np.ndarray[Any, np.dtype[np.float32]]
"""Embedding vector"""

TText = Sequence[int]
"""Tokenized text"""

# ---------------------------------------------------------------------------
# Local embedding model — loaded once, reused for every call.
# Uses all-MiniLM-L6-v2 (90MB, 384-dim, no network required after first run).
# ---------------------------------------------------------------------------
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        logger.debug("Loading local embedding model (all-MiniLM-L6-v2)...")
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


@overload
def get_embedding(input: str | TText, config: Config) -> Embedding:
    ...


@overload
def get_embedding(input: list[str] | list[TText], config: Config) -> list[Embedding]:
    ...


def get_embedding(
    input: str | TText | list[str] | list[TText], config: Config
) -> Embedding | list[Embedding]:
    """Get embeddings using the local sentence-transformers model.

    Fully offline — no OpenAI / Ollama API call required.
    """
    multiple = isinstance(input, list) and all(not isinstance(i, int) for i in input)

    if isinstance(input, str):
        input = input.replace("\n", " ")
        with suppress(NotImplementedError):
            return _get_embedding_with_plugin(input, config)
    elif multiple and isinstance(input[0], str):
        input = [text.replace("\n", " ") for text in input]
        with suppress(NotImplementedError):
            return [_get_embedding_with_plugin(i, config) for i in input]

    model = _get_local_model()
    if multiple:
        vecs = model.encode(list(input), convert_to_numpy=True)
        return [v.tolist() for v in vecs]
    else:
        return model.encode(str(input) if not isinstance(input, str) else input,
                            convert_to_numpy=True).tolist()


def _get_embedding_with_plugin(text: str, config: Config) -> Embedding:
    for plugin in config.plugins:
        if plugin.can_handle_text_embedding(text):
            embedding = plugin.handle_text_embedding(text)
            if embedding is not None:
                return embedding

    raise NotImplementedError
