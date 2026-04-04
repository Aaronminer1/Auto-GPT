import logging

from autogpt.config import Config

from .memory_item import MemoryItem, MemoryItemRelevance
from .providers.base import VectorMemoryProvider as VectorMemory
from .providers.json_file import JSONFileMemory
from .providers.no_memory import NoMemory

# List of supported memory backends
supported_memory = ["json_file", "no_memory"]

try:
    from .providers.chromadb_memory import ChromaMemory

    supported_memory.append("chromadb")
except ImportError:
    ChromaMemory = None

# try:
#     from .providers.redis import RedisMemory
#     supported_memory.append("redis")
# except ImportError:
#     RedisMemory = None


def get_memory(config: Config) -> VectorMemory:
    """Returns a memory object for the configured backend.

    Defaults to 'chromadb' (local persistent vector store, no API key).
    Falls back to 'json_file' if ChromaDB is not available.
    """
    memory = None

    match config.memory_backend:
        case "chromadb":
            if ChromaMemory is None:
                logging.getLogger(__name__).warning(
                    "chromadb package not installed; falling back to json_file memory."
                )
                memory = JSONFileMemory(config)
            else:
                memory = ChromaMemory(config)

        case "json_file":
            memory = JSONFileMemory(config)

        case "no_memory":
            memory = NoMemory()

        case _:
            logging.getLogger(__name__).warning(
                f"Unknown memory backend '{config.memory_backend}'; using chromadb."
            )
            memory = ChromaMemory(config) if ChromaMemory else JSONFileMemory(config)

    return memory


def get_supported_memory_backends():
    return supported_memory


__all__ = [
    "get_memory",
    "MemoryItem",
    "MemoryItemRelevance",
    "JSONFileMemory",
    "NoMemory",
    "VectorMemory",
    # "RedisMemory",
    # "PineconeMemory",
    # "MilvusMemory",
    # "WeaviateMemory",
]
