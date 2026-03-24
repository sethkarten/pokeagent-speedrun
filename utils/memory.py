"""
Backward-compatibility shim — all imports now live in ``utils.stores.memory``.
"""

from utils.stores.memory import (  # noqa: F401
    MemoryEntry,
    KnowledgeEntry,
    Memory,
    KnowledgeBase,
    get_memory_store,
    get_knowledge_base,
)

__all__ = [
    "MemoryEntry",
    "KnowledgeEntry",
    "Memory",
    "KnowledgeBase",
    "get_memory_store",
    "get_knowledge_base",
]
