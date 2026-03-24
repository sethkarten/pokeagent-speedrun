"""DEPRECATED: Use utils.stores.memory instead. This module re-exports for backward compatibility."""

import warnings as _warnings
_warnings.warn(
    "utils.knowledge_base is deprecated; use utils.stores.memory instead",
    DeprecationWarning,
    stacklevel=2,
)

from utils.stores.memory import (  # noqa: F401
    MemoryEntry as KnowledgeEntry,
    Memory as KnowledgeBase,
    get_memory_store as get_knowledge_base,
)
