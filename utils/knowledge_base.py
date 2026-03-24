"""DEPRECATED: Use utils.memory instead. This module re-exports for backward compatibility."""

import warnings as _warnings
_warnings.warn(
    "utils.knowledge_base is deprecated; use utils.memory instead",
    DeprecationWarning,
    stacklevel=2,
)

from utils.memory import (  # noqa: F401
    MemoryEntry as KnowledgeEntry,
    Memory as KnowledgeBase,
    get_memory_store as get_knowledge_base,
)
