"""
Centralized path resolution for porymap / pokeemerald decompilation data.

Data format: Porymap (https://github.com/huderlem/porymap) / pokeemerald layout:
  data/maps/, data/tilesets/, data/layouts/

In-repo only: pokemon_env/porymap/
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_porymap_root() -> Optional[Path]:
    """
    Return the porymap root directory (pokemon_env/porymap).

    Returns:
        Resolved Path to root, or None if not found.
    """
    repo_root = Path(__file__).parent.parent
    porymap_path = repo_root / "pokemon_env" / "porymap"
    if (porymap_path / "data" / "maps").exists():
        return porymap_path.resolve()
    logger.warning(f"Could not find porymap root at {porymap_path}")
    return None
