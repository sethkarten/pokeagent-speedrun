#!/usr/bin/env python3
"""
Singleton manager for MapStitcher to ensure all components use the same instance.
This prevents multiple MapStitcher instances from being created and ensures
consistent map data across the application.
"""

import logging
from utils.map_stitcher import MapStitcher

logger = logging.getLogger(__name__)

# The single global MapStitcher instance
_map_stitcher_instance = None

def get_instance():
    """Get the singleton MapStitcher instance."""
    global _map_stitcher_instance
    if _map_stitcher_instance is None:
        _map_stitcher_instance = MapStitcher()
        logger.info(f"Created singleton MapStitcher with {len(_map_stitcher_instance.map_areas)} areas")
    elif len(_map_stitcher_instance.map_areas) == 0:
        # If we have no data, reload from cache file in case it was updated
        _map_stitcher_instance.load_from_file()
        if len(_map_stitcher_instance.map_areas) > 0:
            logger.info(f"Reloaded MapStitcher from cache, now has {len(_map_stitcher_instance.map_areas)} areas")
    return _map_stitcher_instance

def reset_instance():
    """Reset the singleton instance (mainly for testing)."""
    global _map_stitcher_instance
    _map_stitcher_instance = None
    logger.info("Reset MapStitcher singleton instance")