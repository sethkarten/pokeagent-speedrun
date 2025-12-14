"""
Backup manager for .pokeagent_cache directory.

Creates timestamped zip backups whenever the agent completes objectives.
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_cache_backup(
    objective_id: str,
    objective_description: str,
    cache_dir: str = ".pokeagent_cache",
    backup_base_dir: str = "backups"
) -> Optional[str]:
    """
    Create a zip backup of the .pokeagent_cache directory.

    Args:
        objective_id: ID of the completed objective (e.g., "dynamic_01_navigate_route")
        objective_description: Description of the objective (e.g., "Travel to Route 102")
        cache_dir: Directory to backup (default: .pokeagent_cache)
        backup_base_dir: Directory to store backups (default: backups)

    Returns:
        Path to the created backup zip file, or None if backup failed
    """
    try:
        # Ensure cache directory exists
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            logger.warning(f"Cache directory {cache_dir} does not exist, skipping backup")
            return None

        # Create backups directory if it doesn't exist
        backup_dir = Path(backup_base_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize objective ID and description for filename
        safe_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in objective_id)
        safe_desc = "".join(c if c.isalnum() or c in ('_', '-', ' ') else '_' for c in objective_description)
        safe_desc = safe_desc.replace(' ', '_')[:50]  # Limit length

        # Create backup filename
        backup_name = f"{timestamp}_{safe_id}_{safe_desc}"
        backup_path = backup_dir / backup_name

        # Create the zip archive
        logger.info(f"📦 Creating backup: {backup_name}.zip")
        shutil.make_archive(
            str(backup_path),  # base_name (without .zip extension)
            'zip',             # format
            str(cache_path.parent),  # root_dir
            str(cache_path.name)     # base_dir
        )

        backup_zip = f"{backup_path}.zip"
        backup_size_mb = os.path.getsize(backup_zip) / (1024 * 1024)
        logger.info(f"✅ Backup created: {backup_zip} ({backup_size_mb:.2f} MB)")

        # Optional: Clean up old backups (keep last N backups)
        _cleanup_old_backups(backup_dir, max_backups=50)

        return backup_zip

    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        import traceback
        traceback.print_exc()
        return None


def _cleanup_old_backups(backup_dir: Path, max_backups: int = 50) -> None:
    """
    Remove old backups to prevent excessive disk usage.

    Args:
        backup_dir: Directory containing backups
        max_backups: Maximum number of backups to keep (oldest will be deleted)
    """
    try:
        # Get all backup zip files sorted by modification time
        backups = sorted(
            [f for f in backup_dir.glob("*.zip")],
            key=lambda x: x.stat().st_mtime,
            reverse=True  # Newest first
        )

        # Remove old backups beyond max_backups
        if len(backups) > max_backups:
            old_backups = backups[max_backups:]
            logger.info(f"🧹 Cleaning up {len(old_backups)} old backups (keeping {max_backups} most recent)")

            for old_backup in old_backups:
                try:
                    old_backup.unlink()
                    logger.debug(f"Deleted old backup: {old_backup.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old backup {old_backup.name}: {e}")

    except Exception as e:
        logger.warning(f"Failed to cleanup old backups: {e}")


def restore_cache_from_backup(
    backup_file: str,
    cache_dir: str = ".pokeagent_cache",
    create_backup_of_current: bool = True
) -> bool:
    """
    Restore .pokeagent_cache from a backup zip file.

    Args:
        backup_file: Path to the backup zip file
        cache_dir: Directory to restore to (default: .pokeagent_cache)
        create_backup_of_current: Whether to backup current cache before restoring

    Returns:
        True if restore succeeded, False otherwise
    """
    try:
        backup_path = Path(backup_file)
        if not backup_path.exists():
            logger.error(f"Backup file {backup_file} does not exist")
            return False

        cache_path = Path(cache_dir)

        # Backup current cache before restoring
        if create_backup_of_current and cache_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_backup = f"backups/{timestamp}_pre_restore_backup"
            logger.info(f"Creating backup of current cache before restore: {temp_backup}.zip")
            shutil.make_archive(str(temp_backup), 'zip', str(cache_path.parent), str(cache_path.name))

        # Remove current cache
        if cache_path.exists():
            logger.info(f"Removing current cache: {cache_path}")
            shutil.rmtree(cache_path)

        # Extract backup
        logger.info(f"Restoring from backup: {backup_file}")
        shutil.unpack_archive(backup_file, str(cache_path.parent), 'zip')

        logger.info(f"✅ Successfully restored cache from {backup_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to restore from backup: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_backups(backup_dir: str = "backups", limit: int = 20) -> list:
    """
    List available backups.

    Args:
        backup_dir: Directory containing backups
        limit: Maximum number of backups to return

    Returns:
        List of dicts with backup info (filename, timestamp, size)
    """
    try:
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return []

        backups = sorted(
            [f for f in backup_path.glob("*.zip")],
            key=lambda x: x.stat().st_mtime,
            reverse=True  # Newest first
        )[:limit]

        result = []
        for backup in backups:
            stat = backup.stat()
            result.append({
                "filename": backup.name,
                "path": str(backup),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        return result

    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        return []
