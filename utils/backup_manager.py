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
    cache_dir: str = None,
    backup_base_dir: str = "backups"
) -> Optional[str]:
    """
    Create a zip backup of the .pokeagent_cache directory.

    Args:
        objective_id: ID of the completed objective (e.g., "dynamic_01_navigate_route")
        objective_description: Description of the objective (e.g., "Travel to Route 102")
        cache_dir: Directory to backup (default: run-specific cache)
        backup_base_dir: Directory to store backups (default: backups)

    Returns:
        Path to the created backup zip file, or None if backup failed
    """
    try:
        # Use run-specific cache if not provided
        if cache_dir is None:
            from utils.run_data_manager import get_cache_directory
            cache_dir = str(get_cache_directory())
        
        # Ensure cache directory exists
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            logger.warning(f"Cache directory {cache_dir} does not exist, skipping backup")
            return None

        # Get run_id for creating run-specific backup subfolder
        from utils.run_data_manager import get_run_data_manager
        run_manager = get_run_data_manager()
        run_id = run_manager.run_id if run_manager else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create run-specific backup directory: backups/{run_id}/
        backup_base = Path(backup_base_dir)
        run_backup_dir = backup_base / run_id
        run_backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize objective ID and description for filename
        safe_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in objective_id)
        safe_desc = "".join(c if c.isalnum() or c in ('_', '-', ' ') else '_' for c in objective_description)
        safe_desc = safe_desc.replace(' ', '_')[:50]  # Limit length

        # Create backup filename (timestamp + completed objective)
        backup_name = f"{timestamp}_{safe_id}_{safe_desc}"
        backup_path = run_backup_dir / backup_name

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

        # Optional: Clean up old run directories (keep last N runs)
        _cleanup_old_backups(backup_base, max_run_dirs=50)

        return backup_zip

    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        import traceback
        traceback.print_exc()
        return None


def _cleanup_old_backups(backup_base_dir: Path, max_run_dirs: int = 50) -> None:
    """
    Remove old run backup directories to prevent excessive disk usage.

    Args:
        backup_base_dir: Base backups directory containing run subdirectories
        max_run_dirs: Maximum number of run directories to keep (oldest will be deleted)
    """
    try:
        # Get all run subdirectories sorted by modification time
        run_dirs = sorted(
            [d for d in backup_base_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True  # Newest first
        )

        # Remove old run directories beyond max_run_dirs
        if len(run_dirs) > max_run_dirs:
            old_dirs = run_dirs[max_run_dirs:]
            logger.info(f"🧹 Cleaning up {len(old_dirs)} old run backup directories (keeping {max_run_dirs} most recent)")

            for old_dir in old_dirs:
                try:
                    shutil.rmtree(old_dir)
                    logger.debug(f"Deleted old run backup directory: {old_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old run backup directory {old_dir.name}: {e}")

    except Exception as e:
        logger.warning(f"Failed to cleanup old backups: {e}")


def restore_cache_from_backup(
    backup_file: str,
    cache_dir: str = None,
    create_backup_of_current: bool = True
) -> bool:
    """
    Restore .pokeagent_cache from a backup zip file.

    Args:
        backup_file: Path to the backup zip file
        cache_dir: Directory to restore to (default: run-specific cache)
        create_backup_of_current: Whether to backup current cache before restoring

    Returns:
        True if restore succeeded, False otherwise
    """
    try:
        # Use run-specific cache if not provided
        if cache_dir is None:
            from utils.run_data_manager import get_cache_directory
            cache_dir = str(get_cache_directory())
        
        backup_path = Path(backup_file)
        if not backup_path.exists():
            logger.error(f"Backup file {backup_file} does not exist")
            return False

        cache_path = Path(cache_dir)

        # Backup current cache before restoring
        if create_backup_of_current and cache_path.exists() and any(cache_path.iterdir()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_backup = f"backups/{timestamp}_pre_restore_backup"
            logger.info(f"Creating backup of current cache before restore: {temp_backup}.zip")
            shutil.make_archive(str(temp_backup), 'zip', str(cache_path.parent), str(cache_path.name))

        # Create cache directory if it doesn't exist
        cache_path.mkdir(parents=True, exist_ok=True)

        # Extract backup to temporary location
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Extracting backup: {backup_file}")
            shutil.unpack_archive(backup_file, temp_dir, 'zip')
            
            # Find the extracted run directory (should be the only directory)
            extracted_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
            if not extracted_dirs:
                logger.error("No directory found in backup zip")
                return False
            
            extracted_run_dir = extracted_dirs[0]
            logger.info(f"Found extracted directory: {extracted_run_dir.name}")
            
            # Copy all files from extracted directory to target cache directory
            for item in extracted_run_dir.iterdir():
                dest = cache_path / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                    logger.debug(f"Copied: {item.name}")
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                    logger.debug(f"Copied directory: {item.name}")

        logger.info(f"✅ Successfully restored cache to {cache_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to restore from backup: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_backups(backup_dir: str = "backups", limit: int = 20) -> list:
    """
    List available backups (now organized in run-specific subdirectories).

    Args:
        backup_dir: Base directory containing run subdirectories with backups
        limit: Maximum number of backups to return

    Returns:
        List of dicts with backup info (filename, run_id, timestamp, size)
    """
    try:
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return []

        # Find all zip files in run subdirectories
        all_backups = []
        for run_dir in backup_path.iterdir():
            if run_dir.is_dir():
                for backup_file in run_dir.glob("*.zip"):
                    all_backups.append((backup_file, run_dir.name))

        # Sort by modification time (newest first) and limit
        backups = sorted(
            all_backups,
            key=lambda x: x[0].stat().st_mtime,
            reverse=True
        )[:limit]

        result = []
        for backup, run_id in backups:
            stat = backup.stat()
            result.append({
                "filename": backup.name,
                "run_id": run_id,
                "path": str(backup),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        return result

    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        return []
