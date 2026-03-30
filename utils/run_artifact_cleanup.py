"""
Cleanup helpers for .pokeagent_cache, backups, and run_data.

Supports:
- Deleting files whose modification time is on/after a cutoff (local time).
- Removing top-level directories whose names embed a calendar date (YYYYMMDD_
  or run_YYYYMMDD_) on/after a cutoff.

Used by the local subagent ``subagent_cleanup_run_artifacts``.
"""

from __future__ import annotations

import re
import shutil
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ROOTS = (".pokeagent_cache", "backups", "run_data")

_EMBEDDED_DATE = re.compile(r"^(\d{8})_")
_RUN_EMBEDDED_DATE = re.compile(r"^run_(\d{8})_")


def repo_root_from_utils() -> Path:
    """Repository root (parent of ``utils/``)."""
    return Path(__file__).resolve().parent.parent


def _parse_file_mtime_cutoff(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty file_mtime_on_or_after")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid file_mtime_on_or_after {s!r}; use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS (local)"
    )


def _parse_embedded_date_int(s: str) -> int:
    s = (s or "").strip().replace("-", "")
    if not s:
        raise ValueError("empty directory_embedded_date_on_or_after")
    if len(s) == 8 and s.isdigit():
        y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
        if 1 <= m <= 12 and 1 <= d <= 31:
            return int(s)
    raise ValueError(
        f"Invalid directory_embedded_date_on_or_after {s!r}; use YYYYMMDD or YYYY-MM-DD"
    )


def embedded_date_from_dir_name(name: str) -> Optional[int]:
    m = _EMBEDDED_DATE.match(name)
    if m:
        return int(m.group(1))
    m = _RUN_EMBEDDED_DATE.match(name)
    if m:
        return int(m.group(1))
    return None


def _safe_roots(repo_root: Path, roots: Optional[List[str]]) -> List[Path]:
    names = list(roots) if roots else list(DEFAULT_ROOTS)
    out: List[Path] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            continue
        p = (repo_root / name.strip()).resolve()
        try:
            p.relative_to(repo_root.resolve())
        except ValueError:
            raise ValueError(f"roots entry must stay under repo root: {name!r}")
        out.append(p)
    return out


@dataclass
class CleanupResult:
    success: bool = True
    dry_run: bool = True
    error: Optional[str] = None
    repo_root: str = ""
    file_mtime_on_or_after: Optional[str] = None
    directory_embedded_date_on_or_after: Optional[int] = None
    files_deleted: int = 0
    files_would_delete: int = 0
    dirs_removed: int = 0
    dirs_would_remove: int = 0
    sample_removed_dirs: List[str] = field(default_factory=list)
    sample_deleted_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "success": self.success,
            "dry_run": self.dry_run,
            "repo_root": self.repo_root,
            "file_mtime_on_or_after": self.file_mtime_on_or_after,
            "directory_embedded_date_on_or_after": self.directory_embedded_date_on_or_after,
            "files_deleted": self.files_deleted,
            "files_would_delete": self.files_would_delete,
            "dirs_removed": self.dirs_removed,
            "dirs_would_remove": self.dirs_would_remove,
            "sample_removed_dirs": self.sample_removed_dirs,
            "sample_deleted_files": self.sample_deleted_files,
            "warnings": self.warnings,
        }
        if self.error:
            d["error"] = self.error
        return d


def run_run_artifact_cleanup(
    *,
    repo_root: Optional[Path] = None,
    reasoning: str = "",
    dry_run: bool = True,
    file_mtime_on_or_after: Optional[str] = None,
    directory_embedded_date_on_or_after: Optional[str] = None,
    roots: Optional[List[str]] = None,
    max_path_samples: int = 80,
) -> CleanupResult:
    """
    Apply cleanup rules. At least one of *file_mtime_on_or_after* or
    *directory_embedded_date_on_or_after* must be provided.

    When *dry_run* is True, only counts and samples are returned (no deletes).
    """
    _ = reasoning  # audit / logging hook for callers
    root = (repo_root or repo_root_from_utils()).resolve()
    result = CleanupResult(dry_run=dry_run, repo_root=str(root))

    has_file = bool((file_mtime_on_or_after or "").strip())
    has_dir = bool((directory_embedded_date_on_or_after or "").strip())
    if not has_file and not has_dir:
        result.success = False
        result.error = (
            "Provide at least one of file_mtime_on_or_after or "
            "directory_embedded_date_on_or_after"
        )
        return result

    mtime_dt: Optional[datetime] = None
    mtime_ts: Optional[float] = None
    if has_file:
        try:
            mtime_dt = _parse_file_mtime_cutoff(file_mtime_on_or_after or "")
            mtime_ts = mtime_dt.timestamp()
            result.file_mtime_on_or_after = mtime_dt.isoformat(sep=" ", timespec="seconds")
        except ValueError as e:
            result.success = False
            result.error = str(e)
            return result

    embedded_cutoff: Optional[int] = None
    if has_dir:
        try:
            embedded_cutoff = _parse_embedded_date_int(directory_embedded_date_on_or_after or "")
            result.directory_embedded_date_on_or_after = embedded_cutoff
        except ValueError as e:
            result.success = False
            result.error = str(e)
            return result

    try:
        root_paths = _safe_roots(root, roots)
    except ValueError as e:
        result.success = False
        result.error = str(e)
        return result

    # Phase 1: top-level dated directories
    if embedded_cutoff is not None:
        for base in root_paths:
            if not base.is_dir():
                continue
            try:
                for child in list(base.iterdir()):
                    if not child.is_dir():
                        continue
                    d = embedded_date_from_dir_name(child.name)
                    if d is None or d < embedded_cutoff:
                        continue
                    rel = str(child.relative_to(root))
                    if dry_run:
                        result.dirs_would_remove += 1
                        if len(result.sample_removed_dirs) < max_path_samples:
                            result.sample_removed_dirs.append(rel)
                    else:
                        try:
                            shutil.rmtree(child)
                            result.dirs_removed += 1
                            if len(result.sample_removed_dirs) < max_path_samples:
                                result.sample_removed_dirs.append(rel)
                        except OSError as e:
                            result.warnings.append(f"rmtree {rel}: {e}")
            except OSError as e:
                result.warnings.append(f"list {base}: {e}")

    # Phase 2: files by mtime
    if mtime_ts is not None:
        walk_errors: List[str] = []

        def on_walk_error(exc: OSError) -> None:
            fn = getattr(exc, "filename", None) or "?"
            walk_errors.append(f"{fn}: {exc.strerror or exc}")

        for base in root_paths:
            if not base.is_dir():
                continue
            for dirpath, _dirnames, filenames in os.walk(
                base, topdown=False, onerror=on_walk_error
            ):
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    try:
                        st = fp.stat()
                    except OSError as e:
                        result.warnings.append(f"stat {fp}: {e}")
                        continue
                    if st.st_mtime < mtime_ts:
                        continue
                    rel = str(fp.relative_to(root))
                    if dry_run:
                        result.files_would_delete += 1
                        if len(result.sample_deleted_files) < max_path_samples:
                            result.sample_deleted_files.append(rel)
                    else:
                        try:
                            fp.unlink()
                            result.files_deleted += 1
                            if len(result.sample_deleted_files) < max_path_samples:
                                result.sample_deleted_files.append(rel)
                        except OSError as e:
                            result.warnings.append(f"unlink {rel}: {e}")
            result.warnings.extend(walk_errors)
            walk_errors.clear()

    return result
