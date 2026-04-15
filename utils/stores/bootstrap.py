"""
Bootstrap import — load learned artifacts from a previous run into a new cache.

Reads memory.json, skills.json, subagents.json (and optionally an evolved
orchestrator policy) from a source directory, re-paths all entries under
``bootstrapped/``, sets ``source="bootstrapped"``, and writes them into the
target cache directory so the stores pick them up on init.
"""

import glob
import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

STORE_FILES = ["memory.json", "skills.json", "subagents.json"]

PROMPT_CANONICAL = "EVOLVED_ORCHESTRATOR_POLICY.md"
SANITIZED_PROMPT_FILENAME = "BOOTSTRAP_SANITIZED_ORCHESTRATOR_POLICY.md"
_STEPS_PATTERN = "steps_*.md"


def _resolve_prompt(source_dir: str) -> Optional[str]:
    """Find the evolved orchestrator policy in *source_dir*.

    Fallback chain:
      1. EVOLVED_ORCHESTRATOR_POLICY.md (canonical name from auto-export)
      2. Latest steps_*.md (backwards-compat with manually assembled dirs)
      3. None (no prompt override)
    """
    canonical = os.path.join(source_dir, PROMPT_CANONICAL)
    if os.path.isfile(canonical):
        return canonical

    step_files = sorted(glob.glob(os.path.join(source_dir, _STEPS_PATTERN)))
    if step_files:
        return step_files[-1]

    return None


def _normalize_prompt_text(prompt_text: str) -> str:
    """Normalize prompt text so downstream prompt assembly is clean."""
    if not prompt_text:
        return ""

    normalized = prompt_text.strip()
    if normalized.startswith("IMPROVED BASE PROMPT:"):
        normalized = normalized[len("IMPROVED BASE PROMPT:"):].lstrip()
    return normalized


def _preprocess_bootstrap_prompt(
    prompt_text: str,
    backend: Optional[str],
    model_name: Optional[str],
) -> str:
    """Use one VLM call to conservatively strip run-specific strategy/objective text."""
    normalized_prompt = _normalize_prompt_text(prompt_text)
    if not normalized_prompt or not backend or not model_name:
        return normalized_prompt

    prompt = f"""You are helping bootstrap a new gameplay run from a previously evolved orchestrator policy.

Your job is to conservatively preprocess the policy below so it remains useful in a new run while removing run-specific content tied to the prior session's *current objective or strategy*.

Keep:
- General strategic guidance
- Reusable lessons about navigation, battles, memory usage, tool usage, and exploration
- High-level objective-setting advice
- Stable game knowledge that is still broadly useful

Remove or rewrite conservatively:
- Statements about the previous run's current location, battle, route, or immediate tactical situation
- "Current Strategy", "Immediate Context", and "Immediate Next Steps" content that assumes a specific in-progress state
- References to what the agent is doing *right now* in the previous run

Rules:
- Be conservative: preserve reusable guidance whenever possible
- Do not invent new strategy
- Do not mention bootstrapping or this preprocessing step
- Return only the revised markdown prompt, with no preamble or explanation

## Policy To Preprocess
{normalized_prompt}
"""

    try:
        from utils.agent_infrastructure.vlm_backends import VLM

        text_vlm = VLM(
            backend=backend,
            model_name=model_name,
            tools=None,
            system_instruction=None,
        )
        result = text_vlm.get_text_query(prompt, "BootstrapPromptSanitizer")
        if not isinstance(result, str):
            logger.warning(
                "Bootstrap prompt sanitizer returned non-string response; keeping original prompt"
            )
            return normalized_prompt

        sanitized = _normalize_prompt_text(result)
        if not sanitized or sanitized.startswith("I encountered an error processing the request."):
            logger.warning("Bootstrap prompt sanitizer produced fallback/empty output; keeping original prompt")
            return normalized_prompt

        logger.info(
            "Bootstrap: sanitized prompt from %d chars to %d chars",
            len(normalized_prompt),
            len(sanitized),
        )
        return sanitized
    except Exception as exc:
        logger.warning("Bootstrap: prompt sanitization failed; keeping original prompt: %s", exc)
        return normalized_prompt


def _materialize_prompt_override(
    source_dir: str,
    target_cache_dir: str,
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """Resolve and optionally preprocess the bootstrap prompt into the target cache."""
    prompt_path = _resolve_prompt(source_dir)
    if not prompt_path:
        return None

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except OSError as exc:
        logger.warning("Bootstrap: failed to read prompt %s: %s", prompt_path, exc)
        return None

    final_prompt = _preprocess_bootstrap_prompt(
        prompt_text=prompt_text,
        backend=backend,
        model_name=model_name,
    )

    output_path = os.path.join(target_cache_dir, SANITIZED_PROMPT_FILENAME)
    try:
        os.makedirs(target_cache_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_prompt + "\n")
    except OSError as exc:
        logger.warning("Bootstrap: failed to write sanitized prompt %s: %s", output_path, exc)
        return prompt_path

    return output_path


def _extract_numeric_id(entry_id: str) -> Optional[int]:
    """Extract the numeric suffix from auto-generated IDs like ``mem_0042``."""
    m = re.search(r"(\d+)$", entry_id)
    return int(m.group(1)) if m else None


def _repath_entries(data: dict) -> int:
    """Mutate entries in-place: prefix paths with ``bootstrapped/``, tag source.

    Returns the count of entries processed.
    """
    entries = data.get("entries", {})
    for entry in entries.values():
        original_path = entry.get("path", "")
        entry["path"] = f"bootstrapped/{original_path}" if original_path else "bootstrapped"
        entry["source"] = "bootstrapped"
    return len(entries)


def _merge_into_target(source_data: dict, target_file: str) -> int:
    """Merge bootstrapped entries into an existing target store file.

    If the target doesn't exist or is empty, the source data is written
    directly.  Otherwise bootstrapped entries are added alongside existing
    ones and ``next_id`` is reconciled.

    Returns the number of bootstrapped entries written.
    """
    source_entries = source_data.get("entries", {})
    if not source_entries:
        return 0

    target_data: Dict[str, Any] = {"next_id": 1, "entries": {}}

    if os.path.isfile(target_file):
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and "entries" in existing:
                target_data = existing
        except (json.JSONDecodeError, OSError):
            pass

    for eid, entry in source_entries.items():
        if eid in target_data["entries"]:
            eid = f"bs_{eid}"
            entry["id"] = eid
        target_data["entries"][eid] = entry

    max_numeric = target_data.get("next_id", 1)
    for eid in target_data["entries"]:
        n = _extract_numeric_id(eid)
        if n is not None and n >= max_numeric:
            max_numeric = n + 1
    target_data["next_id"] = max_numeric

    os.makedirs(os.path.dirname(target_file) or ".", exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(target_data, f, indent=2, ensure_ascii=False)

    return len(source_entries)


def bootstrap_stores(
    source_dir: str,
    target_cache_dir: str,
    prompt_backend: Optional[str] = None,
    prompt_model_name: Optional[str] = None,
) -> dict:
    """Load stores from *source_dir*, re-path under ``bootstrapped/``, write to *target_cache_dir*.

    Returns a summary dict::

        {"skills": N, "memory": N, "subagents": N, "prompt_path": str|None}
    """
    if not os.path.isdir(source_dir):
        print(f"ERROR: Bootstrap source directory does not exist: {source_dir}")
        sys.exit(1)

    found_any = False
    for sf in STORE_FILES:
        if os.path.isfile(os.path.join(source_dir, sf)):
            found_any = True
            break

    if not found_any:
        print(
            f"ERROR: Bootstrap source directory contains no store files "
            f"({', '.join(STORE_FILES)}): {source_dir}"
        )
        sys.exit(1)

    summary: Dict[str, Any] = {"skills": 0, "memory": 0, "subagents": 0, "prompt_path": None}

    label_map = {
        "memory.json": "memory",
        "skills.json": "skills",
        "subagents.json": "subagents",
    }

    for store_file in STORE_FILES:
        src_path = os.path.join(source_dir, store_file)
        if not os.path.isfile(src_path):
            logger.info("Bootstrap: %s not found in source, skipping", store_file)
            continue

        try:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Bootstrap: failed to read %s: %s", src_path, exc)
            continue

        count = _repath_entries(data)
        target_file = os.path.join(target_cache_dir, store_file)
        written = _merge_into_target(data, target_file)
        summary[label_map[store_file]] = written
        logger.info("Bootstrap: wrote %d entries from %s", written, store_file)

    summary["prompt_path"] = _materialize_prompt_override(
        source_dir=source_dir,
        target_cache_dir=target_cache_dir,
        backend=prompt_backend,
        model_name=prompt_model_name,
    )
    if summary["prompt_path"]:
        logger.info("Bootstrap: resolved evolved prompt at %s", summary["prompt_path"])

    return summary
