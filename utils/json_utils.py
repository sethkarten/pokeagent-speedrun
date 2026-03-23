"""JSON serialization utilities for converting non-standard Python types to JSON-safe values.

Includes protobuf-to-native converters used by VLM agent scaffolds (Gemini SDK
returns ``MapComposite`` / ``RepeatedComposite`` instead of plain dicts/lists).
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def serialize_for_json(obj):
    """Recursively convert non-JSON-serializable objects to JSON-compatible types.

    Handles:
    - IntEnum/Enum -> int (via .value)
    - numpy types -> native Python int/float
    - dicts -> recursively serialize values
    - lists/tuples -> recursively serialize items
    - Objects with __dict__ -> convert to dict
    - None/bool/str/int/float -> pass through
    """
    from enum import IntEnum, Enum
    import numpy as np

    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj

    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            import base64

            return base64.b64encode(obj).decode("utf-8")

    try:
        if isinstance(
            obj,
            (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64),
        ):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:
        pass

    if isinstance(obj, (IntEnum, Enum)):
        return obj.value

    if isinstance(obj, dict):
        return {str(k): serialize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, set):
        return [serialize_for_json(item) for item in obj]

    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]

    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            return serialize_for_json(obj.__dict__)
        except Exception:
            pass

    try:
        return str(obj)
    except Exception:
        logger.warning(f"Could not serialize object of type {type(obj)}: {obj}")
        return None


# ---------------------------------------------------------------------------
# Protobuf / MapComposite → native Python converters
# ---------------------------------------------------------------------------
# The Gemini SDK returns tool-call arguments as proto-plus wrapper types
# (``MapComposite``, ``RepeatedComposite``, ``RepeatedScalar``).  Other
# backends (OpenAI, Anthropic) already return plain ``dict`` / ``list``.
# These helpers are intentionally backend-agnostic: they are no-ops on
# values that are already native Python types.


def _is_protobuf(value: Any) -> bool:
    """Return True if *value* is a proto-plus / protobuf wrapper type."""
    return (
        value is not None
        and hasattr(value, "__class__")
        and "proto" in getattr(value.__class__, "__module__", "")
    )


def convert_protobuf_value(value: Any) -> Any:
    """Recursively convert a protobuf value to a JSON-serialisable Python type."""
    if value is None:
        return None

    if _is_protobuf(value):
        try:
            dict_value = dict(value)
            return {k: convert_protobuf_value(v) for k, v in dict_value.items()}
        except (TypeError, ValueError):
            pass

        if hasattr(value, "__iter__") and not isinstance(value, (str, dict)):
            try:
                return [convert_protobuf_value(item) for item in value]
            except Exception:
                try:
                    return list(value)
                except Exception:
                    return value

        return value

    if isinstance(value, dict):
        return {k: convert_protobuf_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [convert_protobuf_value(item) for item in value]

    return value


def convert_protobuf_args(proto_args) -> Dict[str, Any]:
    """Convert a protobuf argument mapping to a plain ``dict[str, Any]``.

    Falls back to ``str(value)`` for individual keys that resist conversion.
    """
    arguments: Dict[str, Any] = {}
    for key, value in proto_args.items():
        try:
            arguments[key] = convert_protobuf_value(value)
        except Exception:
            logger.warning("convert_protobuf_args: key '%s' fell back to str()", key, exc_info=True)
            arguments[key] = str(value)
    return arguments


def normalize_replan_edits(edits: Any) -> List[Dict[str, Any]]:
    """Coerce ``edits`` into ``list[dict]`` for ``replan_objectives``.

    Gemini may return the array as a ``RepeatedComposite``, a dict with
    numeric string keys, a single edit dict, or a JSON string.
    """
    raw_list: List[Any]

    if edits is None:
        raw_list = []
    elif isinstance(edits, list):
        raw_list = edits
    elif isinstance(edits, tuple):
        raw_list = list(edits)
    elif isinstance(edits, str):
        try:
            parsed = json.loads(edits)
            return normalize_replan_edits(parsed)
        except json.JSONDecodeError:
            return []
    elif isinstance(edits, dict):
        keys = list(edits.keys())
        if keys and all(str(k).isdigit() for k in keys):
            raw_list = [edits[k] for k in sorted(keys, key=lambda x: int(str(x)))]
        elif "index" in edits:
            raw_list = [edits]
        else:
            raw_list = []
    elif hasattr(edits, "__iter__") and not isinstance(edits, (str, bytes, dict)):
        try:
            raw_list = list(edits)
        except Exception:
            raw_list = []
    else:
        raw_list = []

    out: List[Dict[str, Any]] = []
    for item in raw_list:
        if isinstance(item, dict):
            out.append({k: convert_protobuf_value(v) for k, v in item.items()})
        else:
            converted = convert_protobuf_value(item)
            if isinstance(converted, dict):
                out.append(converted)
    return out
