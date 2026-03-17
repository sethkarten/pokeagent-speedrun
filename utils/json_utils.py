"""JSON serialization utilities for converting non-standard Python types to JSON-safe values."""

import logging
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
