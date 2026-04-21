"""MolmoWeb-8B click oracle inference server.

Loads ``allenai/MolmoWeb-8B`` once into VRAM and exposes a single HTTP
endpoint that the browser harness calls when gemma4 wants to click a
named element. The model produces a full web-agent JSON response of
the form::

    {
      "thought": "...",
      "action": {
        "name": "click",
        "x": 49.9,
        "y": 78.3,
        "button": "left",
        "click_type": "single"
      }
    }

We extract the (x, y) coordinates, convert from MolmoWeb's 0-100
normalized scale to absolute viewport pixels using the supplied canvas
width/height, and return them. The verbose thought is preserved in the
response for logging / trajectory analysis.

Why a separate process: MolmoWeb needs ~17 GB of VRAM in bf16. We pin
it to one GPU (default cuda:0) and let the existing Ollama instance
running gemma4:26b stay on the other GPU. The two never share a card.

Run::

    .venv/bin/python -m molmoweb_server.server --port 11436 --gpu 0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
from typing import Any, Optional

logger = logging.getLogger("molmoweb_server")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logger.addHandler(_h)
logger.propagate = False

# ----------------------------------------------------------------------
# Prompt template
# ----------------------------------------------------------------------
#
# This is the template MolmoWeb-8B was trained against (recovered from
# the upstream allenai/molmoweb repo, agent/multimodal_agent.py). The
# system_message string is the literal "molmo_web_think" prefix — it's
# not a verbose system prompt, it's a style identifier the model
# learned to associate with web-action JSON output.
SYSTEM_MESSAGE = "molmo_web_think"

# Jinja-style template (we render manually with str.format to avoid the
# jinja2 dep at server import time — the template is simple enough).
USER_MSG_TEMPLATE = """
# GOAL
{task_description}

# PREVIOUS STEPS

# CURRENTLY ACTIVE PAGE
Page 1: {page_title} | {page_url}

# NEXT STEP
"""


# Verbs that signal a click intent. We auto-prefix "Click on " when the
# description doesn't start with one of these — empirically MolmoWeb
# returns wildly different action types (scroll, browser_nav,
# keyboard_press) when the description is a bare noun phrase like
# "the START button" instead of an imperative like "Click the START
# button". The standalone probe confirmed this: "Click on X" → click
# action, bare "X" → random action class.
_CLICK_VERBS = (
    "click", "press", "tap", "select", "choose", "open", "activate",
    "find and click",
)


def build_prompt(task_description: str, page_title: str, page_url: str) -> str:
    """Compose the full text prompt MolmoWeb expects."""
    desc = task_description.strip()
    desc_lower = desc.lower()
    if not any(desc_lower.startswith(v) for v in _CLICK_VERBS):
        desc = f"Click on {desc}"
    user_msg = USER_MSG_TEMPLATE.format(
        task_description=desc,
        page_title=page_title,
        page_url=page_url,
    )
    return f"{SYSTEM_MESSAGE}: {user_msg}"


# ----------------------------------------------------------------------
# Output parser
# ----------------------------------------------------------------------

# Strict JSON path: the model emits exactly the {thought, action}
# shape we saw in probes. Sometimes there's a leading space, occasionally
# stray tokens — strip and json.loads.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_output(raw: str) -> dict[str, Any]:
    """Parse a MolmoWeb-8B raw response into a structured dict.

    Returns ``{"thought", "action_name", "x_norm", "y_norm", "button",
    "click_type", "raw"}`` on success, or ``{"error", "raw"}`` on
    failure. Coordinates are still on the 0-100 normalized scale; the
    caller is responsible for converting them to viewport pixels.
    """
    raw_stripped = raw.strip()
    # Find the first JSON object in the response — be tolerant of
    # leading whitespace, stray prefixes, or trailing tokens.
    m = _JSON_OBJECT_RE.search(raw_stripped)
    if m is None:
        return {"error": "no JSON object in model output", "raw": raw}
    json_str = m.group(0)
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"json decode failed: {e}", "raw": raw}

    thought = obj.get("thought", "")
    action = obj.get("action") or {}
    if not isinstance(action, dict):
        return {"error": f"action is not a dict: {type(action).__name__}", "raw": raw}

    name = action.get("name", "")
    x_norm = action.get("x")
    y_norm = action.get("y")
    button = action.get("button", "left")
    click_type = action.get("click_type", "single")

    if x_norm is None or y_norm is None:
        # Non-click actions (send_msg_to_user, type, scroll, etc) don't
        # carry coordinates. Return them so callers can decide what to
        # do — we just won't have a click target.
        return {
            "thought": thought,
            "action_name": name,
            "x_norm": None,
            "y_norm": None,
            "button": button,
            "click_type": click_type,
            "raw_action": action,
            "raw": raw,
        }

    return {
        "thought": thought,
        "action_name": name,
        "x_norm": float(x_norm),
        "y_norm": float(y_norm),
        "button": button,
        "click_type": click_type,
        "raw_action": action,
        "raw": raw,
    }


def normalized_to_pixels(x_norm: float, y_norm: float, width: int, height: int) -> tuple[int, int]:
    """Convert MolmoWeb's 0-100 normalized coordinates to viewport pixels.

    Coordinates are clamped to [0, width-1] x [0, height-1] to defend
    against rare cases where the model returns slightly out-of-range
    values (we've seen >100 once during probing).
    """
    px = round(x_norm * width / 100.0)
    py = round(y_norm * height / 100.0)
    px = max(0, min(px, max(0, width - 1)))
    py = max(0, min(py, max(0, height - 1)))
    return px, py


# ----------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------


class MolmoWebModel:
    """Lazy singleton wrapping the HF model + processor.

    The model is loaded on first call (not at server import time) so the
    process can start fast and respond to /health immediately. ~6
    seconds to load 8 shards on a warm cache.
    """

    _instance: Optional["MolmoWebModel"] = None

    def __init__(self, checkpoint: str, device: str, dtype_str: str):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.checkpoint = checkpoint
        self.device = device
        self.dtype_str = dtype_str

        if dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        elif dtype_str == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"unknown dtype: {dtype_str}")

        logger.info(
            "Loading %s in %s on %s ...", checkpoint, dtype_str, device
        )
        t0 = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="sdpa",
            device_map={"": device},
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            padding_side="left",
        )
        load_time = time.time() - t0
        vram_gb = torch.cuda.memory_allocated(int(device.split(":")[1])) / 1e9
        logger.info(
            "MolmoWeb loaded in %.1fs, VRAM in use: %.1f GB", load_time, vram_gb
        )

    @classmethod
    def get(cls, checkpoint: str, device: str, dtype_str: str) -> "MolmoWebModel":
        if cls._instance is None:
            cls._instance = cls(checkpoint, device, dtype_str)
        return cls._instance

    def find_element(
        self,
        image,
        description: str,
        page_title: str = "Game",
        page_url: str = "about:blank",
        max_new_tokens: int = 200,
    ) -> dict[str, Any]:
        """Run one inference pass and return the parsed result.

        Returns a dict with at least:
            - ``raw``: the model's verbatim text output
            - ``thought``: the parsed thought field
            - ``action_name``: e.g. "click"
            - ``x_norm``, ``y_norm``: 0-100 normalized coordinates (or None)
            - ``latency_s``: wall time for the generate() call
        """
        import torch

        prompt = build_prompt(description, page_title, page_url)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
        # The processor returns token_type_ids on some configs but the
        # model's forward signature doesn't accept it — strip before
        # moving to device.
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
        }

        t0 = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency = time.time() - t0
        generated = output[0, inputs["input_ids"].size(1):]
        raw_text = self.processor.decode(generated, skip_special_tokens=True)
        parsed = parse_model_output(raw_text)
        parsed["latency_s"] = latency
        return parsed


# ----------------------------------------------------------------------
# FastAPI server
# ----------------------------------------------------------------------


def make_app(checkpoint: str, device: str, dtype_str: str):
    from fastapi import FastAPI, HTTPException
    from PIL import Image

    app = FastAPI(title="MolmoWeb click oracle")

    # Eager-load the model on startup so the first /find_element call
    # doesn't pay the 6-second checkpoint load. Costs ~6s of startup
    # time, gains ~6s of latency on the first request.
    @app.on_event("startup")
    async def _startup() -> None:
        MolmoWebModel.get(checkpoint, device, dtype_str)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        loaded = MolmoWebModel._instance is not None
        return {"status": "ok", "model_loaded": loaded, "checkpoint": checkpoint}

    @app.post("/find_element")
    async def find_element(req: dict) -> dict[str, Any]:
        """Find a named element in a screenshot.

        Request body::
            {
              "screenshot_b64": "<base64 PNG>",
              "description": "the START button",
              "canvas_width": 960,
              "canvas_height": 600,
              "page_title": "Folder Dungeon",   // optional
              "page_url": "https://...",         // optional
              "max_new_tokens": 200              // optional
            }

        Response::
            {
              "success": true,
              "x": 480,            // pixel coords (rounded)
              "y": 455,
              "x_norm": 49.9,      // raw 0-100 from the model
              "y_norm": 78.3,
              "action_name": "click",
              "button": "left",
              "click_type": "single",
              "thought": "...",
              "latency_s": 1.6,
              "raw": "..."
            }
        """
        screenshot_b64 = req.get("screenshot_b64") or ""
        description = (req.get("description") or "").strip()
        canvas_width = int(req.get("canvas_width") or 0)
        canvas_height = int(req.get("canvas_height") or 0)
        page_title = req.get("page_title") or "Game"
        page_url = req.get("page_url") or "about:blank"
        max_new_tokens = int(req.get("max_new_tokens") or 200)

        if not screenshot_b64:
            raise HTTPException(status_code=400, detail="screenshot_b64 is required")
        if not description:
            raise HTTPException(status_code=400, detail="description is required")
        if canvas_width <= 0 or canvas_height <= 0:
            raise HTTPException(
                status_code=400,
                detail="canvas_width and canvas_height are required and must be positive",
            )

        try:
            image_bytes = base64.b64decode(screenshot_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to decode screenshot: {e}")

        model = MolmoWebModel.get(checkpoint, device, dtype_str)
        try:
            result = model.find_element(
                image=image,
                description=description,
                page_title=page_title,
                page_url=page_url,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            logger.exception("inference failed")
            raise HTTPException(status_code=500, detail=f"inference failed: {e}")

        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "raw": result.get("raw", ""),
                "latency_s": result.get("latency_s", 0.0),
            }

        x_norm = result.get("x_norm")
        y_norm = result.get("y_norm")
        if x_norm is None or y_norm is None:
            return {
                "success": False,
                "error": (
                    f"model returned non-click action: {result.get('action_name')}"
                ),
                "action_name": result.get("action_name"),
                "thought": result.get("thought", ""),
                "raw_action": result.get("raw_action"),
                "raw": result.get("raw", ""),
                "latency_s": result.get("latency_s", 0.0),
            }

        px, py = normalized_to_pixels(x_norm, y_norm, canvas_width, canvas_height)
        logger.info(
            "find_element('%s') -> (%d, %d) [norm %.1f, %.1f] in %.2fs",
            description[:60], px, py, x_norm, y_norm, result.get("latency_s", 0.0),
        )
        return {
            "success": True,
            "x": px,
            "y": py,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "action_name": result.get("action_name", "click"),
            "button": result.get("button", "left"),
            "click_type": result.get("click_type", "single"),
            "thought": result.get("thought", ""),
            "latency_s": result.get("latency_s", 0.0),
            "raw": result.get("raw", ""),
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("MOLMOWEB_CHECKPOINT", "allenai/MolmoWeb-8B"),
        help="HF checkpoint id or local path",
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("MOLMOWEB_GPU", "0"),
        help="CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("MOLMOWEB_DTYPE", "bfloat16"),
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16, recommended for Hopper/Blackwell)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MOLMOWEB_PORT", "11436")),
        help="HTTP port (default: 11436)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MOLMOWEB_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    logger.info(
        "Starting MolmoWeb click oracle: %s on %s, dtype=%s, listen=%s:%d",
        args.checkpoint, device, args.dtype, args.host, args.port,
    )

    app = make_app(args.checkpoint, device, args.dtype)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
