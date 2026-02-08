from io import BytesIO
from PIL import Image
import json
import os
import sys
import base64
import random
import time
import threading
import logging
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
import numpy as np

# Set up module logging
logger = logging.getLogger(__name__)

# Import LLM logger
from utils.llm_logger import log_llm_interaction, log_llm_error


# Define the retry decorator with exponential backoff
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (Exception,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                # Increase the delay with exponential factor and random jitter
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper


class VLMBackend(ABC):
    """Abstract base class for VLM backends"""

    @abstractmethod
    def get_query(
        self,
        img: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        text: str,
        module_name: str = "Unknown",
    ) -> str:
        """Process an image (or list of images) and text prompt"""
        pass

    @abstractmethod
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt"""
        pass


def _openai_tool_call_part(name: str, args: Dict[str, Any]):
    """Create a Gemini-compatible part object for agent consumption."""

    class FunctionCallPart:
        def __init__(self, fn_name: str, fn_args: Dict[str, Any]):
            self.name = fn_name
            self.args = fn_args

    part = type("Part", (), {})()
    part.function_call = FunctionCallPart(name, args)
    part.text = ""  # No text; agents join part.text and expect str
    return part


def _openai_text_part(text: str):
    """Create a Gemini-compatible text part for agent consumption."""
    part = type("Part", (), {})()
    part.function_call = None
    part.text = text
    return part


def _openai_responses_adapter(response) -> Any:
    """Adapt OpenAI Responses API output to Gemini-like structure for agents.

    Responses API returns response.output: list of items (function_call, message, etc.).
    Agents expect candidates[0].content.parts with .function_call / .text.
    """
    parts = []
    output = getattr(response, "output", None) or []

    for item in output:
        item_type = getattr(item, "type", None) if hasattr(item, "type") else (item.get("type") if isinstance(item, dict) else None)
        if item_type == "function_call":
            name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "")
            raw_args = getattr(item, "arguments", None) or (item.get("arguments", "{}") if isinstance(item, dict) else "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            parts.append(_openai_tool_call_part(name, args))
        elif item_type == "message":
            content_list = getattr(item, "content", None) or (item.get("content", []) if isinstance(item, dict) else [])
            for c in content_list:
                c_type = getattr(c, "type", None) if hasattr(c, "type") else (c.get("type") if isinstance(c, dict) else None)
                if c_type == "output_text":
                    text = getattr(c, "text", None) or (c.get("text", "") if isinstance(c, dict) else "")
                    if text:
                        parts.append(_openai_text_part(text))

    if not parts:
        # Fallback: output_text convenience property on response (SDK)
        text = getattr(response, "output_text", None) or ""
        parts.append(_openai_text_part(text if text else ""))

    content = type("Content", (), {"parts": parts})()
    candidate = type("Candidate", (), {"content": content})()
    adapter = type("ResponseAdapter", (), {"candidates": [candidate]})()
    adapter.usage_metadata = None
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        inp = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
        out = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0)
        total = getattr(usage, "total_tokens", 0) or (inp + out)
        cached = 0
        if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
            cached = getattr(usage.input_tokens_details, "cached_tokens", 0)
        adapter.usage_metadata = type(
            "UsageMetadata",
            (),
            {
                "prompt_token_count": inp,
                "candidates_token_count": out,
                "total_token_count": total,
                "cached_content_token_count": cached,
            },
        )()
    return adapter


class OpenAIBackend(VLMBackend):
    """OpenAI API backend with tool calling and system instructions.

    Modeled after GeminiBackend: supports tools, system_instruction, dual mode
    (function calling returns adapter object; text-only returns string).
    """

    def __init__(self, model_name: str, tools: list = None, system_instruction: str = None, **kwargs):
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")

        self.model_name = model_name
        self.tools = tools or []
        self.system_instruction = system_instruction
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("Error: OpenAI API key is missing! Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.errors = (openai.RateLimitError,)

        if self.tools:
            self._tools_openai = self._convert_tools_to_openai_format()
            log_parts = [f"OpenAI backend initialized with model: {model_name}", f"{len(self.tools)} tools"]
        else:
            self._tools_openai = []
            log_parts = [f"OpenAI backend initialized with model: {model_name}"]
        if self.system_instruction:
            log_parts.append(f"system instructions ({len(self.system_instruction)} chars)")
        logger.info(", ".join(log_parts))

    def _setup_function_calling(self):
        """Update tools (called when agent dynamically updates tool list)."""
        self._tools_openai = self._convert_tools_to_openai_format() if self.tools else []
        logger.info(f"OpenAI model updated with {len(self.tools) if self.tools else 0} tools")

    def _convert_tools_to_openai_format(self) -> list:
        """Convert Gemini-style tool declarations to OpenAI format.

        Responses API expects flat format: {"type":"function","name":...,"description":...,"parameters":...}
        (not nested under "function" like Chat Completions).
        """
        result = []
        for tool in self.tools:
            params = tool.get("parameters", {})
            properties, required = self._build_json_schema_properties(params)
            # Responses API: flat format with name/description/parameters at top level
            result.append({
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {"type": "object", "properties": properties, "required": required},
            })
        return result

    def _build_json_schema_properties(self, params: dict) -> tuple:
        """Build JSON Schema properties from Gemini-style params. Returns (properties, required)."""
        properties = {}
        required = params.get("required", [])
        for prop_name, prop_def in params.get("properties", {}).items():
            t = prop_def.get("type_", "STRING")
            if t == "ARRAY":
                t = "array"
            elif t == "INTEGER":
                t = "integer"
            elif t == "BOOLEAN":
                t = "boolean"
            else:
                t = "string"
            prop = {"type": t, "description": prop_def.get("description", "")}
            if t == "array" and "items" in prop_def:
                items_t = prop_def["items"].get("type_", "STRING") if isinstance(prop_def["items"], dict) else "STRING"
                prop["items"] = {"type": "string" if items_t == "STRING" else "string"}
            if "enum" in prop_def:
                prop["enum"] = prop_def["enum"]
            properties[prop_name] = prop
        return properties, required

    @retry_with_exponential_backoff
    def _call_responses(self, instructions: str | None, input_data, tools: list = None):
        """Calls the Responses API (v1/responses) with optional tools.

        Supports Codex and other models that require the Responses API.
        """
        kwargs = {"model": self.model_name}
        if instructions:
            kwargs["instructions"] = instructions
        if isinstance(input_data, str):
            kwargs["input"] = input_data
        else:
            kwargs["input"] = input_data if isinstance(input_data, list) else [input_data]
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.responses.create(**kwargs)

    def _prepare_image_base64(self, img: Union[Image.Image, np.ndarray]) -> str:
        """Prepare image as base64 string"""
        if hasattr(img, "convert"):  # It's a PIL Image
            image = img
        elif hasattr(img, "shape"):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _extract_thinking_from_response(self, response) -> str:
        """Extract reasoning for logging from response/adapter (candidates[0].content.parts structure).
        
        Prioritizes function calls over text parts to ensure the command name is included,
        since some backends (like Anthropic) return text explanations before tool calls.
        """
        if not response or not getattr(response, "candidates", None):
            return "[Executing function call]"
        candidate = response.candidates[0]
        content = getattr(candidate, "content", None)
        if not content or not getattr(content, "parts", None):
            return "[Executing function call]"
        
        # First pass: prioritize function calls to include command name in output
        for part in content.parts:
            fc = getattr(part, "function_call", None)
            if fc:
                args = getattr(fc, "args", {}) or {}
                reasoning = args.get("reasoning") or args.get("reason", "")
                if reasoning:
                    return f"[{fc.name}] {reasoning}"
                return f"Calling {fc.name}({list(args.keys())[:3]})"
        
        # Second pass: fall back to text parts if no function call found
        for part in content.parts:
            if getattr(part, "text", None):
                return part.text
        
        return "[Executing function call]"

    def get_query(
        self,
        img: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        text: str,
        module_name: str = "Unknown",
    ) -> Union[str, Any]:
        """Process an image (or list of images) and text prompt using OpenAI API.

        Returns:
            - If tools configured: Adapter object (Gemini-like) for agent function calling
            - If no tools: String text response
        """
        start_time = time.time()

        # Build Responses API input (input_text + input_image)
        content_parts = [{"type": "input_text", "text": text}]
        if isinstance(img, list):
            for i in img:
                image_base64 = self._prepare_image_base64(i)
                content_parts.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"})
        else:
            image_base64 = self._prepare_image_base64(img)
            content_parts.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"})

        input_data = {"role": "user", "content": content_parts}

        try:
            response = self._call_responses(
                self.system_instruction,
                input_data,
                tools=self._tools_openai if self._tools_openai else None,
            )
            duration = time.time() - start_time

            token_usage = {}
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                inp = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0)
                out = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0)
                cached = 0
                if hasattr(u, "input_tokens_details") and u.input_tokens_details:
                    cached = getattr(u.input_tokens_details, "cached_tokens", 0)
                token_usage = {
                    "prompt_tokens": inp,
                    "completion_tokens": out,
                    "total_tokens": getattr(u, "total_tokens", 0) or (inp + out),
                    "cached_tokens": cached,
                }

            if self.tools:
                adapter = _openai_responses_adapter(response)
                thinking_text = self._extract_thinking_from_response(adapter)
                log_llm_interaction(
                    interaction_type=f"openai_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "openai",
                        "has_image": True,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "openai"},
                )
                return adapter
            else:
                result = getattr(response, "output_text", None) or ""
                log_llm_interaction(
                    interaction_type=f"openai_{module_name}",
                    prompt=text,
                    response=result,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "openai",
                        "has_image": True,
                        "token_usage": token_usage,
                    },
                    model_info={"model": self.model_name, "backend": "openai"},
                )
                return result
        except Exception as e:
            duration = time.time() - start_time
            err_msg = str(e)
            if hasattr(e, "body") and e.body:
                err_msg = f"{err_msg} | body: {e.body}"
            elif hasattr(e, "response") and e.response is not None:
                try:
                    body = getattr(e.response, "json", lambda: None)()
                    if body is None and hasattr(e.response, "text"):
                        body = e.response.text
                    if body:
                        err_msg = f"{err_msg} | body: {body}"
                except Exception:
                    pass
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=err_msg,
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": True},
            )
            logger.error(f"OpenAI API error: {err_msg}")
            raise

    def get_text_query(self, text: str, module_name: str = "Unknown") -> Union[str, Any]:
        """Process a text-only prompt using OpenAI API.

        Returns:
            - If tools configured: Adapter object for agent function calling
            - If no tools: String text response
        """
        start_time = time.time()

        try:
            response = self._call_responses(
                self.system_instruction,
                text,
                tools=self._tools_openai if self._tools_openai else None,
            )
            duration = time.time() - start_time

            token_usage = {}
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                inp = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0)
                out = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0)
                cached = 0
                if hasattr(u, "input_tokens_details") and u.input_tokens_details:
                    cached = getattr(u.input_tokens_details, "cached_tokens", 0)
                token_usage = {
                    "prompt_tokens": inp,
                    "completion_tokens": out,
                    "total_tokens": getattr(u, "total_tokens", 0) or (inp + out),
                    "cached_tokens": cached,
                }

            if self.tools:
                adapter = _openai_responses_adapter(response)
                thinking_text = self._extract_thinking_from_response(adapter)
                log_llm_interaction(
                    interaction_type=f"openai_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "openai",
                        "has_image": False,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "openai"},
                )
                return adapter
            else:
                result = getattr(response, "output_text", None) or ""
                log_llm_interaction(
                    interaction_type=f"openai_{module_name}",
                    prompt=text,
                    response=result,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "openai",
                        "has_image": False,
                        "token_usage": token_usage,
                    },
                    model_info={"model": self.model_name, "backend": "openai"},
                )
                return result
        except Exception as e:
            duration = time.time() - start_time
            err_msg = str(e)
            if hasattr(e, "body") and e.body:
                err_msg = f"{err_msg} | body: {e.body}"
            elif hasattr(e, "response") and e.response is not None:
                try:
                    body = getattr(e.response, "json", lambda: None)()
                    if body is None and hasattr(e.response, "text"):
                        body = e.response.text
                    if body:
                        err_msg = f"{err_msg} | body: {body}"
                except Exception:
                    pass
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=err_msg,
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": False},
            )
            logger.error(f"OpenAI API error: {err_msg}")
            raise


def _anthropic_response_adapter(response) -> Any:
    """Adapt Anthropic Messages API output to Gemini-like structure for agents.

    Response has .content: list of blocks (type 'text' | 'tool_use').
    Agents expect candidates[0].content.parts with .function_call / .text.
    Reuses _openai_tool_call_part and _openai_text_part (same part shape).
    """
    parts = []
    content = getattr(response, "content", None) or []

    for block in content:
        block_type = getattr(block, "type", None) if hasattr(block, "type") else (block.get("type") if isinstance(block, dict) else None)
        if block_type == "tool_use":
            name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else "")
            inp = getattr(block, "input", None) or (block.get("input", {}) if isinstance(block, dict) else {})
            args = inp if isinstance(inp, dict) else {}
            if name:
                parts.append(_openai_tool_call_part(name, args))
        elif block_type == "text":
            text = getattr(block, "text", None) or (block.get("text", "") if isinstance(block, dict) else "")
            if text:
                parts.append(_openai_text_part(text))

    if not parts:
        parts.append(_openai_text_part(""))

    content_obj = type("Content", (), {"parts": parts})()
    candidate = type("Candidate", (), {"content": content_obj})()
    adapter = type("ResponseAdapter", (), {"candidates": [candidate]})()
    adapter.usage_metadata = None
    if hasattr(response, "usage") and response.usage:
        u = response.usage
        inp = getattr(u, "input_tokens", 0)
        out = getattr(u, "output_tokens", 0)
        total = inp + out
        cached = getattr(u, "cache_creation_input_tokens", 0) or 0
        cached += getattr(u, "cache_read_input_tokens", 0) or 0
        adapter.usage_metadata = type(
            "UsageMetadata",
            (),
            {
                "prompt_token_count": inp,
                "candidates_token_count": out,
                "total_token_count": total,
                "cached_content_token_count": cached,
            },
        )()
    return adapter


class AnthropicBackend(VLMBackend):
    """Anthropic Messages API backend with tool calling and system prompt.

    Uses POST /v1/messages (client.messages.create). Supports tools, system,
    and vision (image blocks). Returns adapter when tools configured, else string.
    """

    def __init__(self, model_name: str, tools: list = None, system_instruction: str = None, **kwargs):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not found. Install with: pip install anthropic")

        self.model_name = model_name
        self.tools = tools or []
        self.system_instruction = system_instruction
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required. Set the environment variable.")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.errors = (anthropic.RateLimitError,)

        if self.tools:
            self._tools_anthropic = self._convert_tools_to_anthropic_format()
            log_parts = [f"Anthropic backend initialized with model: {model_name}", f"{len(self.tools)} tools"]
        else:
            self._tools_anthropic = []
            log_parts = [f"Anthropic backend initialized with model: {model_name}"]
        if self.system_instruction:
            log_parts.append(f"system instructions ({len(self.system_instruction)} chars)")
        logger.info(", ".join(log_parts))

    def _setup_function_calling(self):
        """Update tools when agent dynamically updates tool list."""
        self._tools_anthropic = self._convert_tools_to_anthropic_format() if self.tools else []
        logger.info(f"Anthropic model updated with {len(self.tools) if self.tools else 0} tools")

    def _build_input_schema(self, params: dict) -> tuple:
        """Build JSON Schema properties from Gemini-style params. Returns (properties, required)."""
        properties = {}
        required = params.get("required", [])
        for prop_name, prop_def in params.get("properties", {}).items():
            t = prop_def.get("type_", "STRING")
            if t == "ARRAY":
                t = "array"
            elif t == "INTEGER":
                t = "integer"
            elif t == "BOOLEAN":
                t = "boolean"
            else:
                t = "string"
            prop = {"type": t, "description": prop_def.get("description", "")}
            if t == "array" and "items" in prop_def:
                items_t = prop_def["items"].get("type_", "STRING") if isinstance(prop_def["items"], dict) else "STRING"
                prop["items"] = {"type": "string" if items_t == "STRING" else "string"}
            if "enum" in prop_def:
                prop["enum"] = prop_def["enum"]
            properties[prop_name] = prop
        return properties, required

    def _convert_tools_to_anthropic_format(self) -> list:
        """Convert Gemini-style tool declarations to Anthropic input_schema format."""
        result = []
        for tool in self.tools:
            params = tool.get("parameters", {})
            properties, required = self._build_input_schema(params)
            result.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": {"type": "object", "properties": properties, "required": required},
            })
        return result

    @retry_with_exponential_backoff
    def _call_messages(self, system: Optional[str], messages: list, tools: Optional[list] = None):
        """Call Anthropic Messages API."""
        kwargs = {
            "model": self.model_name,
            "max_tokens": 8192,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**kwargs)

    def _prepare_image_base64(self, img: Union[Image.Image, np.ndarray]) -> str:
        """Prepare image as base64 string."""
        if hasattr(img, "convert"):
            image = img
        elif hasattr(img, "shape"):
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _extract_thinking_from_response(self, response) -> str:
        """Extract reasoning for logging from adapter (candidates[0].content.parts).
        
        Prioritizes function calls over text parts to ensure the command name is included,
        since Anthropic often returns text explanations before tool calls.
        """
        if not response or not getattr(response, "candidates", None):
            return "[Executing function call]"
        candidate = response.candidates[0]
        content = getattr(candidate, "content", None)
        if not content or not getattr(content, "parts", None):
            return "[Executing function call]"
        
        # First pass: prioritize function calls to include command name in output
        for part in content.parts:
            fc = getattr(part, "function_call", None)
            if fc:
                args = getattr(fc, "args", {}) or {}
                reasoning = args.get("reasoning") or args.get("reason", "")
                if reasoning:
                    return f"[{fc.name}] {reasoning}"
                return f"Calling {fc.name}({list(args.keys())[:3]})"
        
        # Second pass: fall back to text parts if no function call found
        for part in content.parts:
            if getattr(part, "text", None):
                return part.text
        
        return "[Executing function call]"

    def get_query(
        self,
        img: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        text: str,
        module_name: str = "Unknown",
    ) -> Union[str, Any]:
        """Process image(s) and text. Returns adapter if tools, else string."""
        start_time = time.time()
        content = [{"type": "text", "text": text}]
        if isinstance(img, list):
            for i in img:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": self._prepare_image_base64(i)},
                })
        else:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": self._prepare_image_base64(img)},
            })
        messages = [{"role": "user", "content": content}]

        try:
            response = self._call_messages(
                self.system_instruction,
                messages,
                tools=self._tools_anthropic if self._tools_anthropic else None,
            )
            duration = time.time() - start_time
            token_usage = {}
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                token_usage = {
                    "prompt_tokens": getattr(u, "input_tokens", 0),
                    "completion_tokens": getattr(u, "output_tokens", 0),
                    "total_tokens": getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0),
                    "cached_tokens": (getattr(u, "cache_creation_input_tokens", 0) or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0),
                }

            if self.tools:
                adapter = _anthropic_response_adapter(response)
                thinking_text = self._extract_thinking_from_response(adapter)
                log_llm_interaction(
                    interaction_type=f"anthropic_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "anthropic",
                        "has_image": True,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "anthropic"},
                )
                return adapter
            result = ""
            for block in (response.content or []):
                if isinstance(block, dict) and block.get("type") == "text":
                    result += block.get("text", "")
                elif getattr(block, "type", None) == "text":
                    result += getattr(block, "text", "") or ""
            log_llm_interaction(
                interaction_type=f"anthropic_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "anthropic", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "anthropic"},
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            err_msg = _anthropic_error_message(e)
            log_llm_error(
                interaction_type=f"anthropic_{module_name}",
                prompt=text,
                error=err_msg,
                metadata={"model": self.model_name, "backend": "anthropic", "duration": duration, "has_image": True},
            )
            logger.error("Anthropic API error: %s", err_msg)
            raise

    def get_text_query(self, text: str, module_name: str = "Unknown") -> Union[str, Any]:
        """Process text-only prompt. Returns adapter if tools, else string."""
        start_time = time.time()
        messages = [{"role": "user", "content": text}]

        try:
            response = self._call_messages(
                self.system_instruction,
                messages,
                tools=self._tools_anthropic if self._tools_anthropic else None,
            )
            duration = time.time() - start_time
            token_usage = {}
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                token_usage = {
                    "prompt_tokens": getattr(u, "input_tokens", 0),
                    "completion_tokens": getattr(u, "output_tokens", 0),
                    "total_tokens": getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0),
                    "cached_tokens": (getattr(u, "cache_creation_input_tokens", 0) or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0),
                }

            if self.tools:
                adapter = _anthropic_response_adapter(response)
                thinking_text = self._extract_thinking_from_response(adapter)
                log_llm_interaction(
                    interaction_type=f"anthropic_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "anthropic",
                        "has_image": False,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "anthropic"},
                )
                return adapter
            result = ""
            for block in (response.content or []):
                if isinstance(block, dict) and block.get("type") == "text":
                    result += block.get("text", "")
                elif getattr(block, "type", None) == "text":
                    result += getattr(block, "text", "") or ""
            log_llm_interaction(
                interaction_type=f"anthropic_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "anthropic", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "anthropic"},
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            err_msg = _anthropic_error_message(e)
            log_llm_error(
                interaction_type=f"anthropic_{module_name}",
                prompt=text,
                error=err_msg,
                metadata={"model": self.model_name, "backend": "anthropic", "duration": duration, "has_image": False},
            )
            logger.error("Anthropic API error: %s", err_msg)
            raise


def _anthropic_error_message(exc: Exception) -> str:
    """Build a detailed error message from an Anthropic API exception (including 400 body)."""
    msg = str(exc)
    # Prefer exc.body first (Anthropic SDK sets this to parsed JSON from the API)
    try:
        err_body = getattr(exc, "body", None)
        if err_body is not None:
            if isinstance(err_body, dict):
                msg = f"{msg} | API body: {json.dumps(err_body)}"
            else:
                msg = f"{msg} | API body: {err_body}"
    except Exception:
        pass
    # Fallback: raw response text
    if "API body:" not in msg:
        try:
            resp = getattr(exc, "response", None)
            if resp is not None:
                raw = getattr(resp, "text", None) or getattr(resp, "content", None)
                if raw is not None:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    if raw and raw.strip():
                        msg = f"{msg} | API body: {raw}"
        except Exception:
            pass
    return msg


def _openrouter_error_message(exc: Exception) -> str:
    """Build a detailed error message from OpenRouter/OpenAI API exception (including 400 body)."""
    msg = str(exc)
    try:
        err_body = getattr(exc, "body", None)
        if err_body is not None:
            if isinstance(err_body, dict):
                msg = f"{msg} | API body: {json.dumps(err_body)}"
            else:
                msg = f"{msg} | API body: {err_body}"
    except Exception:
        pass
    if "API body:" not in msg:
        try:
            resp = getattr(exc, "response", None)
            if resp is not None:
                raw = getattr(resp, "text", None) or getattr(resp, "content", None)
                if raw is not None:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    if raw and raw.strip():
                        msg = f"{msg} | API body: {raw}"
        except Exception:
            pass
    return msg


def _extract_openrouter_cached_tokens(usage) -> int:
    """Extract cached_tokens from OpenRouter usage.prompt_tokens_details.
    Handles both dict and object access (API may return either)."""
    if not usage or not hasattr(usage, "prompt_tokens_details"):
        return 0
    ptd = usage.prompt_tokens_details
    if not ptd:
        return 0
    if isinstance(ptd, dict):
        return ptd.get("cached_tokens", 0) or 0
    return getattr(ptd, "cached_tokens", 0) or 0


def _openrouter_response_adapter(response) -> Any:
    """Adapt OpenRouter/OpenAI Chat Completions response to Gemini-like structure for agents.

    Chat Completions returns choices[0].message with content (string) and optional tool_calls.
    Agents expect candidates[0].content.parts with .function_call / .text.
    """
    parts = []
    if not response or not getattr(response, "choices", None) or not response.choices:
        parts.append(_openai_text_part(""))
    else:
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if content and isinstance(content, str) and content.strip():
            parts.append(_openai_text_part(content))
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if not fn:
                continue
            name = getattr(fn, "name", None) or ""
            raw_args = getattr(fn, "arguments", None) or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            parts.append(_openai_tool_call_part(name, args))
        if not parts:
            parts.append(_openai_text_part(content or ""))

    content = type("Content", (), {"parts": parts})()
    candidate = type("Candidate", (), {"content": content})()
    adapter = type("ResponseAdapter", (), {"candidates": [candidate]})()
    adapter.usage_metadata = None
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        inp = getattr(usage, "prompt_tokens", 0)
        out = getattr(usage, "completion_tokens", 0)
        total = getattr(usage, "total_tokens", 0) or (inp + out)
        cached = _extract_openrouter_cached_tokens(usage)
        adapter.usage_metadata = type(
            "UsageMetadata",
            (),
            {
                "prompt_token_count": inp,
                "candidates_token_count": out,
                "total_token_count": total,
                "cached_content_token_count": cached,
            },
        )()
    return adapter


class OpenRouterBackend(VLMBackend):
    """OpenRouter API backend with tool calling and system instructions.

    Uses OpenAI Chat Completions–compatible API (base_url OpenRouter). Supports tools,
    system message, and vision. Returns adapter when tools configured, else string.
    """

    def __init__(self, model_name: str, tools: list = None, system_instruction: str = None, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")

        self.model_name = model_name
        self.tools = tools or []
        self.system_instruction = system_instruction
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("Error: OpenRouter API key is missing! Set OPENROUTER_API_KEY environment variable.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        if self.tools:
            self._tools_openrouter = self._convert_tools_to_openrouter_format()
            log_parts = [f"OpenRouter backend initialized with model: {model_name}", f"{len(self.tools)} tools"]
        else:
            self._tools_openrouter = []
            log_parts = [f"OpenRouter backend initialized with model: {model_name}"]
        if self.system_instruction:
            log_parts.append(f"system instructions ({len(self.system_instruction)} chars)")
        logger.info(", ".join(log_parts))

    def _setup_function_calling(self):
        """Update tools when agent dynamically updates tool list."""
        self._tools_openrouter = self._convert_tools_to_openrouter_format() if self.tools else []
        logger.info(f"OpenRouter model updated with {len(self.tools) if self.tools else 0} tools")

    def _build_json_schema_properties(self, params: dict) -> tuple:
        """Build JSON Schema properties from Gemini-style params. Returns (properties, required)."""
        properties = {}
        required = params.get("required", [])
        for prop_name, prop_def in params.get("properties", {}).items():
            t = prop_def.get("type_", "STRING")
            if t == "ARRAY":
                t = "array"
            elif t == "INTEGER":
                t = "integer"
            elif t == "BOOLEAN":
                t = "boolean"
            else:
                t = "string"
            prop = {"type": t, "description": prop_def.get("description", "")}
            if t == "array" and "items" in prop_def:
                items_t = prop_def["items"].get("type_", "STRING") if isinstance(prop_def["items"], dict) else "STRING"
                prop["items"] = {"type": "string" if items_t == "STRING" else "string"}
            if "enum" in prop_def:
                prop["enum"] = prop_def["enum"]
            properties[prop_name] = prop
        return properties, required

    def _convert_tools_to_openrouter_format(self) -> list:
        """Convert Gemini-style tool declarations to OpenAI Chat Completions format (nested function)."""
        result = []
        for tool in self.tools:
            params = tool.get("parameters", {})
            properties, required = self._build_json_schema_properties(params)
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": {"type": "object", "properties": properties, "required": required},
                },
            })
        return result

    def _prepare_image_base64(self, img: Union[Image.Image, np.ndarray]) -> str:
        """Prepare image as base64 string."""
        if hasattr(img, "convert"):
            image = img
        elif hasattr(img, "shape"):
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _extract_thinking_from_response(self, response) -> str:
        """Extract reasoning for logging from adapter (candidates[0].content.parts).
        
        Prioritizes function calls over text parts to ensure the command name is included,
        since some models return text explanations before tool calls.
        """
        if not response or not getattr(response, "candidates", None):
            return "[Executing function call]"
        candidate = response.candidates[0]
        content = getattr(candidate, "content", None)
        if not content or not getattr(content, "parts", None):
            return "[Executing function call]"
        
        # First pass: prioritize function calls to include command name in output
        for part in content.parts:
            fc = getattr(part, "function_call", None)
            if fc:
                args = getattr(fc, "args", {}) or {}
                reasoning = args.get("reasoning") or args.get("reason", "")
                if reasoning:
                    return f"[{fc.name}] {reasoning}"
                return f"Calling {fc.name}({list(args.keys())[:3]})"
        
        # Second pass: fall back to text parts if no function call found
        for part in content.parts:
            if getattr(part, "text", None):
                return part.text
        
        return "[Executing function call]"

    @retry_with_exponential_backoff
    def _call_completion(self, messages, tools=None):
        """Calls chat.completions with optional tools and system message."""
        kwargs = {"model": self.model_name, "messages": messages, "max_tokens": 8192}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def get_query(
        self,
        img: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        text: str,
        module_name: str = "Unknown",
    ) -> Union[str, Any]:
        """Process image(s) and text. Returns adapter if tools, else string."""
        start_time = time.time()

        if isinstance(img, list):
            content_parts = [{"type": "text", "text": text}]
            for i in img:
                image_base64 = self._prepare_image_base64(i)
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                )
        else:
            image_base64 = self._prepare_image_base64(img)
            content_parts = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ]

        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.append({"role": "user", "content": content_parts})

        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM IMAGE QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

        try:
            response = self._call_completion(
                messages,
                tools=self._tools_openrouter if self._tools_openrouter else None,
            )
            duration = time.time() - start_time

            token_usage = {}
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                token_usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                    "total_tokens": getattr(u, "total_tokens", 0) or (
                        getattr(u, "prompt_tokens", 0) + getattr(u, "completion_tokens", 0)
                    ),
                    "cached_tokens": _extract_openrouter_cached_tokens(u),
                }

            if self.tools:
                adapter = _openrouter_response_adapter(response)
                thinking_text = self._extract_thinking_from_response(adapter)
                log_llm_interaction(
                    interaction_type=f"openrouter_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "openrouter",
                        "has_image": True,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "openrouter"},
                )
                logger.info(f"[{module_name}] AGENT THINKING: {thinking_text}")
                logger.info(f"[{module_name}] ---")
                return adapter

            result = (getattr(response.choices[0].message, "content", None) or "") or ""
            log_llm_interaction(
                interaction_type=f"openrouter_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={
                    "model": self.model_name,
                    "backend": "openrouter",
                    "has_image": True,
                    "token_usage": token_usage,
                },
                model_info={"model": self.model_name, "backend": "openrouter"},
            )
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            return result
        except Exception as e:
            duration = time.time() - start_time
            err_msg = _openrouter_error_message(e)
            log_llm_error(
                interaction_type=f"openrouter_{module_name}",
                prompt=text,
                error=err_msg,
                metadata={"model": self.model_name, "backend": "openrouter", "duration": duration, "has_image": True},
            )
            logger.error("OpenRouter API error: %s", err_msg)
            raise

    def get_text_query(self, text: str, module_name: str = "Unknown") -> Union[str, Any]:
        """Process text-only prompt. Returns adapter if tools, else string."""
        start_time = time.time()

        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.append({"role": "user", "content": text})

        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM TEXT QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

        try:
            response = self._call_completion(
                messages,
                tools=self._tools_openrouter if self._tools_openrouter else None,
            )
            duration = time.time() - start_time

            token_usage = {}
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                token_usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                    "total_tokens": getattr(u, "total_tokens", 0) or (
                        getattr(u, "prompt_tokens", 0) + getattr(u, "completion_tokens", 0)
                    ),
                    "cached_tokens": _extract_openrouter_cached_tokens(u),
                }

            if self.tools:
                adapter = _openrouter_response_adapter(response)
                thinking_text = self._extract_thinking_from_response(adapter)
                log_llm_interaction(
                    interaction_type=f"openrouter_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "openrouter",
                        "has_image": False,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "openrouter"},
                )
                logger.info(f"[{module_name}] AGENT THINKING: {thinking_text}")
                logger.info(f"[{module_name}] ---")
                return adapter

            result = (getattr(response.choices[0].message, "content", None) or "") or ""
            log_llm_interaction(
                interaction_type=f"openrouter_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={
                    "model": self.model_name,
                    "backend": "openrouter",
                    "has_image": False,
                    "token_usage": token_usage,
                },
                model_info={"model": self.model_name, "backend": "openrouter"},
            )
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            return result
        except Exception as e:
            duration = time.time() - start_time
            err_msg = _openrouter_error_message(e)
            log_llm_error(
                interaction_type=f"openrouter_{module_name}",
                prompt=text,
                error=err_msg,
                metadata={"model": self.model_name, "backend": "openrouter", "duration": duration, "has_image": False},
            )
            logger.error("OpenRouter API error: %s", err_msg)
            raise


class LocalHuggingFaceBackend(VLMBackend):
    """Local HuggingFace transformers backend with bitsandbytes optimization"""

    def __init__(self, model_name: str, device: str = "auto", load_in_4bit: bool = False, **kwargs):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                f"Required packages not found. Install with: pip install torch transformers bitsandbytes accelerate. Error: {e}"
            )

        self.model_name = model_name
        self.device = device
        self.torch = torch

        logger.info(f"Loading local VLM model: {model_name}")

        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization with bitsandbytes")

        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device if device != "auto" else "auto",
                torch_dtype=torch.float16 if not load_in_4bit else None,
                trust_remote_code=True,
            )

            if device != "auto" and not load_in_4bit:
                self.model = self.model.to(device)

            logger.info(f"Model loaded successfully on {device}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _generate_response(self, inputs: Dict[str, Any], text: str, module_name: str) -> str:
        """Generate response using the local model"""
        try:
            start_time = time.time()

            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] LOCAL HF VLM QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

            with self.torch.no_grad():
                # Ensure all inputs are on the correct device
                if hasattr(self.model, "device"):
                    device = self.model.device
                elif hasattr(self.model, "module") and hasattr(self.model.module, "device"):
                    device = self.model.module.device
                else:
                    device = next(self.model.parameters()).device

                # Move inputs to device if needed
                inputs_on_device = {}
                for k, v in inputs.items():
                    if hasattr(v, "to"):
                        inputs_on_device[k] = v.to(device)
                    else:
                        inputs_on_device[k] = v

                generated_ids = self.model.generate(
                    **inputs_on_device,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

                # Decode the response
                generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)

                # Extract only the generated part (remove the prompt)
                if text in generated_text:
                    result = generated_text.split(text)[-1].strip()
                else:
                    result = generated_text.strip()

            # Log the interaction
            duration = time.time() - start_time
            log_llm_interaction(
                interaction_type=f"local_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "local", "has_image": "images" in inputs},
                model_info={"model": self.model_name, "backend": "local"},
            )

            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")

            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using local HuggingFace model"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, "convert"):  # It's a PIL Image
            image = img
        elif hasattr(img, "shape"):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # Prepare messages with proper chat template format
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        formatted_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted_text, images=image, return_tensors="pt")

        return self._generate_response(inputs, text, module_name)

    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using local HuggingFace model"""
        # For text-only queries, use simple text format without image
        messages = [{"role": "user", "content": text}]
        formatted_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted_text, return_tensors="pt")

        return self._generate_response(inputs, text, module_name)


class LegacyOllamaBackend(VLMBackend):
    """Legacy Ollama backend for backward compatibility"""

    def __init__(self, model_name: str, port: int = 8010, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")

        self.model_name = model_name
        self.port = port
        self.client = OpenAI(api_key="", base_url=f"http://localhost:{port}/v1")

    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(model=self.model_name, messages=messages)

    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using legacy Ollama backend"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, "convert"):  # It's a PIL Image
            image = img
        elif hasattr(img, "shape"):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ]

        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OLLAMA VLM IMAGE QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

        response = self._call_completion(messages)
        result = response.choices[0].message.content

        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")

        return result

    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using legacy Ollama backend"""
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OLLAMA VLM TEXT QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

        response = self._call_completion(messages)
        result = response.choices[0].message.content

        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")

        return result


class ThreadSafeGenerativeModelWrapper:
    """
    Thread-safe wrapper for GenerativeModel that protects _prediction_client access.

    This prevents race conditions when multiple threads try to access or refresh
    the gRPC client, especially after credential refresh (which happens after ~30 calls).

    Uses double-check locking pattern to ensure only one thread initializes the client,
    even if multiple threads pass the initial check simultaneously.

    Based on solution from: https://github.com/googleapis/python-aiplatform/issues/3365
    """

    # Class-level lock shared across all instances
    _prediction_client_lock = threading.Lock()

    def __init__(self, model):
        self._model = model
        self._prediction_client_value = None
        self._access_count = 0  # Track how many times client is accessed
        self._init_time = None  # Track when client was initialized

    @property
    def _prediction_client(self):
        """Thread-safe access to _prediction_client using double-check locking pattern"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        self._access_count += 1
        access_num = self._access_count

        logger.info(f"🔍 [_prediction_client] Access #{access_num} from thread {thread_id} ({thread_name})")

        # First check: Fast path if already initialized (no lock needed)
        if self._prediction_client_value is not None:
            logger.debug(f"   ✅ Fast path: client already initialized (init'd at {self._init_time})")
            return self._prediction_client_value

        logger.info(f"   ⚠️  Client not initialized, acquiring lock...")
        lock_start = time.time()

        # Acquire lock: Only one thread can initialize at a time
        with self._prediction_client_lock:
            lock_acquired_time = time.time() - lock_start
            if lock_acquired_time > 0.1:
                logger.warning(f"   ⏱️  Waited {lock_acquired_time:.3f}s to acquire lock (possible contention)")

            logger.info(f"   🔒 Lock acquired by thread {thread_id}")

            # Second check: Another thread might have initialized while we waited for the lock
            if self._prediction_client_value is None:
                logger.info(f"   🏗️  Initializing _prediction_client (first time or after refresh)")
                init_start = time.time()

                try:
                    # Access the underlying model's client (this may trigger credential refresh)
                    # This is the only place where initialization happens
                    self._prediction_client_value = self._model._prediction_client
                    init_duration = time.time() - init_start
                    self._init_time = time.time()

                    logger.info(f"   ✅ Client initialized in {init_duration:.3f}s")
                except Exception as e:
                    logger.error(f"   ❌ Failed to initialize client: {type(e).__name__}: {e}")
                    raise
            else:
                logger.info(f"   ℹ️  Another thread initialized client while we waited (double-check worked)")

        logger.info(f"   🔓 Lock released by thread {thread_id}")
        return self._prediction_client_value

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped model"""
        return getattr(self._model, name)

    def generate_content(self, *args, **kwargs):
        """Delegate generate_content calls to wrapped model with detailed logging"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name

        logger.info(f"📞 [generate_content] Called from thread {thread_id} ({thread_name})")
        logger.info(f"   Args: {len(args)} positional, {len(kwargs)} keyword")

        # Log detailed argument information
        if args:
            logger.debug(f"   Positional args types: {[type(a).__name__ for a in args]}")
        if kwargs:
            logger.debug(f"   Keyword args: {list(kwargs.keys())}")
            if "tools" in kwargs:
                tools = kwargs["tools"]
                logger.info(f"   🔧 Tools provided: {len(tools) if hasattr(tools, '__len__') else 'unknown'} tool(s)")
                if hasattr(tools, "__iter__") and len(list(tools)) > 0:
                    logger.debug(f"   First tool type: {type(list(tools)[0]).__name__}")
            if "generation_config" in kwargs:
                logger.debug(f"   Generation config: {type(kwargs['generation_config']).__name__}")

        gen_start = time.time()
        last_heartbeat = gen_start
        heartbeat_interval = 5.0  # Log heartbeat every 5 seconds

        # Create a flag to track if the call completed
        call_completed = threading.Event()
        call_exception = [None]  # Use list to allow modification in nested function

        def heartbeat_logger():
            """Background thread to log heartbeats during long-running calls"""
            while not call_completed.is_set():
                time.sleep(heartbeat_interval)  # Wait first, then check
                if not call_completed.is_set():  # Only log if still running
                    elapsed = time.time() - gen_start
                    logger.warning(f"   ⏳ Still waiting... {elapsed:.1f}s elapsed (thread {thread_id})")
                    # Log thread state
                    import sys

                    try:
                        frames = len(sys._current_frames())
                        logger.debug(f"   Thread stack: {frames} active frames")
                    except:
                        pass

        heartbeat_thread = None

        try:
            # Access _prediction_client before calling generate_content to ensure it's ready
            logger.info(f"   🔍 Ensuring _prediction_client is accessible...")
            client_access_start = time.time()
            _ = self._prediction_client
            client_access_duration = time.time() - client_access_start
            if client_access_duration > 0.1:
                logger.warning(f"   ⚠️  _prediction_client access took {client_access_duration:.3f}s")
            logger.info(f"   ✅ Client accessible, calling generate_content...")

            # Start heartbeat logger in background
            heartbeat_thread = threading.Thread(target=heartbeat_logger, daemon=True)
            heartbeat_thread.start()

            # Log the exact moment we enter the SDK call
            sdk_call_start = time.time()
            logger.info(f"   🚀 Entering SDK generate_content at {time.strftime('%H:%M:%S.%f')}")

            try:
                result = self._model.generate_content(*args, **kwargs)
            except Exception as sdk_exception:
                sdk_call_duration = time.time() - sdk_call_start
                logger.error(f"   💥 SDK generate_content raised exception after {sdk_call_duration:.3f}s")
                logger.error(f"   Exception type: {type(sdk_exception).__name__}")
                logger.error(f"   Exception message: {str(sdk_exception)}")

                # Check for specific gRPC errors
                if "grpc" in str(type(sdk_exception)).lower() or "rpc" in str(type(sdk_exception)).lower():
                    logger.error(f"   🚨 gRPC-related exception detected!")

                # Log full traceback
                import traceback

                tb_str = traceback.format_exc()
                logger.error(f"   Full SDK exception traceback:\n{tb_str}")

                call_exception[0] = sdk_exception
                raise

            sdk_call_duration = time.time() - sdk_call_start
            logger.info(f"   ✅ SDK generate_content returned after {sdk_call_duration:.3f}s")

            # Check result type
            if result is not None:
                logger.debug(f"   Result type: {type(result).__name__}")
                if hasattr(result, "candidates"):
                    logger.debug(
                        f"   Result has {len(result.candidates) if hasattr(result.candidates, '__len__') else 'unknown'} candidates"
                    )
                # Try to get text preview, but handle multiple parts gracefully
                try:
                    if hasattr(result, "text"):
                        result_preview = str(result.text)[:100] if result.text else "None"
                        logger.debug(f"   Result text preview: {result_preview}...")
                except ValueError as ve:
                    # Multiple content parts (text + function_call) - this is expected for function calling
                    logger.debug(f"   Result contains multiple parts (likely function call): {ve}")

            gen_duration = time.time() - gen_start
            logger.info(f"   ✅ generate_content completed in {gen_duration:.3f}s total")

            return result

        except KeyboardInterrupt:
            gen_duration = time.time() - gen_start
            logger.error(f"   ⛔ KeyboardInterrupt after {gen_duration:.3f}s")
            raise
        except Exception as e:
            gen_duration = time.time() - gen_start
            logger.error(f"   ❌ generate_content failed after {gen_duration:.3f}s: {type(e).__name__}: {e}")

            # Log additional context
            logger.error(f"   Thread: {thread_id} ({thread_name})")
            logger.error(f"   Args count: {len(args)} positional, {len(kwargs)} keyword")

            import traceback

            tb_str = traceback.format_exc()
            logger.error(f"   Full traceback:\n{tb_str}")

            # Check if this is a timeout or hang
            if gen_duration > 25.0:
                logger.error(f"   🚨 LONG-RUNNING CALL: This call took {gen_duration:.1f}s - possible hang!")

            raise
        finally:
            # Stop heartbeat logger
            call_completed.set()
            if heartbeat_thread and heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=1.0)


class VertexBackend(VLMBackend):
    """Google Gemini API with Vertex backend using vertexai.generative_models

    Supports dual mode operation:
    - Function Calling Mode: When tools are provided, returns GenerationResponse objects
      containing structured function calls that the agent can execute
    - Regular Text Mode: When no tools are provided, returns plain text responses
      like the original implementation

    This allows the same backend to work with both function-calling agents and
    traditional text-based agents.
    """

    def __init__(self, model_name: str, tools: list = None, system_instruction: str = None, **kwargs):
        try:
            import vertexai
            from vertexai.generative_models import FunctionDeclaration, GenerationConfig, GenerativeModel, Tool
        except ImportError:
            raise ImportError("Package vertexai not found. Install with: pip install google-cloud-aiplatform")

        self.model_name = model_name
        self.tools = tools or []
        self.system_instruction = system_instruction

        # Initialize VertexAI
        vertexai.init(project="pokeagent-011", location="us-central1")

        # Setup function calling if tools are provided
        if self.tools:
            self._setup_function_calling()

        # Create the base model WITH system instructions (but NOT tools - pass tools at call time)
        if self.system_instruction:
            base_model = GenerativeModel(model_name, system_instruction=[self.system_instruction])
            logger.info(
                f"Vertex backend initialized with model: {model_name} and system instructions ({len(self.system_instruction)} chars)"
            )
        else:
            base_model = GenerativeModel(model_name)
            logger.info(f"Vertex backend initialized with model: {model_name}")

        # Wrap the model with thread-safe _prediction_client access
        # This prevents race conditions when credentials refresh after many calls
        self.model = ThreadSafeGenerativeModelWrapper(base_model)

        # if self.tools:
        #     logger.info(f"Function calling enabled with {len(self.tools)} tools (will be passed at call time)")

        # Pre-initialize the client now (synchronously, before any threads)
        # This ensures the client is created in the main thread, avoiding initial race conditions
        try:
            _ = self.model._prediction_client
            logger.info("✅ Pre-initialized _prediction_client with thread-safe wrapper")
        except Exception as e:
            logger.warning(f"Could not pre-initialize _prediction_client: {e}")

    def _setup_function_calling(self):
        """Setup function calling for VertexAI using FunctionDeclaration and Tool objects"""
        try:
            from vertexai.generative_models import FunctionDeclaration, Tool

            # Convert tools to VertexAI FunctionDeclaration objects
            function_declarations = []
            for tool in self.tools:
                # Convert parameters format from Gemini to VertexAI
                parameters = self._convert_parameters_format(tool["parameters"])

                # Create FunctionDeclaration object
                function_declaration = FunctionDeclaration(
                    name=tool["name"], description=tool["description"], parameters=parameters
                )
                function_declarations.append(function_declaration)

            # Create Tool object with function declarations
            self._tools_vertex = [Tool(function_declarations=function_declarations)]

            logger.info(f"🔧 Configured function calling with {len(function_declarations)} functions")

        except Exception as e:
            logger.error(f"Failed to setup function calling: {e}")
            self._tools_vertex = []

    def _convert_parameters_format(self, gemini_params):
        """Convert Gemini tool parameters to VertexAI format"""
        # Convert Gemini's type_ format to standard JSON schema
        properties = {}
        required = []

        for prop_name, prop_def in gemini_params.get("properties", {}).items():
            # Convert Gemini type_ to standard type
            prop_type = prop_def.get("type_", "STRING")
            if prop_type == "ARRAY":
                prop_type = "array"
            elif prop_type == "INTEGER":
                prop_type = "integer"
            elif prop_type == "BOOLEAN":
                prop_type = "boolean"
            else:
                prop_type = "string"

            properties[prop_name] = {"type": prop_type, "description": prop_def.get("description", "")}

            # Handle array items
            if prop_type == "array" and "items" in prop_def:
                items_type = prop_def["items"].get("type_", "STRING")
                if items_type == "STRING":
                    items_type = "string"
                elif items_type == "INTEGER":
                    items_type = "integer"
                properties[prop_name]["items"] = {"type": items_type}

        # Get required fields
        required = gemini_params.get("required", [])

        return {"type": "object", "properties": properties, "required": required}

    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API - upscale to 4x resolution (HD)"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, "convert"):  # It's a PIL Image
            image = img
        elif hasattr(img, "shape"):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # Upscale to 4x resolution for better detail
        original_size = image.size  # (width, height)
        upscaled_size = (original_size[0] * 4, original_size[1] * 4)

        # Use LANCZOS for high-quality upscaling
        upscaled_image = image.resize(upscaled_size, Image.Resampling.LANCZOS)

        # Save debug copy to see what it looks like
        try:
            from pathlib import Path
            from utils.run_data_manager import get_cache_directory
            debug_dir = get_cache_directory()
            debug_path = debug_dir / "debug_upscaled_frame.png"
            upscaled_image.save(debug_path)
            logger.debug(f"Image upscaled: {original_size} → {upscaled_size} (4x) [saved to {debug_path}]")
        except Exception as e:
            logger.debug(f"Could not save debug image: {e}")
            logger.debug(f"Image upscaled: {original_size} → {upscaled_size} (4x)")

        return upscaled_image

    def _extract_thinking_from_response(self, response):
        """Extract thinking/reasoning text from response for logging

        Args:
            response: GenerationResponse object

        Returns:
            String containing extracted reasoning or function call info
        """
        thinking_text = ""

        # Try to extract reasoning from function calls
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                content = candidate.content
                if hasattr(content, "parts"):
                    for part in content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            function_call = part.function_call
                            # Extract reasoning from common argument names
                            if hasattr(function_call, "args"):
                                args = dict(function_call.args)
                                reasoning = args.get("reasoning") or args.get("reason") or ""
                                if reasoning:
                                    thinking_text = f"[{function_call.name}] {reasoning}"
                                else:
                                    # Show function call with args if no reasoning
                                    args_str = ", ".join(
                                        f"{k}={v}" for k, v in list(args.items())[:3]
                                    )  # Limit to first 3 args
                                    if len(args) > 3:
                                        args_str += ", ..."
                                    thinking_text = f"Calling {function_call.name}({args_str})"
                        elif hasattr(part, "text") and part.text:
                            # If there's also text content, use that
                            thinking_text = part.text

        # If no thinking text found, use a default
        if not thinking_text:
            thinking_text = "[Executing function call]"

        return thinking_text

    def _extract_text_from_response(self, response):
        """Extract text from VertexAI response, handling multiple parts

        Args:
            response: GenerationResponse object

        Returns:
            String containing all text parts concatenated, or empty string if no text found
        """
        text_parts = []

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                content = candidate.content
                if hasattr(content, "parts"):
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)

        # Join all text parts, or return empty string if none found
        return " ".join(text_parts) if text_parts else ""

    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method using the VertexAI SDK pattern."""
        from vertexai.generative_models import Content, Part, GenerationConfig
        import io

        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        logger.info(f"🚀 [_call_generate_content] Starting in thread {thread_id} ({thread_name})")

        # Build Part objects following the official VertexAI pattern
        parts = []
        part_start = time.time()
        for part in content_parts:
            if isinstance(part, str):
                parts.append(Part.from_text(part))
            elif hasattr(part, "mode"):  # PIL Image
                # Convert PIL Image to bytes for VertexAI
                img_byte_arr = io.BytesIO()
                part.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                parts.append(Part.from_data(img_byte_arr.read(), mime_type="image/png"))
            else:
                logger.warning(f"Unknown content part type: {type(part)}")

        part_duration = time.time() - part_start
        logger.info(f"   📦 Built {len(parts)} content parts in {part_duration:.3f}s")

        # Create user prompt Content object
        user_prompt_content = Content(role="user", parts=parts)

        # Add timeout logging and monitoring
        call_start_time = time.time()
        has_tools = hasattr(self, "_tools_vertex") and self._tools_vertex

        # logger.info(f"   🔧 has_tools={has_tools}, about to call generate_content...")

        # Log content details
        if user_prompt_content:
            logger.debug(
                f"   Content parts: {len(user_prompt_content.parts) if hasattr(user_prompt_content, 'parts') else 'unknown'}"
            )
            if hasattr(user_prompt_content, "parts") and user_prompt_content.parts:
                part_types = [type(p).__name__ for p in user_prompt_content.parts]
                logger.debug(f"   Part types: {part_types}")

        try:
            if has_tools:
                # Pass tools at call time (not at model creation to avoid gRPC race condition)
                # logger.info(f"   📞 Calling generate_content with function calling (tools passed at call time)")
                # logger.info(f"   ⏱️  Call started at {time.strftime('%H:%M:%S.%f')}")
                # logger.debug(f"   Tools type: {type(self._tools_vertex).__name__}")
                # logger.debug(f"   Generation config: temperature=0")

                try:
                    response = self.model.generate_content(
                        user_prompt_content,
                        generation_config=GenerationConfig(temperature=0),
                        tools=self._tools_vertex,  # Pass tools at call time
                    )
                except Exception as inner_e:
                    inner_duration = time.time() - call_start_time
                    logger.error(f"   💥 Inner generate_content exception after {inner_duration:.3f}s")
                    logger.error(f"   Exception: {type(inner_e).__name__}: {inner_e}")
                    import traceback

                    logger.error(f"   Inner traceback:\n{traceback.format_exc()}")
                    raise
            else:
                logger.info(f"   📞 Calling generate_content without function calling")
                logger.info(f"   ⏱️  Call started at {time.strftime('%H:%M:%S.%f')}")

                try:
                    response = self.model.generate_content(user_prompt_content)
                except Exception as inner_e:
                    inner_duration = time.time() - call_start_time
                    logger.error(f"   💥 Inner generate_content exception after {inner_duration:.3f}s")
                    logger.error(f"   Exception: {type(inner_e).__name__}: {inner_e}")
                    import traceback

                    logger.error(f"   Inner traceback:\n{traceback.format_exc()}")
                    raise

            call_duration = time.time() - call_start_time
            logger.info(f"   ✅ generate_content returned after {call_duration:.3f}s")

            # Log response details
            if response is not None:
                logger.debug(f"   Response type: {type(response).__name__}")
                if hasattr(response, "candidates"):
                    num_candidates = len(response.candidates) if hasattr(response.candidates, "__len__") else "unknown"
                    logger.debug(f"   Response has {num_candidates} candidate(s)")
                    if hasattr(response.candidates, "__iter__") and len(list(response.candidates)) > 0:
                        first_candidate = list(response.candidates)[0]
                        if hasattr(first_candidate, "finish_reason"):
                            logger.debug(f"   First candidate finish_reason: {first_candidate.finish_reason}")

            if call_duration > 5.0:  # Log slow calls
                logger.warning(f"⚠️  Slow generate_content call: {call_duration:.2f}s (has_tools={has_tools})")
            else:
                logger.debug(f"✅ generate_content completed in {call_duration:.2f}s")

            return response
        except KeyboardInterrupt:
            call_duration = time.time() - call_start_time
            logger.error(f"⛔ KeyboardInterrupt in _call_generate_content after {call_duration:.3f}s")
            raise
        except Exception as e:
            call_duration = time.time() - call_start_time
            logger.error(
                f"❌ generate_content failed after {call_duration:.2f}s (has_tools={has_tools}): {type(e).__name__}: {e}"
            )

            # Additional error context
            logger.error(f"   Thread: {threading.current_thread().ident} ({threading.current_thread().name})")
            logger.error(
                f"   Content parts: {len(user_prompt_content.parts) if hasattr(user_prompt_content, 'parts') else 'unknown'}"
            )

            # Check for gRPC-specific errors
            error_str = str(e).lower()
            if "grpc" in error_str or "rpc" in error_str or "deadline" in error_str or "timeout" in error_str:
                logger.error(f"   🚨 Network/gRPC-related error detected!")

            import traceback

            tb_str = traceback.format_exc()
            logger.error(f"   Full traceback:\n{tb_str}")

            # If it took a long time, this might be a hang
            if call_duration > 25.0:
                logger.error(f"   🚨 VERY LONG CALL: {call_duration:.1f}s - this might indicate a hang!")

            raise

    def get_query(
        self,
        img: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        text: str,
        module_name: str = "Unknown",
    ) -> str:
        """Process an image (or list of images) and text prompt using VertexAI

        Returns:
            - If tools are configured: Returns GenerationResponse object for function calling
            - If no tools: Returns string text response
        """
        try:
            start_time = time.time()

            # Handle list of images (video-like input)
            if isinstance(img, list):
                images = [self._prepare_image(i) for i in img]
                content_parts = [text] + images
                has_multiple_images = True
            else:
                image = self._prepare_image(img)
                content_parts = [text, image]
                has_multiple_images = False

            # Log the prompt
            prompt_preview = text
            logger.info(f"[{module_name}] VERTEX VLM {'VIDEO' if has_multiple_images else 'IMAGE'} QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

            # Generate response
            response = self._call_generate_content(content_parts)

            # Check for safety filter or content policy issues
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason == 12:
                    logger.warning(
                        f"[{module_name}] Vertex safety filter triggered (finish_reason=12). Trying text-only fallback."
                    )
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)

            duration = time.time() - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0),
                    "cached_tokens": getattr(usage, "cached_content_token_count", 0),
                }

            # DUAL MODE: Function calling vs Regular text response
            if self.tools:
                # Function calling mode: Extract reasoning for logging, return response object
                thinking_text = self._extract_thinking_from_response(response)

                # Log the interaction with extracted thinking
                log_llm_interaction(
                    interaction_type=f"vertex_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "vertex",
                        "has_image": True,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "vertex"},
                )

                # Log the full agent thinking (no truncation)
                logger.info(f"[{module_name}] AGENT THINKING: {thinking_text}")
                logger.info(f"[{module_name}] ---")

                # Return response object for function calling
                return response
            else:
                # Regular text mode: Extract and return text response
                # Use helper method to handle multiple parts
                result = self._extract_text_from_response(response)

                if not result:
                    # Fallback: try response.text if extraction fails
                    try:
                        result = response.text
                    except Exception as e:
                        logger.warning(f"[{module_name}] Could not extract text from response: {e}")
                        result = "I received a response but could not extract the text content."

                # Log the interaction
                log_llm_interaction(
                    interaction_type=f"vertex_{module_name}",
                    prompt=text,
                    response=result,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "vertex",
                        "has_image": True,
                        "token_usage": token_usage,
                    },
                    model_info={"model": self.model_name, "backend": "vertex"},
                )

                # Log the response
                result_preview = result[:1000] + "..." if len(result) > 1000 else result
                logger.info(f"[{module_name}] RESPONSE: {result_preview}")
                logger.info(f"[{module_name}] ---")

                return result

        except Exception as e:
            print(f"Error in Gemini image query: {e}")
            logger.error(f"Error in Gemini image query: {e}")
            # Try text-only fallback for any Gemini error
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e

    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using VertexAI

        Returns:
            String text response (function calling not supported in text-only mode)
        """
        try:
            start_time = time.time()
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] VERTEX VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

            # Generate response (without tools for text-only)
            response = self._call_generate_content([text])

            # Check for safety filter or content policy issues
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason == 12:
                    logger.warning(
                        f"[{module_name}] Vertex safety filter triggered (finish_reason=12). Returning default response."
                    )
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."

            # Use helper method to handle multiple parts
            result = self._extract_text_from_response(response)

            if not result:
                # Fallback: try response.text if extraction fails
                try:
                    result = response.text
                except Exception as e:
                    logger.warning(f"[{module_name}] Could not extract text from response: {e}")
                    result = "I received a response but could not extract the text content."

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0),
                    "cached_tokens": getattr(usage, "cached_content_token_count", 0),
                }

            # Log the interaction
            duration = time.time() - start_time
            log_llm_interaction(
                interaction_type=f"vertex_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={
                    "model": self.model_name,
                    "backend": "vertex",
                    "has_image": False,
                    "token_usage": token_usage,
                },
                model_info={"model": self.model_name, "backend": "vertex"},
            )

            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")

            return result

        except Exception as e:
            print(f"Error in Vertex text query: {e}")
            logger.error(f"Error in Vertex text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."


class GeminiBackend(VLMBackend):
    """Google Gemini API backend"""

    def __init__(self, model_name: str, tools: list = None, system_instruction: str = None, **kwargs):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai")

        self.model_name = model_name
        self.tools = tools or []
        self.system_instruction = system_instruction
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Error: Gemini API key is missing! Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Initialize the model WITH tools and system instructions if provided
        model_kwargs = {}
        if self.system_instruction:
            model_kwargs["system_instruction"] = self.system_instruction
        if self.tools:
            model_kwargs["tools"] = self.tools

        self.model = genai.GenerativeModel(model_name, **model_kwargs)

        # Log initialization details
        log_parts = [f"Gemini backend initialized with model: {model_name}"]
        if self.tools:
            log_parts.append(f"{len(self.tools)} tools")
        if self.system_instruction:
            log_parts.append(f"system instructions ({len(self.system_instruction)} chars)")
        logger.info(", ".join(log_parts))

        self.genai = genai

    def _setup_function_calling(self):
        """Update the model with current tools"""
        model_kwargs = {}
        if self.system_instruction:
            model_kwargs["system_instruction"] = self.system_instruction
        if self.tools:
            model_kwargs["tools"] = self.tools

        self.model = self.genai.GenerativeModel(self.model_name, **model_kwargs)
        logger.info(f"Gemini model updated with {len(self.tools) if self.tools else 0} tools")

    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API - upscale to 4x resolution (HD)"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, "convert"):  # It's a PIL Image
            image = img
        elif hasattr(img, "shape"):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # Upscale to 4x resolution for better detail
        original_size = image.size  # (width, height)
        upscaled_size = (original_size[0] * 4, original_size[1] * 4)

        # Use LANCZOS for high-quality upscaling
        image = image.resize(upscaled_size, Image.Resampling.LANCZOS)

        return image

    def _extract_text_from_response(self, response):
        """Extract text from Gemini response, handling multiple parts

        Args:
            response: GenerationResponse object

        Returns:
            String containing all text parts concatenated, or empty string if no text found
        """
        text_parts = []

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                content = candidate.content
                if hasattr(content, "parts"):
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)

        # Join all text parts, or return empty string if none found
        return " ".join(text_parts) if text_parts else ""

    def _extract_thinking_from_response(self, response):
        """Extract thinking/reasoning text from response for logging

        Args:
            response: GenerationResponse object

        Returns:
            String containing extracted reasoning or function call info
        """
        thinking_text = ""
        function_name = None

        # Try to extract reasoning from function calls
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                content = candidate.content
                if hasattr(content, "parts"):
                    for part in content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            function_call = part.function_call
                            function_name = function_call.name  # Always capture the function name

                            # Extract reasoning from common argument names
                            try:
                                args = dict(function_call.args) if hasattr(function_call, "args") else {}
                                reasoning = args.get("reasoning") or args.get("reason") or ""
                                if reasoning:
                                    thinking_text = f"[{function_call.name}] {reasoning}"
                                else:
                                    # Show function call with args if no reasoning
                                    args_str = ", ".join(
                                        f"{k}={v}" for k, v in list(args.items())[:3]
                                    )  # Limit to first 3 args
                                    if len(args) > 3:
                                        args_str += ", ..."
                                    thinking_text = f"Calling {function_call.name}({args_str})"
                            except Exception as e:
                                # If we can't extract args, at least show the function name
                                thinking_text = f"Calling {function_call.name}()"
                        elif hasattr(part, "text") and part.text:
                            # If there's also text content, use that
                            thinking_text = part.text

        # If no thinking text found, use function name if available, otherwise generic message
        if not thinking_text:
            if function_name:
                thinking_text = f"[Executing {function_name}]"
            else:
                thinking_text = "[Executing function call - unable to extract details]"

        return thinking_text

    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff for rate limits.

        Handles 429 rate limit errors with exponential backoff.
        Uses 180 second timeout for slow preview models like gemini-3-pro-preview.
        """

        # Use longer timeout for preview models which are much slower
        timeout = 180 if "preview" in self.model_name or "3-pro" in self.model_name else 60

        max_retries = 5
        base_delay = 2  # Start with 2 second delay

        for attempt in range(max_retries):
            try:
                # Configure tool calling to require function calls when tools are available
                generation_kwargs = {"request_options": {"timeout": timeout}}

                if self.tools:
                    # Force function calling mode - require the model to call a function
                    # Use dict format compatible with both Gemini API and VertexAI
                    generation_kwargs["tool_config"] = {
                        "function_calling_config": {
                            "mode": "ANY"  # Require function call (ANY, AUTO, or NONE)
                        }
                    }

                response = self.model.generate_content(content_parts, **generation_kwargs)
                return response
            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a rate limit error (429 or quota exceeded)
                if "429" in error_str or "quota" in error_str or "rate" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries")
                        raise
                else:
                    # Not a rate limit error, raise immediately
                    logger.error(f"Error in generate_content: {e}")
                    raise

        raise Exception("Max retries exceeded")

    def get_query(
        self,
        img: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        text: str,
        module_name: str = "Unknown",
    ) -> str:
        """Process an image (or list of images) and text prompt using Gemini API"""
        start_time = time.time()
        try:
            # Handle list of images (video-like input)
            if isinstance(img, list):
                images = [self._prepare_image(i) for i in img]
                content_parts = [text] + images
                has_multiple_images = True
            else:
                image = self._prepare_image(img)
                content_parts = [text, image]
                has_multiple_images = False

            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM {'VIDEO' if has_multiple_images else 'IMAGE'} QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            print(text)

            # Generate response
            response = self._call_generate_content(content_parts)

            # Check for safety filter or content policy issues
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason == 12:
                    logger.warning(
                        f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback."
                    )
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)

            duration = time.time() - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0),
                    "cached_tokens": getattr(usage, "cached_content_token_count", 0),
                }

            # DUAL MODE: Function calling vs Regular text response
            if self.tools:
                # Function calling mode: Extract reasoning for logging, return response object
                thinking_text = self._extract_thinking_from_response(response)

                # Log the interaction with extracted thinking
                log_llm_interaction(
                    interaction_type=f"gemini_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "gemini",
                        "has_image": True,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "gemini"},
                )

                # Log the full agent thinking (no truncation)
                logger.info(f"[{module_name}] AGENT THINKING: {thinking_text}")
                logger.info(f"[{module_name}] ---")

                # Debug logging for response structure
                logger.debug(f"[{module_name}] Response type: {type(response)}")
                logger.debug(f"[{module_name}] Has parts: {hasattr(response, 'parts')}")
                if hasattr(response, "parts"):
                    logger.debug(f"[{module_name}] Parts count: {len(response.parts) if response.parts else 0}")
                if hasattr(response, "candidates"):
                    logger.debug(
                        f"[{module_name}] Candidates count: {len(response.candidates) if response.candidates else 0}"
                    )

                # Return response object for function calling
                return response
            else:
                # Regular text mode: Extract and return text response
                # Use helper method to handle multiple parts
                result = self._extract_text_from_response(response)

                if not result:
                    # Fallback: try response.text if extraction fails
                    try:
                        result = response.text
                    except Exception as e:
                        logger.warning(f"[{module_name}] Could not extract text from response: {e}")
                        result = "I received a response but could not extract the text content."

                # Log the interaction
                log_llm_interaction(
                    interaction_type=f"gemini_{module_name}",
                    prompt=text,
                    response=result,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "gemini",
                        "has_image": True,
                        "token_usage": token_usage,
                    },
                    model_info={"model": self.model_name, "backend": "gemini"},
                )

                # Log the response
                result_preview = result[:1000] + "..." if len(result) > 1000 else result
                logger.info(f"[{module_name}] RESPONSE: {result_preview}")
                logger.info(f"[{module_name}] ---")

                return result

        except Exception as e:
            logger.error(f"Error in Gemini image query: {e}")
            # Try text-only fallback for any Gemini error
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e

    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using Gemini API"""
        start_time = time.time()
        try:
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

            # Generate response
            response = self._call_generate_content([text])

            # Check for safety filter or content policy issues
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason == 12:
                    logger.warning(
                        f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response."
                    )
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."

            duration = time.time() - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0),
                    "cached_tokens": getattr(usage, "cached_content_token_count", 0),
                }

            # DUAL MODE: Function calling vs Regular text response
            if self.tools:
                # Function calling mode: Extract reasoning for logging, return response object
                thinking_text = self._extract_thinking_from_response(response)

                # Log the interaction with extracted thinking
                log_llm_interaction(
                    interaction_type=f"gemini_{module_name}",
                    prompt=text,
                    response=thinking_text,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "gemini",
                        "has_image": False,
                        "token_usage": token_usage,
                        "has_function_call": True,
                    },
                    model_info={"model": self.model_name, "backend": "gemini"},
                )

                # Log the response preview
                thinking_preview = thinking_text
                logger.info(f"[{module_name}] AGENT THINKING: {thinking_preview}")
                logger.info(f"[{module_name}] ---")

                # Return response object for function calling
                return response
            else:
                # Regular text mode: Extract and return text response
                # Use helper method to handle multiple parts
                result = self._extract_text_from_response(response)

                if not result:
                    # Fallback: try response.text if extraction fails
                    try:
                        result = response.text
                    except Exception as e:
                        logger.warning(f"[{module_name}] Could not extract text from response: {e}")
                        result = "I received a response but could not extract the text content."

                # Log the interaction
                log_llm_interaction(
                    interaction_type=f"gemini_{module_name}",
                    prompt=text,
                    response=result,
                    duration=duration,
                    metadata={
                        "model": self.model_name,
                        "backend": "gemini",
                        "has_image": False,
                        "token_usage": token_usage,
                    },
                    model_info={"model": self.model_name, "backend": "gemini"},
                )

                # Log the response
                result_preview = result[:1000] + "..." if len(result) > 1000 else result
                logger.info(f"[{module_name}] RESPONSE: {result_preview}")
                logger.info(f"[{module_name}] ---")

                return result

        except Exception as e:
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."


class VLM:
    """Main VLM class that supports multiple backends"""

    BACKENDS = {
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "openrouter": OpenRouterBackend,
        "local": LocalHuggingFaceBackend,
        "gemini": GeminiBackend,
        "ollama": LegacyOllamaBackend,  # Legacy support
        "vertex": VertexBackend,  # Added Vertex backend
    }

    def __init__(
        self,
        model_name: str,
        backend: str = "openai",
        port: int = 8010,
        tools: list = None,
        system_instruction: str = None,
        **kwargs,
    ):
        """
        Initialize VLM with specified backend

        Args:
            model_name: Name of the model to use
            backend: Backend type ('openai', 'anthropic', 'openrouter', 'local', 'gemini', 'ollama', 'vertex')
            port: Port for Ollama backend (legacy)
            tools: List of tool declarations for function calling
            system_instruction: System instructions for the model (supported by vertex, gemini)
            **kwargs: Additional arguments passed to backend
        """
        self.model_name = model_name
        self.backend_type = backend.lower()
        self.tools = tools or []
        self.system_instruction = system_instruction

        # Auto-detect backend based on model name if not explicitly specified
        if backend == "auto":
            self.backend_type = self._auto_detect_backend(model_name)

        if self.backend_type not in self.BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend_type}. Available: {list(self.BACKENDS.keys())}")

        # Initialize the appropriate backend
        backend_class = self.BACKENDS[self.backend_type]

        # Pass port parameter for legacy Ollama backend
        if self.backend_type == "ollama":
            self.backend = backend_class(model_name, port=port, **kwargs)
        else:
            # Pass tools and system_instruction to backends that support function calling
            if self.backend_type in ["vertex", "gemini", "openai", "anthropic", "openrouter"]:
                self.backend = backend_class(
                    model_name, tools=self.tools, system_instruction=self.system_instruction, **kwargs
                )
            else:
                self.backend = backend_class(model_name, **kwargs)

        logger.info(f"VLM initialized with {self.backend_type} backend using model: {model_name}")

    def _auto_detect_backend(self, model_name: str) -> str:
        """Auto-detect backend based on model name"""
        model_lower = model_name.lower()
        # Native Anthropic model ids (e.g. claude-sonnet-4-5) have no slash; OpenRouter uses "anthropic/claude-..."
        if model_lower.startswith("claude-") and "/" not in model_name:
            return "anthropic"
        if any(x in model_lower for x in ["gpt", "o4-mini", "o3", "claude"]):
            return "openai"
        elif any(x in model_lower for x in ["gemini", "palm"]):
            return "gemini"
        elif any(x in model_lower for x in ["llama", "mistral", "qwen", "phi"]):
            return "local"
        else:
            return "openai"

    def get_query(
        self,
        img: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
        text: str,
        module_name: str = "Unknown",
    ) -> Any:
        """Process an image (or list of images) and text prompt"""
        try:
            # Backend handles its own logging, so we don't duplicate it here
            result = self.backend.get_query(img, text, module_name)
            return result
        except Exception as e:
            # Only log errors that aren't already logged by the backend
            duration = 0  # Backend tracks actual duration
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={
                    "model": self.model_name,
                    "backend": self.backend.__class__.__name__,
                    "duration": duration,
                    "has_image": True,
                },
            )
            raise

    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt"""
        try:
            # Backend handles its own logging, so we don't duplicate it here
            result = self.backend.get_text_query(text, module_name)
            return result
        except Exception as e:
            # Only log errors that aren't already logged by the backend
            duration = 0  # Backend tracks actual duration
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={
                    "model": self.model_name,
                    "backend": self.backend.__class__.__name__,
                    "duration": duration,
                    "has_image": False,
                },
            )
            raise
