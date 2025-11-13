from io import BytesIO
from PIL import Image
import os
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
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt"""
        pass
    
    @abstractmethod
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt"""
        pass

class OpenAIBackend(VLMBackend):
    """OpenAI API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: OpenAI API key is missing! Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.errors = (openai.RateLimitError,)
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using OpenAI API"""
        start_time = time.time()
        
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
        
        try:
            response = self._call_completion(messages)
            result = response.choices[0].message.content
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "openai", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "openai"}
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": True}
            )
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using OpenAI API"""
        start_time = time.time()
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
        try:
            response = self._call_completion(messages)
            result = response.choices[0].message.content
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "openai", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "openai"}
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": False}
            )
            logger.error(f"OpenAI API error: {e}")
            raise

class OpenRouterBackend(VLMBackend):
    """OpenRouter API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: OpenRouter API key is missing! Set OPENROUTER_API_KEY environment variable.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using OpenRouter API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM IMAGE QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using OpenRouter API"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM TEXT QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result

class LocalHuggingFaceBackend(VLMBackend):
    """Local HuggingFace transformers backend with bitsandbytes optimization"""
    
    def __init__(self, model_name: str, device: str = "auto", load_in_4bit: bool = False, **kwargs):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
            from PIL import Image
        except ImportError as e:
            raise ImportError(f"Required packages not found. Install with: pip install torch transformers bitsandbytes accelerate. Error: {e}")
        
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
                bnb_4bit_quant_type="nf4"
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
                trust_remote_code=True
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
                if hasattr(self.model, 'device'):
                    device = self.model.device
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'device'):
                    device = self.model.module.device
                else:
                    device = next(self.model.parameters()).device
                
                # Move inputs to device if needed
                inputs_on_device = {}
                for k, v in inputs.items():
                    if hasattr(v, 'to'):
                        inputs_on_device[k] = v.to(device)
                    else:
                        inputs_on_device[k] = v
                
                generated_ids = self.model.generate(
                    **inputs_on_device,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
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
                model_info={"model": self.model_name, "backend": "local"}
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
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Prepare messages with proper chat template format
        messages = [
            {"role": "user",
             "content": [
                 {"type": "image", "image": image},
                 {"type": "text", "text": text}
             ]}
        ]
        formatted_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted_text, images=image, return_tensors="pt")
        
        return self._generate_response(inputs, text, module_name)
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using local HuggingFace model"""
        # For text-only queries, use simple text format without image
        messages = [
            {"role": "user", "content": text}
        ]
        formatted_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
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
        self.client = OpenAI(api_key='', base_url=f'http://localhost:{port}/v1')
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using legacy Ollama backend"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }]
        
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
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
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
            if 'tools' in kwargs:
                tools = kwargs['tools']
                logger.info(f"   🔧 Tools provided: {len(tools) if hasattr(tools, '__len__') else 'unknown'} tool(s)")
                if hasattr(tools, '__iter__') and len(list(tools)) > 0:
                    logger.debug(f"   First tool type: {type(list(tools)[0]).__name__}")
            if 'generation_config' in kwargs:
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
                if 'grpc' in str(type(sdk_exception)).lower() or 'rpc' in str(type(sdk_exception)).lower():
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
                if hasattr(result, 'candidates'):
                    logger.debug(f"   Result has {len(result.candidates) if hasattr(result.candidates, '__len__') else 'unknown'} candidates")
                if hasattr(result, 'text'):
                    result_preview = str(result.text)[:100] if result.text else "None"
                    logger.debug(f"   Result text preview: {result_preview}...")
            
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
            from vertexai.generative_models import (
                FunctionDeclaration,
                GenerationConfig,
                GenerativeModel,
                Tool
            )
        except ImportError:
            raise ImportError("Package vertexai not found. Install with: pip install google-cloud-aiplatform")
        
        self.model_name = model_name
        self.tools = tools or []
        self.system_instruction = system_instruction
        
        # Initialize VertexAI
        vertexai.init(project='pokeagent-011', location='us-central1')
        
        # Setup function calling if tools are provided
        if self.tools:
            self._setup_function_calling()
        
        # Create the base model WITH system instructions (but NOT tools - pass tools at call time)
        if self.system_instruction:
            base_model = GenerativeModel(model_name, system_instruction=[self.system_instruction])
            logger.info(f"Vertex backend initialized with model: {model_name} and system instructions ({len(self.system_instruction)} chars)")
        else:
            base_model = GenerativeModel(model_name)
            logger.info(f"Vertex backend initialized with model: {model_name}")
        
        # Wrap the model with thread-safe _prediction_client access
        # This prevents race conditions when credentials refresh after many calls
        self.model = ThreadSafeGenerativeModelWrapper(base_model)
        
        if self.tools:
            logger.info(f"Function calling enabled with {len(self.tools)} tools (will be passed at call time)")
        
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
                parameters = self._convert_parameters_format(tool['parameters'])
                
                # Create FunctionDeclaration object
                function_declaration = FunctionDeclaration(
                    name=tool['name'],
                    description=tool['description'],
                    parameters=parameters
                )
                function_declarations.append(function_declaration)
            
            # Create Tool object with function declarations
            self.tools_for_vertex = [Tool(function_declarations=function_declarations)]
            
            logger.info(f"🔧 Configured function calling with {len(function_declarations)} functions")
            
        except Exception as e:
            logger.error(f"Failed to setup function calling: {e}")
            self.tools_for_vertex = []
    
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
            
            properties[prop_name] = {
                "type": prop_type,
                "description": prop_def.get("description", "")
            }
            
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
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    def _extract_thinking_from_response(self, response):
        """Extract thinking/reasoning text from response for logging
        
        Args:
            response: GenerationResponse object
            
        Returns:
            String containing extracted reasoning or function call info
        """
        thinking_text = ""
        
        # Try to extract reasoning from function calls
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            # Extract reasoning from common argument names
                            if hasattr(function_call, 'args'):
                                args = dict(function_call.args)
                                reasoning = args.get('reasoning') or args.get('reason') or ''
                                if reasoning:
                                    thinking_text = f"[{function_call.name}] {reasoning}"
                                else:
                                    # Show function call with args if no reasoning
                                    args_str = ', '.join(f'{k}={v}' for k, v in list(args.items())[:3])  # Limit to first 3 args
                                    if len(args) > 3:
                                        args_str += ', ...'
                                    thinking_text = f"Calling {function_call.name}({args_str})"
                        elif hasattr(part, 'text') and part.text:
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
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
        
        # Join all text parts, or return empty string if none found
        return ' '.join(text_parts) if text_parts else ""
    
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
            elif hasattr(part, 'mode'):  # PIL Image
                # Convert PIL Image to bytes for VertexAI
                img_byte_arr = io.BytesIO()
                part.save(img_byte_arr, format='PNG')
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
        has_tools = hasattr(self, 'tools_for_vertex') and self.tools_for_vertex
        
        logger.info(f"   🔧 has_tools={has_tools}, about to call generate_content...")
        
        # Log content details
        if user_prompt_content:
            logger.debug(f"   Content parts: {len(user_prompt_content.parts) if hasattr(user_prompt_content, 'parts') else 'unknown'}")
            if hasattr(user_prompt_content, 'parts') and user_prompt_content.parts:
                part_types = [type(p).__name__ for p in user_prompt_content.parts]
                logger.debug(f"   Part types: {part_types}")
        
        try:
            if has_tools:
                # Pass tools at call time (not at model creation to avoid gRPC race condition)
                logger.info(f"   📞 Calling generate_content with function calling (tools passed at call time)")
                logger.info(f"   ⏱️  Call started at {time.strftime('%H:%M:%S.%f')}")
                logger.debug(f"   Tools type: {type(self.tools_for_vertex).__name__}")
                logger.debug(f"   Generation config: temperature=0")
                
                try:
                    response = self.model.generate_content(
                        user_prompt_content,
                        generation_config=GenerationConfig(temperature=0),
                        tools=self.tools_for_vertex  # Pass tools at call time
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
                if hasattr(response, 'candidates'):
                    num_candidates = len(response.candidates) if hasattr(response.candidates, '__len__') else 'unknown'
                    logger.debug(f"   Response has {num_candidates} candidate(s)")
                    if hasattr(response.candidates, '__iter__') and len(list(response.candidates)) > 0:
                        first_candidate = list(response.candidates)[0]
                        if hasattr(first_candidate, 'finish_reason'):
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
            logger.error(f"❌ generate_content failed after {call_duration:.2f}s (has_tools={has_tools}): {type(e).__name__}: {e}")
            
            # Additional error context
            logger.error(f"   Thread: {threading.current_thread().ident} ({threading.current_thread().name})")
            logger.error(f"   Content parts: {len(user_prompt_content.parts) if hasattr(user_prompt_content, 'parts') else 'unknown'}")
            
            # Check for gRPC-specific errors
            error_str = str(e).lower()
            if 'grpc' in error_str or 'rpc' in error_str or 'deadline' in error_str or 'timeout' in error_str:
                logger.error(f"   🚨 Network/gRPC-related error detected!")
            
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"   Full traceback:\n{tb_str}")
            
            # If it took a long time, this might be a hang
            if call_duration > 25.0:
                logger.error(f"   🚨 VERY LONG CALL: {call_duration:.1f}s - this might indicate a hang!")
            
            raise
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using VertexAI
        
        Returns:
            - If tools are configured: Returns GenerationResponse object for function calling
            - If no tools: Returns string text response
        """
        try:
            start_time = time.time()
            image = self._prepare_image(img)
            
            # Prepare content for VertexAI
            content_parts = [text, image]
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] VERTEX VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Vertex safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)
            
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
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
                    metadata={"model": self.model_name, "backend": "vertex", "has_image": True, "token_usage": token_usage, "has_function_call": True},
                    model_info={"model": self.model_name, "backend": "vertex"}
                )
                
                # Log the response preview
                thinking_preview = thinking_text[:200] + "..." if len(thinking_text) > 200 else thinking_text
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
                    interaction_type=f"vertex_{module_name}",
                    prompt=text,
                    response=result,
                    duration=duration,
                    metadata={"model": self.model_name, "backend": "vertex", "has_image": True, "token_usage": token_usage},
                    model_info={"model": self.model_name, "backend": "vertex"}
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
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Vertex safety filter triggered (finish_reason=12). Returning default response.")
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
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            duration = time.time() - start_time
            log_llm_interaction(
                interaction_type=f"vertex_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "vertex", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "vertex"}
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
            raise ValueError("Error: Gemini API key is missing! Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model WITH system instructions if provided
        if self.system_instruction:
            self.model = genai.GenerativeModel(model_name, system_instruction=self.system_instruction)
            logger.info(f"Gemini backend initialized with model: {model_name} and system instructions ({len(self.system_instruction)} chars)")
        else:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Gemini backend initialized with model: {model_name}")
        
        self.genai = genai
    
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        response = self.model.generate_content(content_parts)
        response.resolve()
        return response
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using Gemini API"""
        start_time = time.time()
        try:
            image = self._prepare_image(img)
            
            # Prepare content for Gemini
            content_parts = [text, image]
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)
            
            result = response.text
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "gemini"}
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
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."
            
            result = response.text
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "gemini"}
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
        'openai': OpenAIBackend,
        'openrouter': OpenRouterBackend,
        'local': LocalHuggingFaceBackend,
        'gemini': GeminiBackend,
        'ollama': LegacyOllamaBackend,  # Legacy support
        'vertex': VertexBackend,  # Added Vertex backend
    }
    
    def __init__(self, model_name: str, backend: str = 'openai', port: int = 8010, tools: list = None, system_instruction: str = None, **kwargs):
        """
        Initialize VLM with specified backend
        
        Args:
            model_name: Name of the model to use
            backend: Backend type ('openai', 'openrouter', 'local', 'gemini', 'ollama', 'vertex')
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
        if backend == 'auto':
            self.backend_type = self._auto_detect_backend(model_name)
        
        if self.backend_type not in self.BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend_type}. Available: {list(self.BACKENDS.keys())}")
        
        # Initialize the appropriate backend
        backend_class = self.BACKENDS[self.backend_type]
        
        # Pass port parameter for legacy Ollama backend
        if self.backend_type == 'ollama':
            self.backend = backend_class(model_name, port=port, **kwargs)
        else:
            # Pass tools and system_instruction to backends that support function calling
            if self.backend_type in ['vertex', 'gemini']:
                self.backend = backend_class(model_name, tools=self.tools, system_instruction=self.system_instruction, **kwargs)
            else:
                self.backend = backend_class(model_name, **kwargs)
        
        logger.info(f"VLM initialized with {self.backend_type} backend using model: {model_name}")
    
    def _auto_detect_backend(self, model_name: str) -> str:
        """Auto-detect backend based on model name"""
        model_lower = model_name.lower()
        
        if any(x in model_lower for x in ['gpt', 'o4-mini', 'o3', 'claude']):
            return 'openai'
        elif any(x in model_lower for x in ['gemini', 'palm']):
            return 'gemini'
        elif any(x in model_lower for x in ['llama', 'mistral', 'qwen', 'phi']):
            return 'local'
        else:
            # Default to OpenAI for unknown models
            return 'openai'
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt"""
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
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": True}
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
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": False}
            )
            raise