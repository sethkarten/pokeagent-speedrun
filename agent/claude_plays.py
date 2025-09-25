"""
ClaudePlaysPokemon Agent for Pokemon Emerald
=============================================

Faithful adaptation of David Hershey's ClaudePlaysPokemonStarter implementation.
Uses Anthropic-style tool-based interaction with explicit button press sequences.

Original source: https://github.com/davidhershey/ClaudePlaysPokemonStarter

This implementation preserves the original's approach:
- Tool-based interaction (press_buttons, navigate_to)  
- Automatic history summarization when context gets too long
- Maintains message history in Anthropic format
- Supports multi-step button sequences with queuing
"""

import json
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from io import BytesIO
from PIL import Image

from utils.vlm import VLMClient
from utils.llm_logger import LLMLogger
from utils.state_formatter import format_state_for_llm
from utils.pathfinding import Pathfinder

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call parsed from agent response."""
    name: str
    parameters: Dict[str, Any]


class ClaudePlaysAgent:
    """
    Faithful implementation of ClaudePlaysPokemon agent.
    
    Preserves the original's core features:
    - Tool-based control (press_buttons, navigate_to)
    - Message history management with summarization
    - Anthropic-style conversation format
    - Button sequence queuing
    """
    
    # Default configuration matching original
    DEFAULT_MAX_HISTORY = 30  # Messages before summarization (original default)
    
    def __init__(
        self,
        vlm_client: Optional[VLMClient] = None,
        max_history: int = 30,
        enable_navigation: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the ClaudePlaysPokemon agent.
        
        Args:
            vlm_client: Vision-language model client for LLM queries
            max_history: Maximum message history before summarization (default 30 from original)
            enable_navigation: Whether to enable navigate_to tool
            verbose: Whether to print detailed action logs
        """
        self.vlm_client = vlm_client or VLMClient()
        self.max_history = max_history
        self.enable_navigation = enable_navigation
        self.verbose = verbose
        
        # Message history in Anthropic format
        self.messages: List[Dict[str, Any]] = []
        
        # Button queue for multi-step sequences
        self.button_queue: List[str] = []
        
        # Step tracking
        self.step_count = 0
        self.running = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.llm_logger = LLMLogger()
        
        # Pathfinding for navigation
        self.pathfinder = Pathfinder()
    
    def get_system_prompt(self) -> str:
        """Get the system prompt (matching original's structure)."""
        return """You are playing Pokemon Emerald on a Game Boy Advance emulator.

Your goal is to progress through the game, defeat gym leaders, and ultimately become the Pokemon Champion.

You interact with the game by using tools to press buttons on the emulator. Each button press is executed sequentially.

Sometimes, you may receive a summary of what happened recently instead of a full history.

Be strategic in your decisions - consider type advantages in battles, manage your Pokemon's health, and explore thoroughly to find items and trainers."""
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions in Anthropic format."""
        tools = [
            {
                "name": "press_buttons",
                "description": "Press a sequence of Game Boy Advance buttons",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "buttons": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", 
                                       "START", "SELECT", "L", "R"]
                            },
                            "description": "List of buttons to press in sequence"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Explanation for why you're pressing these buttons"
                        }
                    },
                    "required": ["buttons"]
                }
            }
        ]
        
        if self.enable_navigation:
            tools.append({
                "name": "navigate_to",
                "description": "Navigate to a specific coordinate using A* pathfinding with collision detection. Automatically finds optimal path around obstacles, NPCs, and walls.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "integer",
                            "description": "Target X coordinate on the map"
                        },
                        "y": {
                            "type": "integer",
                            "description": "Target Y coordinate on the map"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why navigate to this location"
                        }
                    },
                    "required": ["x", "y"]
                }
            })
        
        return tools
    
    def step(self, state: Dict[str, Any], screenshot: Any = None) -> str:
        """
        Execute one step of the agent (compatible with run.py).
        
        Args:
            state: Current game state from memory reader
            screenshot: Current game screenshot (PIL Image)
            
        Returns:
            Single button command (e.g., "A", "UP", "NONE")
        """
        self.step_count += 1
        
        # If we have queued buttons from previous tool call, return next one
        if self.button_queue:
            button = self.button_queue.pop(0)
            if self.verbose:
                print(f"  Executing queued button: {button}")
            return button
        
        # Check if we need to summarize history (matching original behavior)
        if len(self.messages) > self.max_history:
            self._summarize_history()
        
        # Create user message with screenshot and state
        user_message = self._create_user_message(state, screenshot)
        self.messages.append(user_message)
        
        # Query the VLM for next action
        assistant_response = self._query_vlm(screenshot)
        self.messages.append({"role": "assistant", "content": assistant_response})
        
        # Parse and process tool calls
        tool_calls = self._parse_tool_calls(assistant_response)
        
        if tool_calls:
            for tool_call in tool_calls:
                self._process_tool_call(tool_call, state)
        
        # Return next button or NONE
        return self.button_queue.pop(0) if self.button_queue else "NONE"
    
    def _create_user_message(self, state: Dict[str, Any], screenshot: Any) -> Dict[str, Any]:
        """Create a user message in Anthropic format."""
        content = []
        
        # Add state description
        state_text = format_state_for_llm(state)
        content.append({
            "type": "text",
            "text": f"Current game state:\n{state_text}\n\nWhat should we do next?"
        })
        
        # Add screenshot if available
        if screenshot is not None:
            if isinstance(screenshot, Image.Image):
                # Convert PIL Image to base64
                buffered = BytesIO()
                screenshot.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64
                    }
                })
        
        return {"role": "user", "content": content}
    
    def _query_vlm(self, screenshot: Any) -> str:
        """Query the VLM with current context."""
        # Build prompt with tools
        tools_text = self._format_tools_text()
        
        # Build conversation context
        context = self._build_context()
        
        prompt = f"""{self.get_system_prompt()}

{tools_text}

{context}

Analyze the current game state and decide what action to take next. Use the press_buttons tool to control the game."""
        
        # Query VLM
        response = self.vlm_client.query(
            prompt=prompt,
            image=screenshot,
            max_tokens=1024,
            temperature=0.0  # Matching original's deterministic approach
        )
        
        # Log the interaction
        self.llm_logger.log_llm_interaction(
            module="claude_plays",
            prompt=prompt,
            response=response
        )
        
        return response
    
    def _format_tools_text(self) -> str:
        """Format tools for inclusion in prompt."""
        tools = self.get_tools()
        lines = ["Available tools:"]
        
        for tool in tools:
            lines.append(f"\n{tool['name']}:")
            lines.append(f"  {tool['description']}")
            if "input_schema" in tool and "properties" in tool["input_schema"]:
                lines.append("  Parameters:")
                for param, details in tool["input_schema"]["properties"].items():
                    req = " (required)" if param in tool["input_schema"].get("required", []) else ""
                    lines.append(f"    - {param}: {details.get('description', 'No description')}{req}")
        
        lines.append("\nTo use a tool, format your response as:")
        lines.append("<use_tool>")
        lines.append('{"tool": "tool_name", "parameters": {...}}')
        lines.append("</use_tool>")
        lines.append("\nYou can explain your reasoning before or after the tool use.")
        
        return "\n".join(lines)
    
    def _build_context(self) -> str:
        """Build conversation context from message history."""
        if not self.messages:
            return "This is the start of the game."
        
        # Take recent messages for context (not full history to save tokens)
        recent = self.messages[-10:]  # Last 10 messages
        
        context_lines = ["Recent conversation:"]
        for msg in recent:
            if msg["role"] == "user":
                if isinstance(msg["content"], list):
                    # Extract text from content
                    text_parts = [c["text"] for c in msg["content"] if c["type"] == "text"]
                    if text_parts:
                        context_lines.append(f"User: {text_parts[0][:200]}...")
                else:
                    context_lines.append(f"User: {msg['content'][:200]}...")
            elif msg["role"] == "assistant":
                # Show tool uses from assistant
                content = msg["content"]
                if "<use_tool>" in content:
                    # Extract tool use
                    start = content.find("<use_tool>")
                    end = content.find("</use_tool>") + len("</use_tool>")
                    tool_text = content[start:end]
                    context_lines.append(f"Assistant: {tool_text}")
                else:
                    context_lines.append(f"Assistant: {content[:200]}...")
        
        return "\n".join(context_lines)
    
    def _parse_tool_calls(self, response: str) -> List[ToolCall]:
        """Parse tool calls from assistant response."""
        tool_calls = []
        
        # Look for <use_tool> blocks
        start_tag = "<use_tool>"
        end_tag = "</use_tool>"
        
        pos = 0
        while start_tag in response[pos:]:
            start = response.index(start_tag, pos) + len(start_tag)
            if end_tag in response[start:]:
                end = response.index(end_tag, start)
                tool_json = response[start:end].strip()
                
                try:
                    tool_data = json.loads(tool_json)
                    tool_name = tool_data.get("tool")
                    parameters = tool_data.get("parameters", {})
                    
                    if tool_name:
                        tool_calls.append(ToolCall(name=tool_name, parameters=parameters))
                        
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse tool JSON: {tool_json}")
                
                pos = end + len(end_tag)
            else:
                break
        
        return tool_calls
    
    def _process_tool_call(self, tool_call: ToolCall, state: Dict[str, Any]) -> None:
        """Process a tool call and update button queue."""
        if tool_call.name == "press_buttons":
            buttons = tool_call.parameters.get("buttons", [])
            reason = tool_call.parameters.get("reason", "No reason provided")
            
            if self.verbose:
                print(f"🎮 Tool: press_buttons")
                print(f"   Buttons: {buttons}")
                print(f"   Reason: {reason}")
            
            self.logger.info(f"Pressing buttons: {buttons} - {reason}")
            
            # Add buttons to queue
            for button in buttons:
                if button in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R"]:
                    self.button_queue.append(button)
            
        elif tool_call.name == "navigate_to" and self.enable_navigation:
            target_x = tool_call.parameters.get("x", 0)
            target_y = tool_call.parameters.get("y", 0)
            reason = tool_call.parameters.get("reason", "No reason provided")
            
            if self.verbose:
                print(f"🗺️ Tool: navigate_to")
                print(f"   Target: ({target_x}, {target_y})")
                print(f"   Reason: {reason}")
            
            self.logger.info(f"Navigating to ({target_x}, {target_y}) - {reason}")
            
            # Get current position
            current_pos = state.get("player_position", {})
            current_x = current_pos.get("x", 0)
            current_y = current_pos.get("y", 0)
            start = (current_x, current_y)
            goal = (target_x, target_y)
            
            # Use advanced pathfinding with collision detection
            path_buttons = self.pathfinder.find_path(start, goal, state)
            
            if path_buttons:
                if self.verbose:
                    print(f"   Path found: {len(path_buttons)} steps")
                    if len(path_buttons) <= 10:
                        print(f"   Path: {' → '.join(path_buttons)}")
                    else:
                        print(f"   Path: {' → '.join(path_buttons[:5])} ... {' → '.join(path_buttons[-5:])}")
                
                # Add path to button queue
                self.button_queue.extend(path_buttons)
            else:
                if self.verbose:
                    print(f"   ⚠️ No path found to ({target_x}, {target_y})")
                    print(f"   Attempting simple movement as fallback")
                
                # Fallback to simple movement
                x_diff = target_x - current_x
                y_diff = target_y - current_y
                
                # Limit to reasonable number of steps
                if abs(x_diff) <= 10:
                    if x_diff > 0:
                        self.button_queue.extend(["RIGHT"] * min(x_diff, 5))
                    elif x_diff < 0:
                        self.button_queue.extend(["LEFT"] * min(abs(x_diff), 5))
                
                if abs(y_diff) <= 10:
                    if y_diff > 0:
                        self.button_queue.extend(["DOWN"] * min(y_diff, 5))
                    elif y_diff < 0:
                        self.button_queue.extend(["UP"] * min(abs(y_diff), 5))
    
    def _summarize_history(self) -> None:
        """Summarize conversation history when it gets too long (matching original)."""
        if self.verbose:
            print("📚 Summarizing conversation history...")
        
        # Build summary prompt
        history_text = []
        for msg in self.messages[-20:]:  # Summarize last 20 messages
            if msg["role"] == "user":
                if isinstance(msg["content"], list):
                    text_parts = [c["text"] for c in msg["content"] if c["type"] == "text"]
                    history_text.append(f"User: {' '.join(text_parts)[:300]}")
            elif msg["role"] == "assistant":
                history_text.append(f"Assistant: {msg['content'][:300]}")
        
        summary_prompt = f"""Summarize this Pokemon Emerald gameplay conversation in 2-3 concise sentences:

{chr(10).join(history_text)}

Focus on: current location, recent achievements, current objective, and any problems encountered."""
        
        summary = self.vlm_client.query(
            prompt=summary_prompt,
            max_tokens=200,
            temperature=0.0
        )
        
        self.logger.info(f"History summarized: {summary}")
        
        # Replace history with summary
        self.messages = [
            {"role": "system", "content": f"Previous game summary: {summary}"}
        ]
        
        if self.verbose:
            print(f"   Summary: {summary}")
    
    # Convenience methods for compatibility
    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the agent for specified steps (for standalone usage)."""
        self.running = True
        self.logger.info("ClaudePlaysPokemon agent started")
        
        steps = 0
        while self.running:
            if num_steps and steps >= num_steps:
                break
            steps += 1
            
            # This would be called by the game loop
            # For now just track that we're running
            if self.verbose and steps % 100 == 0:
                print(f"Agent running... {steps} steps taken")
    
    def stop(self) -> None:
        """Stop the agent."""
        self.running = False
        self.logger.info("ClaudePlaysPokemon agent stopped")


# Factory function for easy integration
def create_claude_plays_agent(**kwargs) -> ClaudePlaysAgent:
    """Create a ClaudePlaysPokemon agent instance."""
    return ClaudePlaysAgent(**kwargs)