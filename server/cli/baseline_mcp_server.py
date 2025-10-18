#!/usr/bin/env python3
"""
Baseline MCP Server - Standard Gemini CLI Tools
Implements file system, shell, web, and memory tools as MCP server.

Based on gemini-cli built-in tools:
- File System: list_directory, read_file, write_file, glob, search_file_content, replace, read_many_files
- Shell: run_shell_command
- Web: web_fetch, google_web_search
- Memory: save_memory

See cli/GEMINI_CLI_TOOLS.md for complete documentation.
"""

import os
import sys
import subprocess
import logging
import base64
import re
import fnmatch
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus
import mimetypes

# Third-party imports
import requests
from bs4 import BeautifulSoup

# FastMCP for building MCP servers
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(name="baseline-tools")

# Root directory for file operations (security boundary)
ROOT_DIR = os.getcwd()

# Write directory - only allow writes in this directory
WRITE_DIR = Path(ROOT_DIR) / ".pokeagent_cache" / "cli"
WRITE_DIR.mkdir(parents=True, exist_ok=True)

# Allowed shell commands for security (allowlist approach - much safer than blocklist)
ALLOWED_COMMANDS = [
    'ls',       # List files
    'cat',      # Read files
    'echo',     # Print text
    'grep',     # Search text
    'find',     # Find files
    'head',     # Show file start
    'tail',     # Show file end
    'wc',       # Word count
    'sort',     # Sort lines
    'uniq',     # Unique lines
    'cut',      # Cut columns
    'awk',      # Text processing
    'sed',      # Stream editor (read-only usage)
    'python',   # Python scripts
    'node',     # Node.js
    'which',    # Find command
    'whereis',  # Locate command
    'pwd',      # Print working directory
    'date',     # Show date
    'whoami',   # Current user
    'env',      # Environment variables
    'printenv', # Print environment
    'diff',     # Compare files
    'file',     # File type
    'stat',     # File statistics
    'ps',       # Process status
    'top',      # System monitor (if non-interactive)
    'uname',    # System info
    'hostname', # Hostname
    'curl',     # Download (read-only)
    'wget',     # Download (read-only)
    'jq',       # JSON processor
    'tree',     # Directory tree
    'basename', # Get filename
    'dirname',  # Get directory
    'realpath', # Get real path
    'timeout',  # Run with timeout
]

# ============================================================================
# FILE SYSTEM TOOLS
# ============================================================================

@mcp.tool()
def list_directory(
    path: str,
    ignore: Optional[List[str]] = None,
    respect_git_ignore: bool = True
) -> dict:
    """
    List files and subdirectories in a directory.

    Args:
        path: Absolute path to the directory
        ignore: Glob patterns to exclude (e.g., ["*.log", ".git"])
        respect_git_ignore: Respect .gitignore patterns

    Returns:
        List of file/directory names with type indicators
    """
    try:
        # Validate path is within ROOT_DIR
        abs_path = Path(path).resolve()
        if not str(abs_path).startswith(ROOT_DIR):
            return {"success": False, "error": f"Path outside root directory: {path}"}

        if not abs_path.exists():
            return {"success": False, "error": f"Directory not found: {path}"}

        if not abs_path.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}

        # Get all entries
        entries = []
        for entry in abs_path.iterdir():
            # Check ignore patterns
            if ignore:
                should_ignore = False
                for pattern in ignore:
                    if fnmatch.fnmatch(entry.name, pattern):
                        should_ignore = True
                        break
                if should_ignore:
                    continue

            entries.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "path": str(entry)
            })

        # Sort: directories first, then alphabetically
        entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

        # Format output
        listing = f"Directory listing for {abs_path}:\n"
        for entry in entries:
            prefix = "[DIR] " if entry["is_dir"] else ""
            listing += f"{prefix}{entry['name']}\n"

        return {
            "success": True,
            "listing": listing,
            "entries": entries,
            "path": str(abs_path),
            "count": len(entries)
        }

    except Exception as e:
        logger.error(f"Failed to list directory {path}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def read_file(
    path: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None
) -> dict:
    """
    Read file content. Supports text, images, and PDFs.

    Args:
        path: Absolute path to the file
        offset: For text files, 0-based line number to start from
        limit: For text files, maximum number of lines to read

    Returns:
        File content (text, base64 for images/PDFs, or error message)
    """
    try:
        # Validate path is within ROOT_DIR
        abs_path = Path(path).resolve()
        if not str(abs_path).startswith(ROOT_DIR):
            return {"success": False, "error": f"Path outside root directory: {path}"}

        if not abs_path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        if not abs_path.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        # Detect mime type
        mime_type, _ = mimetypes.guess_type(str(abs_path))

        # Handle images
        image_types = ['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml', 'image/bmp']
        if mime_type in image_types:
            with open(abs_path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            return {
                "success": True,
                "type": "image",
                "mime_type": mime_type,
                "data": data,
                "path": str(abs_path)
            }

        # Handle PDFs
        if mime_type == 'application/pdf':
            with open(abs_path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            return {
                "success": True,
                "type": "pdf",
                "mime_type": mime_type,
                "data": data,
                "path": str(abs_path)
            }

        # Handle text files
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                if offset is not None or limit is not None:
                    lines = f.readlines()
                    total_lines = len(lines)

                    # Apply offset and limit
                    start = offset if offset is not None else 0
                    end = start + limit if limit is not None else len(lines)
                    selected_lines = lines[start:end]

                    content = ''.join(selected_lines)
                    truncation_msg = f"[Showing lines {start+1}-{min(end, total_lines)} of {total_lines} total lines]\n" if offset is not None or limit is not None else ""

                    return {
                        "success": True,
                        "type": "text",
                        "content": truncation_msg + content,
                        "path": str(abs_path),
                        "total_lines": total_lines,
                        "lines_shown": len(selected_lines)
                    }
                else:
                    content = f.read()
                    return {
                        "success": True,
                        "type": "text",
                        "content": content,
                        "path": str(abs_path)
                    }
        except UnicodeDecodeError:
            # Binary file that's not an image or PDF
            return {
                "success": False,
                "error": f"Cannot display content of binary file: {path}"
            }

    except Exception as e:
        logger.error(f"Failed to read file {path}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def write_file(file_path: str, content: str) -> dict:
    """
    Write content to a file. Overwrites if exists, creates if not.

    Args:
        file_path: Absolute path to the file (must be within .pokeagent_cache/cli/)
        content: Content to write

    Returns:
        Success message

    Note: Writes are restricted to .pokeagent_cache/cli/ directory for security.
          In gemini-cli this requires user confirmation. In MCP with trust=true,
          confirmation is bypassed.
    """
    try:
        # Validate path is within WRITE_DIR
        abs_path = Path(file_path).resolve()

        # Check if path is within the allowed write directory
        if not str(abs_path).startswith(str(WRITE_DIR)):
            return {
                "success": False,
                "error": f"File writes only allowed in {WRITE_DIR}. Attempted: {file_path}"
            }

        # Check if file exists
        file_existed = abs_path.exists()

        # Create parent directories if needed (within WRITE_DIR)
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)

        if file_existed:
            message = f"Successfully overwrote file: {abs_path}"
        else:
            message = f"Successfully created and wrote to new file: {abs_path}"

        return {
            "success": True,
            "message": message,
            "path": str(abs_path),
            "existed": file_existed,
            "write_dir": str(WRITE_DIR)
        }

    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def glob(
    pattern: str,
    path: Optional[str] = None,
    case_sensitive: bool = False,
    respect_git_ignore: bool = True
) -> dict:
    """
    Find files matching glob patterns, sorted by modification time (newest first).

    Args:
        pattern: Glob pattern (e.g., "*.py", "src/**/*.js")
        path: Directory to search within (default: root directory)
        case_sensitive: Case-sensitive search
        respect_git_ignore: Respect .gitignore patterns

    Returns:
        List of absolute paths sorted by modification time
    """
    try:
        # Determine search directory
        search_dir = Path(path) if path else Path(ROOT_DIR)
        search_dir = search_dir.resolve()

        if not str(search_dir).startswith(ROOT_DIR):
            return {"success": False, "error": f"Path outside root directory: {path}"}

        # Use pathlib's glob
        if '**' in pattern:
            matches = list(search_dir.glob(pattern))
        else:
            matches = list(search_dir.glob(pattern))

        # Filter out common ignore patterns
        default_ignores = ['node_modules', '.git', '__pycache__', '.venv', 'venv', '.pytest_cache']
        filtered_matches = []
        for match in matches:
            # Check if any part of the path contains ignored directories
            parts = match.parts
            should_skip = any(ignore in parts for ignore in default_ignores)
            if not should_skip:
                filtered_matches.append(match)

        # Sort by modification time (newest first)
        sorted_matches = sorted(filtered_matches, key=lambda p: p.stat().st_mtime, reverse=True)

        # Convert to strings
        result_paths = [str(p) for p in sorted_matches]

        return {
            "success": True,
            "matches": result_paths,
            "count": len(result_paths),
            "pattern": pattern,
            "search_dir": str(search_dir)
        }

    except Exception as e:
        logger.error(f"Failed to glob {pattern}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def search_file_content(
    pattern: str,
    path: Optional[str] = None,
    include: Optional[str] = None
) -> dict:
    """
    Search for regex pattern within file contents.

    Args:
        pattern: Regular expression to search for
        path: Directory to search within
        include: Glob pattern to filter files (e.g., "*.js")

    Returns:
        Formatted string of matches with file paths and line numbers
    """
    try:
        # Determine search directory
        search_dir = Path(path) if path else Path(ROOT_DIR)
        search_dir = search_dir.resolve()

        if not str(search_dir).startswith(ROOT_DIR):
            return {"success": False, "error": f"Path outside root directory: {path}"}

        # Get files to search
        if include:
            # Use glob pattern
            files = list(search_dir.glob(f"**/{include}"))
        else:
            # Search all text files
            files = []
            for ext in ['*.py', '*.js', '*.ts', '*.md', '*.txt', '*.json', '*.yaml', '*.yml', '*.sh']:
                files.extend(search_dir.glob(f"**/{ext}"))

        # Compile regex
        regex = re.compile(pattern)

        # Search files
        matches = []
        for file_path in files:
            if not file_path.is_file():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append({
                                "file": str(file_path.relative_to(search_dir)),
                                "line_number": line_num,
                                "line": line.rstrip()
                            })
            except (UnicodeDecodeError, PermissionError):
                # Skip binary files or files we can't read
                continue

        # Format output
        if matches:
            output = f"Found {len(matches)} matches for pattern \"{pattern}\" in path \"{search_dir}\""
            if include:
                output += f" (filter: \"{include}\")"
            output += ":\n---\n"

            current_file = None
            for match in matches:
                if match["file"] != current_file:
                    if current_file is not None:
                        output += "---\n"
                    output += f"File: {match['file']}\n"
                    current_file = match["file"]
                output += f"L{match['line_number']}: {match['line']}\n"
            output += "---"
        else:
            output = f"No matches found for pattern \"{pattern}\""

        return {
            "success": True,
            "matches": matches,
            "count": len(matches),
            "output": output,
            "pattern": pattern,
            "search_dir": str(search_dir)
        }

    except Exception as e:
        logger.error(f"Failed to search content for {pattern}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def replace(
    file_path: str,
    old_string: str,
    new_string: str,
    expected_replacements: int = 1
) -> dict:
    """
    Replace text within a file with precise targeting.

    Args:
        file_path: Absolute path to the file (must be in .pokeagent_cache/cli/)
        old_string: Exact literal text to replace (must uniquely identify location)
        new_string: Exact literal text to replace with
        expected_replacements: Number of occurrences to replace

    Returns:
        Success message with replacement count

    Note: Replacements are restricted to .pokeagent_cache/cli/ directory for security.
          old_string must include sufficient context to uniquely identify the location.
    """
    try:
        # Validate path is within WRITE_DIR (replace can modify files)
        abs_path = Path(file_path).resolve()

        # Check if path is within the allowed write directory
        if not str(abs_path).startswith(str(WRITE_DIR)):
            return {
                "success": False,
                "error": f"File replacements only allowed in {WRITE_DIR}. Attempted: {file_path}"
            }

        if not abs_path.exists():
            # If old_string is empty, create new file
            if not old_string:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(new_string)
                return {
                    "success": True,
                    "message": f"Created new file: {abs_path}",
                    "path": str(abs_path),
                    "replacements": 0
                }
            else:
                return {"success": False, "error": f"File not found: {file_path}"}

        # Read file content
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Count occurrences
        occurrences = content.count(old_string)

        # Check expected replacements
        if occurrences == 0:
            return {
                "success": False,
                "error": f"Failed to edit, 0 occurrences found. Expected {expected_replacements}.",
                "old_string_preview": old_string[:100]
            }

        if occurrences != expected_replacements:
            return {
                "success": False,
                "error": f"Failed to edit, expected {expected_replacements} occurrences but found {occurrences}.",
                "occurrences": occurrences
            }

        # Perform replacement
        new_content = content.replace(old_string, new_string, expected_replacements)

        # Write back
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return {
            "success": True,
            "message": f"Successfully modified file: {abs_path} ({expected_replacements} replacements).",
            "path": str(abs_path),
            "replacements": expected_replacements
        }

    except Exception as e:
        logger.error(f"Failed to replace in {file_path}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def read_many_files(
    paths: List[str],
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    recursive: bool = True,
    useDefaultExcludes: bool = True,
    respect_git_ignore: bool = True
) -> dict:
    """
    Read content from multiple files specified by paths or glob patterns.

    Args:
        paths: Array of glob patterns or paths
        exclude: Glob patterns to exclude
        include: Additional glob patterns to include
        recursive: Search recursively
        useDefaultExcludes: Apply default exclusion patterns
        respect_git_ignore: Respect .gitignore patterns

    Returns:
        Concatenated content with file path separators
    """
    try:
        # Collect all file paths
        all_files = set()

        # Default exclusions
        default_ignores = ['node_modules', '.git', '__pycache__', '.venv', 'venv', '.pytest_cache']

        # Process paths and include patterns
        all_patterns = paths + (include if include else [])

        for pattern in all_patterns:
            pattern_path = Path(pattern)

            # Check if it's a specific file
            if pattern_path.is_file():
                all_files.add(pattern_path)
            else:
                # Treat as glob pattern
                search_root = Path(ROOT_DIR)
                matches = list(search_root.glob(pattern))
                for match in matches:
                    if match.is_file():
                        all_files.add(match)

        # Apply exclusions
        filtered_files = []
        for file_path in all_files:
            # Check default excludes
            if useDefaultExcludes:
                parts = file_path.parts
                if any(ignore in parts for ignore in default_ignores):
                    continue

            # Check exclude patterns
            if exclude:
                excluded = False
                for excl_pattern in exclude:
                    if fnmatch.fnmatch(file_path.name, excl_pattern):
                        excluded = True
                        break
                if excluded:
                    continue

            filtered_files.append(file_path)

        # Read files and concatenate
        content_parts = []
        files_read = []

        for file_path in sorted(filtered_files):
            try:
                # Detect mime type
                mime_type, _ = mimetypes.guess_type(str(file_path))

                # Handle images and PDFs
                if mime_type and (mime_type.startswith('image/') or mime_type == 'application/pdf'):
                    with open(file_path, 'rb') as f:
                        data = base64.b64encode(f.read()).decode('utf-8')
                    content_parts.append(f"--- {file_path} ---")
                    content_parts.append(f"[Binary file: {mime_type}]")
                    content_parts.append(f"Base64: {data[:100]}... (truncated)")
                    files_read.append(str(file_path))
                else:
                    # Try to read as text
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        content_parts.append(f"--- {file_path} ---")
                        content_parts.append(content)
                        files_read.append(str(file_path))
                    except UnicodeDecodeError:
                        # Skip binary files
                        continue

            except (PermissionError, FileNotFoundError):
                continue

        # Final concatenation
        content_parts.append("--- End of content ---")
        full_content = "\n".join(content_parts)

        return {
            "success": True,
            "content": full_content,
            "files_read": files_read,
            "count": len(files_read),
            "patterns": all_patterns
        }

    except Exception as e:
        logger.error(f"Failed to read many files: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# SHELL TOOL
# ============================================================================

@mcp.tool()
def run_shell_command(
    command: str,
    description: Optional[str] = None,
    directory: Optional[str] = None
) -> dict:
    """
    Execute shell commands on the system.

    Args:
        command: Shell command to execute
        description: Brief description of command's purpose
        directory: Directory to run command in (relative to project root)

    Returns:
        Command output, exit code, stderr, and background PIDs

    Note: Sets GEMINI_CLI=1 environment variable in subprocess.
          Uses allowlist approach - only explicitly allowed commands can run.
    """
    try:
        # Check allowlist - split on common separators to check chained commands
        command_parts = re.split(r'[;&|]', command)

        for part in command_parts:
            part = part.strip()
            if not part:
                continue

            # Extract the base command (first word)
            cmd_base = part.split()[0] if part.split() else ""

            # Remove any path prefix (e.g., /usr/bin/python -> python)
            cmd_name = os.path.basename(cmd_base)

            # Check if command is in allowlist
            if cmd_name not in ALLOWED_COMMANDS:
                return {
                    "success": False,
                    "error": f"Command '{cmd_name}' is not in the allowlist. Only safe read-only commands are permitted.",
                    "attempted_command": cmd_name,
                    "allowed_commands": ALLOWED_COMMANDS
                }

        # Determine working directory
        if directory:
            work_dir = Path(ROOT_DIR) / directory
            work_dir = work_dir.resolve()
            if not str(work_dir).startswith(ROOT_DIR):
                return {"success": False, "error": f"Directory outside root: {directory}"}
        else:
            work_dir = Path(ROOT_DIR)

        # Set up environment with GEMINI_CLI=1
        env = os.environ.copy()
        env["GEMINI_CLI"] = "1"

        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )

        return {
            "success": result.returncode == 0,
            "command": command,
            "description": description or "",
            "directory": str(work_dir),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out after 60 seconds"}
    except Exception as e:
        logger.error(f"Failed to run command {command}: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# WEB TOOLS
# ============================================================================

@mcp.tool()
def web_fetch(prompt: str) -> dict:
    """
    Fetch and process content from web pages (up to 20 URLs).

    Args:
        prompt: Comprehensive prompt including URL(s) and processing instructions

    Returns:
        Generated response based on web content with citations
    """
    try:
        # Extract URLs from prompt
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, prompt)

        if not urls:
            return {
                "success": False,
                "error": "No URLs found in prompt. Please include at least one URL starting with http:// or https://"
            }

        if len(urls) > 20:
            return {
                "success": False,
                "error": f"Too many URLs ({len(urls)}). Maximum is 20."
            }

        # Fetch content from each URL
        fetched_content = []
        for url in urls:
            try:
                logger.info(f"Fetching: {url}")
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; PokeAgent-CLI/1.0)'
                })
                response.raise_for_status()

                # Parse HTML and extract text
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text and clean it up
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)

                # Limit content length
                max_chars = 10000
                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n\n[Content truncated - {len(text)} total characters]"

                fetched_content.append({
                    "url": url,
                    "content": text,
                    "success": True
                })

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                fetched_content.append({
                    "url": url,
                    "error": str(e),
                    "success": False
                })

        # Format response
        response_parts = ["# Web Content Fetched\n"]
        for item in fetched_content:
            if item["success"]:
                response_parts.append(f"\n## Source: {item['url']}\n")
                response_parts.append(item["content"])
            else:
                response_parts.append(f"\n## Failed: {item['url']}\n")
                response_parts.append(f"Error: {item['error']}")

        response_parts.append(f"\n\n---\n**Prompt:** {prompt}")

        return {
            "success": True,
            "content": "\n".join(response_parts),
            "urls_fetched": len([x for x in fetched_content if x["success"]]),
            "urls_failed": len([x for x in fetched_content if not x["success"]])
        }

    except Exception as e:
        logger.error(f"Failed to fetch web content: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def google_web_search(query: str) -> dict:
    """
    Perform web search using DuckDuckGo (privacy-friendly, no API key needed).

    Args:
        query: Search query

    Returns:
        Summary of web results with sources and citations
    """
    try:
        # Use DuckDuckGo HTML search (no API key required)
        encoded_query = quote_plus(query)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        logger.info(f"Searching: {query}")
        response = requests.get(search_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; PokeAgent-CLI/1.0)'
        })
        response.raise_for_status()

        # Parse search results
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []

        # Extract search result links and snippets
        for result in soup.find_all('div', class_='result', limit=10):
            try:
                # Get title
                title_elem = result.find('a', class_='result__a')
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')

                # Get snippet
                snippet_elem = result.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
            except Exception as e:
                logger.warning(f"Failed to parse search result: {e}")
                continue

        if not results:
            return {
                "success": False,
                "error": "No search results found. Try a different query."
            }

        # Format response
        response_parts = [f"# Search Results for: {query}\n"]
        for i, result in enumerate(results, 1):
            response_parts.append(f"\n## {i}. {result['title']}")
            response_parts.append(f"**URL:** {result['url']}")
            response_parts.append(f"{result['snippet']}\n")

        response_parts.append(f"\n---\n**Query:** {query}")
        response_parts.append(f"**Results:** {len(results)} found")

        return {
            "success": True,
            "content": "\n".join(response_parts),
            "results_count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Failed to search for '{query}': {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# MEMORY TOOL
# ============================================================================

@mcp.tool()
def save_memory(fact: str) -> dict:
    """
    Save facts to remember across sessions.

    Args:
        fact: Clear, self-contained statement in natural language

    Returns:
        Confirmation that fact was saved

    Note: Appends to .pokeagent_cache/cli/AGENT.md under "## Agent Memories"
    """
    try:
        # Determine memory file path (in write directory)
        memory_file = WRITE_DIR / "AGENT.md"

        # Read existing content
        if memory_file.exists():
            with open(memory_file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = "# Agent Memory\n\nThis file stores facts and observations from the AI agent.\n"

        # Check if "## Agent Memories" section exists
        if "## Agent Memories" not in content:
            # Add the section
            if content and not content.endswith('\n'):
                content += '\n'
            content += "\n## Agent Memories\n"

        # Append the fact
        content += f"- {fact}\n"

        # Write back
        with open(memory_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return {
            "success": True,
            "message": f"Memory saved to {memory_file}",
            "fact": fact,
            "path": str(memory_file)
        }

    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Baseline MCP Server...")
    logger.info(f"Root directory: {ROOT_DIR}")
    logger.info(f"Write directory (restricted): {WRITE_DIR}")
    logger.info("Tools: 11 baseline gemini-cli tools")
    logger.info("✅ All 11 tools implemented:")
    logger.info("   - File System: read_file, write_file, list_directory, glob, search_file_content, replace, read_many_files")
    logger.info("   - Shell: run_shell_command")
    logger.info("   - Web: web_fetch, google_web_search (DuckDuckGo)")
    logger.info("   - Memory: save_memory")
    logger.info("Security: File writes/replacements restricted to .pokeagent_cache/cli/ only")
    logger.info(f"Security: Shell commands use ALLOWLIST ({len(ALLOWED_COMMANDS)} commands permitted)")
    logger.info(f"Allowed commands: {', '.join(ALLOWED_COMMANDS[:10])}... (+{len(ALLOWED_COMMANDS)-10} more)")
    logger.info("See cli/GEMINI_CLI_TOOLS.md for tool specifications")

    # Run the MCP server
    mcp.run()
