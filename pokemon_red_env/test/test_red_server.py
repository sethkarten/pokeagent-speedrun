#!/usr/bin/env python3
"""
Integration test: start server with --game red, load a test state,
hit /stream, /state, /whole_map and other endpoints, validate responses.

Usage:
    python pokemon_red_env/test/test_red_server.py
"""

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import requests

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ROM = REPO / "PokemonRed-GBC" / "pokered.gbc"
# Use Oak's Lab state — has NPCs, warps, and a known map
TEST_STATE = REPO / "PokemonRed-GBC" / "test_states" / "pokered_battle_1.state"
PORT = 18765
BASE = f"http://localhost:{PORT}"

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  {detail}")


def wait_for_server(timeout=15):
    """Poll /health until server is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def start_server():
    """Start server subprocess with --game red."""
    if not ROM.exists():
        print(f"ROM not found: {ROM}")
        sys.exit(1)
    if not TEST_STATE.exists():
        print(f"Test state not found: {TEST_STATE}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "server.app",
        "--game", "red",
        "--port", str(PORT),
        "--load-state", str(TEST_STATE),
        "--no-ocr",
    ]
    env = os.environ.copy()
    env["GAME_TYPE"] = "red"
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


# ---------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------

def test_health():
    print("\n--- /health ---")
    r = requests.get(f"{BASE}/health")
    check("status 200", r.status_code == 200)
    data = r.json()
    check("status healthy", data.get("status") == "healthy")


def test_config():
    print("\n--- /config ---")
    r = requests.get(f"{BASE}/config")
    check("status 200", r.status_code == 200)
    data = r.json()
    check("game == red", data.get("game") == "red", f"got {data.get('game')}")
    check("width == 160", data.get("width") == 160, f"got {data.get('width')}")
    check("height == 144", data.get("height") == 144, f"got {data.get('height')}")


def test_stream():
    print("\n--- /stream ---")
    r = requests.get(f"{BASE}/stream")
    check("status 200", r.status_code == 200)
    check("HTML content", "text/html" in r.headers.get("content-type", ""))


def test_screenshot():
    print("\n--- /screenshot ---")
    r = requests.get(f"{BASE}/screenshot")
    check("status 200", r.status_code == 200)
    data = r.json()
    check("has screenshot_base64", "screenshot_base64" in data)
    resolution = data.get("resolution", [])
    check("resolution [160, 144]", resolution == [160, 144], f"got {resolution}")


def test_state():
    print("\n--- /state ---")
    r = requests.get(f"{BASE}/state")
    check("status 200", r.status_code == 200)
    data = r.json()

    # Top-level keys
    for key in ("visual", "player", "game", "map"):
        check(f"has '{key}' key", key in data)

    # Player
    player = data.get("player", {})
    location = player.get("location")
    check("player.location exists", location is not None and location != "Unknown", f"got {location}")
    position = player.get("position", {})
    check("player.position has x/y", "x" in position and "y" in position, f"got {position}")

    # Map
    map_data = data.get("map", {})
    visual_map = map_data.get("visual_map")
    map_source = map_data.get("map_source")
    check("visual_map exists", visual_map is not None and len(visual_map) > 0)
    check("map_source is red_map_reader", map_source == "red_map_reader", f"got {map_source}")

    if visual_map:
        # Visual map should contain player marker 'P'
        check("visual_map contains 'P' (player)", "P" in visual_map)
        lines = visual_map.strip().split("\n")
        check(f"visual_map has rows (got {len(lines)})", len(lines) > 0)
        print(f"  visual_map preview ({len(lines)} rows):")
        for line in lines[:5]:
            print(f"    {line}")

    # Game
    game_data = data.get("game", {})
    check("game.game_state exists", "game_state" in game_data, f"keys: {list(game_data.keys())}")


def test_whole_map():
    print("\n--- /whole_map ---")
    r = requests.get(f"{BASE}/whole_map")
    check("status 200", r.status_code == 200)
    data = r.json()

    check("has 'location'", "location" in data)
    check("has 'grid'", "grid" in data)
    check("has 'dimensions'", "dimensions" in data)
    check("has 'player_position'", "player_position" in data)
    check("has 'warps'", "warps" in data)
    check("has 'objects'", "objects" in data)
    check("has 'raw_tiles'", "raw_tiles" in data)
    check("has 'behavior_map'", "behavior_map" in data)
    check("has 'elevation_map'", "elevation_map" in data)

    dims = data.get("dimensions", {})
    check("dimensions has width/height", "width" in dims and "height" in dims, f"got {dims}")

    grid = data.get("grid", [])
    check(f"grid has rows (got {len(grid)})", len(grid) > 0)

    location = data.get("location", "")
    print(f"  location: {location}")
    print(f"  dimensions: {dims}")
    print(f"  warps: {len(data.get('warps', []))}")
    print(f"  objects: {len(data.get('objects', []))}")


def test_action():
    print("\n--- /action ---")
    r = requests.post(f"{BASE}/action", json={"buttons": ["A"], "speed": "fast"})
    check("status 200", r.status_code == 200)
    data = r.json()
    check("status success", data.get("status") == "success")
    check("actions_queued == 1", data.get("actions_queued") == 1, f"got {data.get('actions_queued')}")


# ---------------------------------------------------------------
# MCP endpoint tests
# ---------------------------------------------------------------

def test_mcp_get_game_state():
    print("\n--- /mcp/get_game_state ---")
    r = requests.post(f"{BASE}/mcp/get_game_state")
    check("status 200", r.status_code == 200)
    data = r.json()
    check("success", data.get("success") == True)
    check("has state_text", bool(data.get("state_text")))
    check("has screenshot_base64", bool(data.get("screenshot_base64")))
    check("has raw_state", "raw_state" in data)


def test_mcp_press_buttons():
    print("\n--- /mcp/press_buttons ---")
    # Valid button
    r = requests.post(f"{BASE}/mcp/press_buttons", json={"buttons": ["A"], "reasoning": "test"})
    check("status 200", r.status_code == 200)
    check("success", r.json().get("success") == True)
    # L/R should be rejected for Red
    r2 = requests.post(f"{BASE}/mcp/press_buttons", json={"buttons": ["L"], "reasoning": "test"})
    check("L button rejected", r2.json().get("success") == False)
    r3 = requests.post(f"{BASE}/mcp/press_buttons", json={"buttons": ["R"], "reasoning": "test"})
    check("R button rejected", r3.json().get("success") == False)


def test_mcp_navigate_to():
    print("\n--- /mcp/navigate_to ---")
    r = requests.post(f"{BASE}/mcp/navigate_to", json={"x": 5, "y": 5, "reason": "test"})
    check("status 200", r.status_code == 200)
    data = r.json()
    check("success or graceful error", "success" in data)
    if data.get("success"):
        check("has buttons_queued", "buttons_queued" in data)


def test_mcp_knowledge():
    print("\n--- /mcp/add_knowledge ---")
    r = requests.post(f"{BASE}/mcp/add_knowledge", json={
        "category": "test", "title": "Test Entry", "content": "Test content", "importance": 5
    })
    check("add status 200", r.status_code == 200)
    check("add success", r.json().get("success") == True)

    print("\n--- /mcp/search_knowledge ---")
    r = requests.post(f"{BASE}/mcp/search_knowledge", json={"query": "Test"})
    check("search status 200", r.status_code == 200)
    check("search found results", r.json().get("count", 0) > 0)

    print("\n--- /mcp/get_knowledge_summary ---")
    r = requests.post(f"{BASE}/mcp/get_knowledge_summary", json={"min_importance": 1})
    check("summary status 200", r.status_code == 200)
    check("summary success", r.json().get("success") == True)


def test_mcp_reflect():
    print("\n--- /mcp/reflect ---")
    r = requests.post(f"{BASE}/mcp/reflect", json={"situation": "test reflection"})
    check("status 200", r.status_code == 200)
    data = r.json()
    check("success", data.get("success") == True)
    check("has context", "context" in data)


def test_mcp_get_walkthrough():
    print("\n--- /mcp/get_walkthrough ---")
    r = requests.post(f"{BASE}/mcp/get_walkthrough", json={"part": 1})
    check("status 200", r.status_code == 200)
    data = r.json()
    check("success", data.get("success") == True)
    check("is Red walkthrough", "Red" in data.get("url", ""))


def test_mcp_progress_summary():
    print("\n--- /mcp/get_progress_summary ---")
    r = requests.post(f"{BASE}/mcp/get_progress_summary")
    check("status 200", r.status_code == 200)
    data = r.json()
    check("success", data.get("success") == True)
    check("has progress", "progress" in data)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    global PASSED, FAILED

    print(f"Starting Red server integration test...")
    print(f"  ROM: {ROM}")
    print(f"  State: {TEST_STATE}")
    print(f"  Port: {PORT}")

    proc = start_server()
    try:
        print("Waiting for server to start...")
        if not wait_for_server():
            print("Server failed to start! Dumping output:")
            proc.kill()
            stdout, _ = proc.communicate(timeout=5)
            print(stdout.decode(errors="replace")[-3000:])
            sys.exit(1)
        print("Server is ready.\n")

        test_health()
        test_config()
        test_stream()
        test_screenshot()
        test_state()
        test_whole_map()
        test_action()

        # MCP tool endpoints
        test_mcp_get_game_state()
        test_mcp_press_buttons()
        test_mcp_navigate_to()
        test_mcp_knowledge()
        test_mcp_reflect()
        test_mcp_get_walkthrough()
        test_mcp_progress_summary()

        print(f"\n{'='*60}")
        print(f"Results: {PASSED} passed, {FAILED} failed")
        if FAILED == 0:
            print("All tests passed!")
        else:
            print("Some tests FAILED.")
            sys.exit(1)

    finally:
        # Clean up server process
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        print("Server stopped.")


if __name__ == "__main__":
    main()
