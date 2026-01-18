import requests
import json


def defeat_jared_and_move():
    url = "http://localhost:8000/mcp/press_buttons"

    # Sequence of buttons:
    # 1. UP to encounter/interact with Jared at (4,14)
    # 2. A sequence of A presses to clear dialogue and battle
    # 3. Two more UP presses to move from (4,14) through (4,13) to (4,12)
    # Total UP presses: 3 (moving from y=15 to y=12)
    buttons = ["UP"] + ["A"] * 60 + ["UP", "UP"]

    payload = {
        "buttons": buttons,
        "reasoning": "Moving from (4,15) to (4,12), defeating Bird Keeper Jared at (4,14) along the way.",
        "speed": "fast",
    }

    print(f"Sending {len(buttons)} button presses to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Response from server:")
        print(json.dumps(result, indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")


if __name__ == "__main__":
    defeat_jared_and_move()
