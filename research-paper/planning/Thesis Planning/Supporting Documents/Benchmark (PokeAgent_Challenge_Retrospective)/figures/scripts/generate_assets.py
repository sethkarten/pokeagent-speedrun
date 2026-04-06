#!/usr/bin/env python3
"""Generate Pokemon battle assets using Gemini image generation."""

import os
import base64
from pathlib import Path

# Use new google.genai package
from google import genai
from google.genai import types

def setup_api():
    """Set up Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    return client

def generate_battlefield(client, output_path):
    """Generate Pokemon battle arena background."""
    prompt = """Generate a clean, minimalist top-down view of a Pokemon battle arena.

Requirements:
- Simple green grass field in the center (hex color #55A868)
- Two circular trainer platform areas on opposite sides (left and right)
- White boundary lines marking the arena
- Flat, vector-style illustration suitable for academic publication
- No Pokemon, no trainers, just the empty arena
- Subtle depth with lighter green grass towards edges
- Clean, professional look for a research paper figure
- Aspect ratio: wide rectangle (approximately 4:1)
- Background should be very light gray (#F7FAFC) or transparent

Style: Minimalist, flat design, no gradients, clean lines, academic quality."""

    print("Generating battlefield asset...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"]
            )
        )

        # Check if we got an image
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    with open(output_path, 'wb') as f:
                        if isinstance(image_data, str):
                            f.write(base64.b64decode(image_data))
                        else:
                            f.write(image_data)
                    print(f"  -> Saved to {output_path}")
                    return True

        print("  -> No image generated in response")
        if response.text:
            print(f"  -> Text response: {response.text[:200]}")
        return False
    except Exception as e:
        print(f"  -> Error: {e}")
        return False

def generate_pokemon_pool(client, output_path):
    """Generate grid of Pokemon silhouettes."""
    prompt = """Generate a grid showing 30-40 different creature silhouettes arranged in neat rows, representing a variety of fantasy battle monsters.

Requirements:
- Various recognizable creature shapes - dragons, humanoid fighters, quadrupeds, flying creatures, aquatic monsters
- Simple dark gray (#4A5568) silhouettes on white background
- Clean, flat style suitable for academic publication
- Grid layout: approximately 8 columns x 4-5 rows
- Each silhouette should be clearly distinguishable
- Solid filled silhouettes
- Minimal style, no text labels

Style: Flat silhouettes, academic quality, clean professional look."""

    print("Generating Pokemon pool asset...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"]
            )
        )

        # Check if we got an image
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    with open(output_path, 'wb') as f:
                        if isinstance(image_data, str):
                            f.write(base64.b64decode(image_data))
                        else:
                            f.write(image_data)
                    print(f"  -> Saved to {output_path}")
                    return True

        print("  -> No image generated in response")
        if response.text:
            print(f"  -> Text response: {response.text[:200]}")
        return False
    except Exception as e:
        print(f"  -> Error: {e}")
        return False

def main():
    script_dir = Path(__file__).parent.parent
    assets_dir = script_dir / "assets" / "custom"
    assets_dir.mkdir(parents=True, exist_ok=True)

    client = setup_api()

    # Generate assets
    battlefield_path = assets_dir / "battlefield.png"
    pool_path = assets_dir / "pokemon_pool.png"

    generate_battlefield(client, battlefield_path)
    generate_pokemon_pool(client, pool_path)

    print("\nAsset generation complete!")

if __name__ == "__main__":
    main()
