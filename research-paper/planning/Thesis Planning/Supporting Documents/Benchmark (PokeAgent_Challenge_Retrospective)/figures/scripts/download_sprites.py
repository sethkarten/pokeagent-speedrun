#!/usr/bin/env python3
"""Download Pokemon sprites from PokeAPI for the battle loop diagram."""

import urllib.request
import os
from pathlib import Path

# 12 popular/iconic Pokemon (mix of types and recognizable shapes)
POKEMON = [
    # Team 1 (6 Pokemon)
    {"name": "pikachu", "id": 25},
    {"name": "charizard", "id": 6},
    {"name": "mewtwo", "id": 150},
    {"name": "gengar", "id": 94},
    {"name": "gyarados", "id": 130},
    {"name": "snorlax", "id": 143},
    # Team 2 (6 Pokemon)
    {"name": "blastoise", "id": 9},
    {"name": "dragonite", "id": 149},
    {"name": "alakazam", "id": 65},
    {"name": "machamp", "id": 68},
    {"name": "lapras", "id": 131},
    {"name": "venusaur", "id": 3},
]

def download_sprites():
    """Download front sprites for each Pokemon."""
    script_dir = Path(__file__).parent.parent
    sprites_dir = script_dir / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

    for pokemon in POKEMON:
        name = pokemon["name"]
        pid = pokemon["id"]

        # Download front sprite
        url = f"{base_url}/{pid}.png"
        output_path = sprites_dir / f"{name}.png"

        print(f"Downloading {name} (#{pid})...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"  -> Saved to {output_path}")
        except Exception as e:
            print(f"  -> Error: {e}")

    print(f"\nDownloaded {len(POKEMON)} sprites to {sprites_dir}")

    # Print the list for SVG embedding
    print("\nTeam 1:", [p["name"] for p in POKEMON[:6]])
    print("Team 2:", [p["name"] for p in POKEMON[6:]])

if __name__ == "__main__":
    download_sprites()
