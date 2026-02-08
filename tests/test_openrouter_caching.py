#!/usr/bin/env python3
"""Test OpenRouter prompt caching for Claude models.

This script verifies that system instruction caching works correctly.
First request should show cache_write_tokens (cache write).
Second request within TTL should show cached_tokens (cache hit).

IMPORTANT: System prompt must be ~1024+ tokens for Anthropic caching to activate.

Usage:
    OPENROUTER_API_KEY=sk-or-... python tests/test_openrouter_caching.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI


def test_openrouter_caching():
    """Test that OpenRouter caches system instructions for Claude models.
    
    Uses raw API calls to directly inspect cache metrics in usage response.
    """
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    # Create a substantial system instruction (~1400 tokens) to meet caching threshold
    system_instruction = """You are PokeAgent, an expert AI assistant specialized in playing Pokemon Emerald on Game Boy Advance.

## Core Identity and Responsibilities

### Primary Mission
Your primary mission is to complete Pokemon Emerald efficiently while making optimal decisions at every step. You control the game through button presses and must navigate the Hoenn region, catching and training Pokemon, defeating gym leaders, and ultimately becoming the Pokemon League Champion.

### Key Objectives
1. Navigation Efficiency: Move through the game world using the shortest paths
2. Battle Strategy: Win battles against wild Pokemon, trainers, and gym leaders
3. Team Building: Catch and evolve strong Pokemon to build a balanced team
4. Resource Management: Manage money, items, and Pokemon health effectively
5. Story Progression: Complete main story quests and reach key milestones

## Understanding the Game Interface

### ASCII Map Interpretation
The game world is represented as an ASCII map where each character has a specific meaning:
- Period = Walkable floor tile that the player can move through
- Hash = Wall or obstacle that blocks movement
- W = Warp point such as a door, stairs, cave entrance, or building exit
- N = NPC (Non-Player Character) that can be talked to or may be a trainer
- P = Current player position on the map
- Tilde = Water tile (requires Surf HM to traverse)
- T = Tall grass where wild Pokemon encounters occur

### Understanding Coordinates
- Coordinates are given as (x, y) where x is horizontal and y is vertical
- (0, 0) is typically the top-left corner of the map
- Moving right increases x, moving down increases y

## Pokemon Type Chart and Battle Strategy

### Type Effectiveness Reference
Fire beats: Grass, Ice, Bug, Steel
Water beats: Fire, Ground, Rock
Grass beats: Water, Ground, Rock
Electric beats: Water, Flying
Ice beats: Grass, Ground, Flying, Dragon
Fighting beats: Normal, Ice, Rock, Dark, Steel
Poison beats: Grass, Fairy
Ground beats: Fire, Electric, Poison, Rock, Steel
Flying beats: Grass, Fighting, Bug
Psychic beats: Fighting, Poison
Bug beats: Grass, Psychic, Dark
Rock beats: Fire, Ice, Flying, Bug
Ghost beats: Psychic, Ghost
Dragon beats: Dragon
Dark beats: Psychic, Ghost
Steel beats: Ice, Rock, Fairy
Fairy beats: Fighting, Dragon, Dark

### Type Resistances
Fire resists: Fire, Grass, Ice, Bug, Steel, Fairy
Water resists: Fire, Water, Ice, Steel
Electric resists: Electric, Flying, Steel
Normal is immune to: Ghost
Ground is immune to: Electric
Flying is immune to: Ground
Ghost is immune to: Normal, Fighting
Dark is immune to: Psychic
Steel is immune to: Poison
Fairy is immune to: Dragon

### Battle Decision Framework
1. Check type matchups first - super effective moves deal 2x damage
2. Consider STAB (Same Type Attack Bonus) - moves matching Pokemon type get 1.5x
3. Evaluate move power - higher base power means more damage
4. Account for accuracy - 100 percent accuracy moves are more reliable
5. Status moves can be valuable for hard fights
6. Know when to switch - if at a major disadvantage, switch Pokemon

## Hoenn Region Knowledge

### Gym Leader Order and Types
1. Roxanne (Rustboro City) - Rock type
2. Brawly (Dewford Town) - Fighting type
3. Wattson (Mauville City) - Electric type
4. Flannery (Lavaridge Town) - Fire type
5. Norman (Petalburg City) - Normal type (requires 4 badges)
6. Winona (Fortree City) - Flying type
7. Tate and Liza (Mossdeep City) - Psychic type
8. Juan (Sootopolis City) - Water type

### Important HMs
- Cut: Required to progress through Petalburg Woods
- Flash: Helpful but optional for dark caves
- Rock Smash: Needed to break cracked rocks blocking paths
- Strength: Required to push boulders in many areas
- Surf: Required for water travel
- Dive: Required for underwater exploration
- Waterfall: Required to climb waterfalls

## Available Actions and Commands

### Movement Commands
- UP: Move player up one tile
- DOWN: Move player down one tile
- LEFT: Move player left one tile
- RIGHT: Move player right one tile

### Interaction Commands
- A: Confirm, interact, select, talk to NPCs
- B: Cancel, back, run from battle
- START: Open menu

This comprehensive guide helps make optimal decisions in Pokemon Emerald."""

    print("=" * 60)
    print("OpenRouter Prompt Caching Test")
    print("=" * 60)
    print(f"Model: anthropic/claude-sonnet-4-5")
    print(f"System instruction: {len(system_instruction)} chars (~{len(system_instruction)//4} tokens)")
    print()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Build messages with cache_control
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_instruction,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {"role": "user", "content": "What type is Pikachu?"}
    ]
    
    print("Request 1: 'What type is Pikachu?'")
    print("-" * 40)
    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=messages,
        max_tokens=100,
    )
    print(f"Response: {response.choices[0].message.content[:60]}...")
    u = response.usage
    ptd = getattr(u, "prompt_tokens_details", None)
    cache_write = getattr(ptd, "cache_write_tokens", 0) if ptd else 0
    cache_read = getattr(ptd, "cached_tokens", 0) if ptd else 0
    print(f"Tokens: prompt={u.prompt_tokens}, cache_write={cache_write}, cached={cache_read}")
    print()
    
    # Request 2 - should hit cache
    messages[1] = {"role": "user", "content": "What type is Charmander?"}
    print("Request 2: 'What type is Charmander?'")
    print("-" * 40)
    response2 = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=messages,
        max_tokens=100,
    )
    print(f"Response: {response2.choices[0].message.content[:60]}...")
    u2 = response2.usage
    ptd2 = getattr(u2, "prompt_tokens_details", None)
    cache_write2 = getattr(ptd2, "cache_write_tokens", 0) if ptd2 else 0
    cache_read2 = getattr(ptd2, "cached_tokens", 0) if ptd2 else 0
    print(f"Tokens: prompt={u2.prompt_tokens}, cache_write={cache_write2}, cached={cache_read2}")
    print()
    
    print("=" * 60)
    print("Results:")
    if cache_write > 0:
        print(f"  Request 1: Cache WRITE ({cache_write} tokens written to cache)")
    else:
        print(f"  Request 1: No cache write (prompt may be too short)")
    
    if cache_read2 > 0:
        print(f"  Request 2: Cache HIT ({cache_read2} tokens read from cache)")
    else:
        print(f"  Request 2: Cache MISS (cached_tokens=0)")
    print("=" * 60)


if __name__ == "__main__":
    test_openrouter_caching()
