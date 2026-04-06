"""
Systematic enumeration of ALL battle state variables in Pokemon Showdown.

Parses the Showdown simulator source code to extract every volatile status,
side condition, field condition, slot condition, and per-pokemon persistent
state. Computes state space sizes for Gen 1 OU, Gen 9 OU, and Gen 9 VGC.

Usage:
    uv run python analysis/enumerate_battle_state.py
"""

import re
import json
import math
from pathlib import Path
from dataclasses import dataclass, field as dc_field, asdict
from typing import Optional
from collections import defaultdict

SHOWDOWN = Path(__file__).resolve().parent.parent / "pokemon-showdown"

# ── helpers ──────────────────────────────────────────────────────────────────

def read(path: str) -> str:
    return (SHOWDOWN / path).read_text(encoding="utf-8", errors="replace")


def extract_quoted(pattern: str, text: str) -> list[str]:
    """Extract all single-quoted strings matching pattern."""
    return re.findall(pattern, text)


# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class StateVar:
    name: str
    category: str        # weather, terrain, pseudo_weather, side_condition,
                         # slot_condition, volatile, nonvolatile_status,
                         # per_pokemon, per_active, stat_stage
    scope: str           # field, per_side, per_slot, per_active, per_pokemon
    num_values: int      # number of distinct states (including "off")
    gen_min: int = 1
    gen_max: int = 9
    doubles_only: bool = False
    singles_only: bool = False
    notes: str = ""
    source: str = ""     # which move/ability creates it


# ── 1. WEATHER ───────────────────────────────────────────────────────────────

def enumerate_weather() -> list[StateVar]:
    """Weather: None + 4 standard (1-8 turns each) + 3 primal (indefinite)."""
    return [
        StateVar("No Weather", "weather", "field", 1, notes="baseline"),
        StateVar("Sun", "weather", "field", 8, gen_min=2, notes="1-8 turns (Heat Rock extends)"),
        StateVar("Rain", "weather", "field", 8, gen_min=2, notes="1-8 turns (Damp Rock extends)"),
        StateVar("Sand", "weather", "field", 8, gen_min=2, notes="1-8 turns (Smooth Rock extends)"),
        StateVar("Snow", "weather", "field", 8, gen_min=9, notes="1-8 turns (Icy Rock extends); replaces Hail"),
        StateVar("Harsh Sun", "weather", "field", 1, gen_min=6, notes="Desolate Land; indefinite"),
        StateVar("Heavy Rain", "weather", "field", 1, gen_min=6, notes="Primordial Sea; indefinite"),
        StateVar("Strong Winds", "weather", "field", 1, gen_min=6, notes="Delta Stream; indefinite"),
    ]


# ── 2. TERRAIN ───────────────────────────────────────────────────────────────

def enumerate_terrain() -> list[StateVar]:
    return [
        StateVar("No Terrain", "terrain", "field", 1, gen_min=6),
        StateVar("Electric Terrain", "terrain", "field", 8, gen_min=6, notes="1-8 turns"),
        StateVar("Grassy Terrain", "terrain", "field", 8, gen_min=6, notes="1-8 turns"),
        StateVar("Misty Terrain", "terrain", "field", 8, gen_min=6, notes="1-8 turns"),
        StateVar("Psychic Terrain", "terrain", "field", 8, gen_min=6, notes="1-8 turns"),
    ]


# ── 3. PSEUDO-WEATHER ───────────────────────────────────────────────────────

def enumerate_pseudo_weather_from_source() -> list[StateVar]:
    """Parse moves.ts for pseudoWeather declarations."""
    moves_ts = read("data/moves.ts")
    # Find all pseudoWeather values
    pw_names = set(extract_quoted(r"pseudoWeather:\s*'([^']+)'", moves_ts))

    # Manual annotations for duration and gen range
    pw_info = {
        "trickroom":   (5, 4, 9, False, "Inverts speed order"),
        "gravity":     (5, 4, 9, False, "Grounds all Pokemon, disables some moves"),
        "magicroom":   (5, 5, 9, False, "Suppresses all held items"),
        "wonderroom":  (5, 5, 9, False, "Swaps Def and SpD for all Pokemon"),
        "fairylock":   (2, 6, 9, False, "Prevents switching for 1 turn"),
        "iondeluge":   (1, 6, 7, False, "Normal->Electric; removed in Gen 8+"),
        "mudsport":    (5, 3, 6, False, "Weakens Electric; removed in Gen 7+"),
        "watersport":  (5, 3, 6, False, "Weakens Fire; removed in Gen 7+"),
    }

    results = []
    for name in sorted(pw_names):
        key = name.lower().replace(" ", "")
        if key in pw_info:
            dur, gmin, gmax, dbl, note = pw_info[key]
            results.append(StateVar(
                name, "pseudo_weather", "field",
                dur + 1,  # +1 for "off" state
                gen_min=gmin, gen_max=gmax, doubles_only=dbl,
                notes=f"0-{dur} turns; {note}",
            ))
        else:
            # Unknown pseudo-weather, assume binary
            results.append(StateVar(
                name, "pseudo_weather", "field", 2,
                notes="Unknown duration; found in source",
            ))

    return results


# ── 4. SIDE CONDITIONS ──────────────────────────────────────────────────────

def enumerate_side_conditions_from_source() -> list[StateVar]:
    """Parse moves.ts for sideCondition declarations + addSideCondition calls."""
    moves_ts = read("data/moves.ts")

    sc_from_field = set(extract_quoted(r"sideCondition:\s*'([^']+)'", moves_ts))
    sc_from_add = set(extract_quoted(r"addSideCondition\('([^']+)'", moves_ts))
    all_sc = sc_from_field | sc_from_add

    # Manual annotations: (num_values, gen_min, gen_max, doubles_only, notes)
    sc_info = {
        "stealthrock":    (2, 4, 9, False, "Binary entry hazard"),
        "spikes":         (4, 2, 9, False, "0-3 layers"),
        "toxicspikes":    (3, 4, 9, False, "0-2 layers"),
        "stickyweb":      (2, 6, 9, False, "Binary entry hazard"),
        "reflect":        (9, 2, 9, False, "0-8 turns (Light Clay extends)"),
        "lightscreen":    (9, 2, 9, False, "0-8 turns (Light Clay extends)"),
        "auroraveil":     (9, 7, 9, False, "0-8 turns (Light Clay extends)"),
        "tailwind":       (5, 5, 9, False, "0-4 turns"),
        "safeguard":      (6, 2, 9, False, "0-5 turns"),
        "mist":           (6, 1, 9, False, "0-5 turns; prevents stat drops"),
        "luckychant":     (6, 4, 7, False, "0-5 turns; anti-crit; removed Gen 8+"),
        "firepledge":     (5, 5, 9, True,  "0-4 turns; Sea of Fire (doubles combo)"),
        "waterpledge":    (5, 5, 9, True,  "0-4 turns; Rainbow (doubles combo)"),
        "grasspledge":    (5, 5, 9, True,  "0-4 turns; Swamp (doubles combo)"),
        "gmaxsteelsurge": (2, 8, 8, False, "Binary; Gen 8 G-Max only"),
        "gmaxcannonade":  (5, 8, 8, False, "0-4 turns; Gen 8 G-Max only"),
        "gmaxvinelash":   (5, 8, 8, False, "0-4 turns; Gen 8 G-Max only"),
        "gmaxvolcalith":  (5, 8, 8, False, "0-4 turns; Gen 8 G-Max only"),
        "gmaxwildfire":   (5, 8, 8, False, "0-4 turns; Gen 8 G-Max only"),
        "quickguard":     (2, 5, 9, False, "1-turn protect; mainly doubles"),
        "wideguard":      (2, 5, 9, False, "1-turn protect; mainly doubles"),
        "craftyshield":   (2, 6, 9, False, "1-turn protect; mainly doubles"),
        "matblock":       (2, 6, 9, True,  "1-turn; first turn only; doubles"),
    }

    results = []
    for name in sorted(all_sc):
        key = name.lower().replace(" ", "").replace("-", "")
        if key in sc_info:
            nv, gmin, gmax, dbl, note = sc_info[key]
            results.append(StateVar(
                name, "side_condition", "per_side", nv,
                gen_min=gmin, gen_max=gmax, doubles_only=dbl,
                notes=note,
            ))
        else:
            results.append(StateVar(
                name, "side_condition", "per_side", 2,
                notes=f"Unknown; found in source as '{name}'",
            ))

    return results


# ── 5. SLOT CONDITIONS ──────────────────────────────────────────────────────

def enumerate_slot_conditions() -> list[StateVar]:
    return [
        StateVar("Wish", "slot_condition", "per_slot", 2, gen_min=3, notes="Pending heal next turn"),
        StateVar("Future Sight / Doom Desire", "slot_condition", "per_slot", 3, gen_min=2, notes="0-2 turns until hit"),
        StateVar("Healing Wish", "slot_condition", "per_slot", 2, gen_min=4, notes="Triggers on switch-in"),
        StateVar("Lunar Dance", "slot_condition", "per_slot", 2, gen_min=4, notes="Triggers on switch-in"),
        StateVar("Revival Blessing", "slot_condition", "per_slot", 2, gen_min=9, notes="Revives fainted Pokemon"),
    ]


# ── 6. VOLATILE STATUSES ────────────────────────────────────────────────────

def enumerate_volatiles_from_source() -> list[StateVar]:
    """Parse moves.ts and conditions.ts for all volatile statuses."""
    moves_ts = read("data/moves.ts")
    conditions_ts = read("data/conditions.ts")
    abilities_ts = read("data/abilities.ts")

    # Collect all volatile names from source
    vol_from_field = set(extract_quoted(r"volatileStatus:\s*'([^']+)'", moves_ts))
    vol_from_add_moves = set(extract_quoted(r"addVolatile\('([^']+)'", moves_ts))
    vol_from_add_cond = set(extract_quoted(r"addVolatile\('([^']+)'", conditions_ts))
    vol_from_add_abil = set(extract_quoted(r"addVolatile\('([^']+)'", abilities_ts))

    all_vol = vol_from_field | vol_from_add_moves | vol_from_add_cond | vol_from_add_abil

    # Manual annotations: (num_values, gen_min, gen_max, doubles_only, scope, notes)
    # num_values includes "off" state
    vol_info = {
        # -- Core combat volatiles --
        "confusion":          (5, 1, 9, False, "per_active", "0 or 2-5 turns"),
        "attract":            (2, 2, 9, False, "per_active", "Binary; Infatuation"),
        "flinch":             (2, 1, 9, False, "per_active", "1-turn; prevents action"),
        "partiallytrapped":   (8, 1, 9, False, "per_active", "0 or 1-7 turns (Grip Claw=7)"),
        "substitute":         (2, 1, 9, False, "per_active", "Binary (HP tracked separately)"),
        "leechseed":          (2, 1, 9, False, "per_active", "Binary"),
        "taunt":              (5, 3, 9, False, "per_active", "0-4 turns (Gen 5+: 3 turns)"),
        "encore":             (4, 2, 9, False, "per_active", "0-3 turns"),
        "disable":            (5, 1, 9, False, "per_active", "0-4 turns + which move disabled"),
        "torment":            (2, 3, 9, False, "per_active", "Binary"),
        "healblock":          (6, 4, 9, False, "per_active", "0-5 turns"),
        "embargo":            (6, 4, 9, False, "per_active", "0-5 turns"),
        "aquaring":           (2, 4, 9, False, "per_active", "Binary"),
        "ingrain":            (2, 3, 9, False, "per_active", "Binary"),
        "focusenergy":        (2, 1, 9, False, "per_active", "Binary; +2 crit stages"),
        "yawn":               (3, 3, 9, False, "per_active", "0-2 turns pending sleep"),
        "curse":              (2, 2, 9, False, "per_active", "Binary; Ghost-type Curse"),
        "nightmare":          (2, 2, 9, False, "per_active", "Binary; damage during sleep"),
        "perishsong":         (4, 2, 9, False, "per_active", "0-3 turn counter"),
        "destinybond":        (2, 2, 9, False, "per_active", "Binary; 1-turn"),
        "grudge":             (2, 3, 9, False, "per_active", "Binary"),
        "octolock":           (2, 8, 9, False, "per_active", "Binary; traps + stat drop each turn"),
        "noretreat":          (2, 8, 9, False, "per_active", "Binary; prevents switching"),
        "tarshot":            (2, 8, 9, False, "per_active", "Binary; adds Fire weakness"),
        "saltcure":           (2, 9, 9, False, "per_active", "Binary; residual damage"),
        "syrupbomb":          (4, 9, 9, False, "per_active", "0-3 turns; lowers Speed"),

        # -- Move lock / charge / recharge --
        "lockedmove":         (4, 1, 9, False, "per_active", "0 or 1-3 turns (Outrage/Thrash/Petal Dance)"),
        "twoturnmove":        (2, 1, 9, False, "per_active", "Binary; charging/invulnerable (Fly/Dig/Dive/Bounce/etc.)"),
        "mustrecharge":       (2, 1, 9, False, "per_active", "Binary; Hyper Beam recharge"),
        "bide":               (3, 1, 9, False, "per_active", "0-2 turns; accumulating damage"),
        "uproar":             (4, 3, 9, False, "per_active", "0-3 turns"),
        "rollout":            (6, 2, 9, False, "per_active", "0-5 hit counter (Rollout/Ice Ball)"),
        "iceball":            (6, 3, 9, False, "per_active", "0-5 hit counter"),
        "furycutter":         (4, 2, 9, False, "per_active", "0-3 power doublings"),

        # -- Protection moves --
        "protect":            (2, 2, 9, False, "per_active", "1-turn shield"),
        "banefulbunker":      (2, 7, 9, False, "per_active", "1-turn; poisons on contact"),
        "burningbulwark":     (2, 9, 9, False, "per_active", "1-turn; burns on contact"),
        "kingsshield":        (2, 6, 9, False, "per_active", "1-turn; lowers Atk on contact"),
        "obstruct":           (2, 8, 9, False, "per_active", "1-turn; lowers Def on contact"),
        "silktrap":           (2, 9, 9, False, "per_active", "1-turn; lowers Speed on contact"),
        "spikyshield":        (2, 6, 9, False, "per_active", "1-turn; damages on contact"),
        "endure":             (2, 2, 9, False, "per_active", "1-turn; survive at 1 HP"),
        "stall":              (2, 2, 9, False, "per_active", "Protect success rate tracker"),

        # -- Type-changing effects --
        "typechange":         (19, 3, 9, False, "per_active", "Soak/Burn Up/etc. change type; 18 types + typeless"),
        "typeadd":            (19, 6, 9, False, "per_active", "Forest's Curse/Trick-or-Treat add a type"),

        # -- Ability/stat modification --
        "gastroacid":         (2, 4, 9, False, "per_active", "Binary; suppresses ability"),
        "powertrick":         (2, 4, 9, False, "per_active", "Binary; swaps Atk and Def"),
        "transform":          (2, 1, 9, False, "per_active", "Binary; copies target completely"),
        "flashfire":          (2, 3, 9, False, "per_active", "Binary; ability-triggered Fire boost"),
        "unburden":           (2, 3, 9, False, "per_active", "Binary; ability-triggered Speed double"),
        "slowstart":          (6, 4, 9, False, "per_active", "0-5 turns; halves Atk/Spe"),
        "protosynthesis":     (2, 9, 9, False, "per_active", "Binary; Paradox ability boost"),
        "quarkdrive":         (2, 9, 9, False, "per_active", "Binary; Paradox ability boost"),

        # -- Identification / grounding --
        "foresight":          (2, 2, 9, False, "per_active", "Binary; removes Ghost immunity"),
        "miracleeye":         (2, 4, 9, False, "per_active", "Binary; removes Dark immunity"),
        "smackdown":          (2, 5, 9, False, "per_active", "Binary; grounds target"),
        "telekinesis":        (4, 5, 9, False, "per_active", "0-3 turns; levitate but always hit"),
        "magnetrise":         (6, 4, 9, False, "per_active", "0-5 turns; levitate"),
        "charge":             (2, 3, 9, False, "per_active", "Binary; doubles next Electric move"),

        # -- Misc combat --
        "stockpile":          (4, 3, 9, False, "per_active", "0-3 layers"),
        "minimize":           (2, 1, 9, False, "per_active", "Binary; affects Stomp/etc. damage"),
        "defensecurl":        (2, 2, 9, False, "per_active", "Binary; doubles Rollout power"),
        "imprison":           (2, 3, 9, False, "per_active", "Binary; blocks shared moves"),
        "laserfocus":         (2, 7, 9, False, "per_active", "1-turn; guarantees crit"),
        "glaiverush":         (2, 9, 9, False, "per_active", "Binary; doubles damage taken"),
        "roost":              (2, 4, 9, False, "per_active", "1-turn; removes Flying type"),
        "powder":             (2, 6, 9, False, "per_active", "1-turn; punishes Fire moves"),
        "electrify":          (2, 6, 9, False, "per_active", "1-turn; makes target's move Electric"),
        "rage":               (2, 1, 9, False, "per_active", "Binary; boosts Atk when hit"),
        "snatch":             (2, 3, 9, False, "per_active", "1-turn; steals support moves"),
        "magiccoat":          (2, 3, 9, False, "per_active", "1-turn; reflects status moves"),

        # -- Doubles-specific --
        "helpinghand":        (2, 3, 9, True, "per_active", "1-turn; boosts ally damage"),
        "followme":           (2, 3, 9, True, "per_active", "1-turn; redirects attacks"),
        "ragepowder":         (2, 5, 9, True, "per_active", "1-turn; redirects attacks"),
        "spotlight":          (2, 7, 9, True, "per_active", "1-turn; redirects attacks"),
        "dragoncheer":        (2, 9, 9, True, "per_active", "Binary; crit boost (stronger for Dragon types)"),
        "allyswitch":         (2, 5, 9, True, "per_active", "1-turn; swaps positions"),

        # -- Ability-triggered (commander, etc.) --
        "commanding":         (2, 9, 9, True, "per_active", "Commander ability; inside Dondozo"),
        "commanded":          (2, 9, 9, True, "per_active", "Commander ability; Dondozo hosting"),
    }

    results = []
    seen = set()
    for name in sorted(all_vol):
        key = name.lower().replace(" ", "").replace("-", "")
        if key in seen:
            continue
        seen.add(key)
        if key in vol_info:
            nv, gmin, gmax, dbl, scope, note = vol_info[key]
            results.append(StateVar(
                name, "volatile", scope, nv,
                gen_min=gmin, gen_max=gmax, doubles_only=dbl,
                notes=note, source=name,
            ))
        # else: skip unknown/internal volatiles (e.g., move-tracking helpers)

    # Add manually-known volatiles not found by regex (internal names differ)
    manual_additions = {k: v for k, v in vol_info.items()
                        if k not in seen and k not in {n.lower().replace(" ", "").replace("-", "") for n in all_vol}}
    for key, (nv, gmin, gmax, dbl, scope, note) in manual_additions.items():
        results.append(StateVar(
            key, "volatile", scope, nv,
            gen_min=gmin, gen_max=gmax, doubles_only=dbl,
            notes=note, source="manual",
        ))

    return results


# ── 7. NON-VOLATILE STATUSES ────────────────────────────────────────────────

def enumerate_nonvolatile() -> list[StateVar]:
    return [
        StateVar("Healthy", "nonvolatile_status", "per_pokemon", 1, notes="No status"),
        StateVar("Burn", "nonvolatile_status", "per_pokemon", 1, gen_min=1),
        StateVar("Freeze", "nonvolatile_status", "per_pokemon", 1, gen_min=1, gen_max=8, notes="Replaced by Frostbite in Legends Arceus but still in Gen 9 OU"),
        StateVar("Paralysis", "nonvolatile_status", "per_pokemon", 1, gen_min=1),
        StateVar("Poison", "nonvolatile_status", "per_pokemon", 1, gen_min=1),
        StateVar("Toxic", "nonvolatile_status", "per_pokemon", 1, gen_min=1, notes="Escalating damage; turn counter matters"),
        StateVar("Sleep", "nonvolatile_status", "per_pokemon", 3, gen_min=1, notes="1-3 turn counter (Gen 5+)"),
    ]


# ── 8. PER-POKEMON PERSISTENT STATE ─────────────────────────────────────────

def enumerate_per_pokemon() -> list[StateVar]:
    """State that persists across switches for each of 12 Pokemon."""
    return [
        StateVar("HP", "per_pokemon", "per_pokemon", 301, notes="0-300 (approx max HP); (A)"),
        StateVar("Fainted", "per_pokemon", "per_pokemon", 2, notes="Binary"),
        StateVar("PP (per move slot)", "per_pokemon", "per_pokemon", 65,
                 notes="0-64 PP per move (max PP with PP Max); 4 slots per Pokemon"),
        StateVar("Item consumed", "per_pokemon", "per_pokemon", 3, gen_min=2,
                 notes="Item present / consumed / knocked off"),
        StateVar("Ability changed", "per_pokemon", "per_pokemon", 2, gen_min=3,
                 notes="Binary; changed via Skill Swap/Mummy/Wandering Spirit/etc."),
        StateVar("Toxic counter", "per_pokemon", "per_pokemon", 16, gen_min=1, gen_max=1,
                 notes="Gen 1 only: toxic counter 0-15, persists through Rest"),
    ]


# ── 9. PER-ACTIVE STATE ─────────────────────────────────────────────────────

def enumerate_per_active() -> list[StateVar]:
    return [
        StateVar("Stat stages (7 stats)", "stat_stage", "per_active", 13**7,
                 notes="Atk/Def/SpA/SpD/Spe/Acc/Eva each in [-6,+6] = 13 values"),
        StateVar("Last move used", "per_active", "per_active", 1,
                 notes="Tracked for Encore/Disable/etc. but combinatorial, omitted"),
    ]


# ── 10. GEN 1 SPECIFIC VOLATILES ────────────────────────────────────────────

def enumerate_gen1_volatiles() -> list[StateVar]:
    """Gen 1 has unique mechanics not in later gens."""
    gen1_cond = read("data/mods/gen1/conditions.ts")
    gen1_moves = read("data/mods/gen1/moves.ts")

    return [
        StateVar("partialtrappinglock", "volatile", "per_active", 5,
                 gen_min=1, gen_max=1, notes="Attacker locked into Wrap/Bind/etc; 2-5 turns"),
        StateVar("invulnerability", "volatile", "per_active", 2,
                 gen_min=1, gen_max=1, notes="Dig/Fly semi-invulnerable turn"),
        StateVar("residualdmg", "volatile", "per_active", 16,
                 gen_min=1, gen_max=1, notes="Toxic counter 0-15; unique Gen 1 mechanic"),
        StateVar("Reflect (per-Pokemon)", "volatile", "per_active", 2,
                 gen_min=1, gen_max=1, notes="Gen 1: Reflect is per-Pokemon volatile, not side condition"),
        StateVar("Light Screen (per-Pokemon)", "volatile", "per_active", 2,
                 gen_min=1, gen_max=1, notes="Gen 1: Light Screen is per-Pokemon volatile, not side condition"),
        StateVar("Mist (per-Pokemon)", "volatile", "per_active", 2,
                 gen_min=1, gen_max=1, notes="Gen 1: Mist is per-Pokemon volatile"),
        StateVar("Focus Energy (Gen 1)", "volatile", "per_active", 2,
                 gen_min=1, gen_max=1, notes="Gen 1: bugged, actually reduces crit rate"),
    ]


# ── COMPUTE STATE SPACES ────────────────────────────────────────────────────

def compute_gen9_ou(all_vars: list[StateVar]) -> dict:
    """Compute Gen 9 OU (singles) battle state space."""
    # Filter to Gen 9, singles
    gen9 = [v for v in all_vars if v.gen_min <= 9 <= v.gen_max and not v.doubles_only]

    components = {}

    # Team config (from paper): 10^215 per player, squared
    components["Team config (2 players)"] = (10**215)**2

    # Active Pokemon: 6 choices per side
    components["Active Pokemon"] = 6 * 6

    # Weather: None + 4 standard × 8 turns + 3 primal = 36
    weather_states = 1  # None
    for v in gen9:
        if v.category == "weather" and v.name != "No Weather":
            weather_states += v.num_values
    components["Weather"] = weather_states

    # Terrain: None + 4 × 8 = 33
    terrain_states = 1
    for v in gen9:
        if v.category == "terrain" and v.name != "No Terrain":
            terrain_states += v.num_values
    components["Terrain"] = terrain_states

    # Pseudo-weather: product of all (each independent)
    pw = 1
    pw_detail = {}
    for v in gen9:
        if v.category == "pseudo_weather":
            pw *= v.num_values
            pw_detail[v.name] = v.num_values
    components["Pseudo-weather"] = pw

    # Side conditions: product of all, squared (both sides)
    sc = 1
    sc_detail = {}
    for v in gen9:
        if v.category == "side_condition":
            sc *= v.num_values
            sc_detail[v.name] = v.num_values
    components["Side conditions (per side)"] = sc
    components["Side conditions (both sides)"] = sc ** 2

    # Slot conditions: product, per slot, 6 slots per side, 2 sides
    slot = 1
    for v in gen9:
        if v.category == "slot_condition":
            slot *= v.num_values
    components["Slot conditions (per slot)"] = slot
    components["Slot conditions (all 12 slots)"] = slot ** 12

    # Volatiles: product of all per-active, squared (2 active Pokemon)
    vol = 1
    vol_detail = {}
    for v in gen9:
        if v.category == "volatile" and v.scope == "per_active":
            vol *= v.num_values
            vol_detail[v.name] = v.num_values
    components["Volatiles (per active)"] = vol
    components["Volatiles (2 actives)"] = vol ** 2

    # HP: 301 states per Pokemon, 12 Pokemon
    components["HP (12 Pokemon)"] = 301 ** 12

    # Non-volatile status: 9 states per Pokemon (healthy + brn + frz + par + psn + toxic + sleep×3)
    nv_states = sum(v.num_values for v in enumerate_nonvolatile() if v.gen_min <= 9 <= v.gen_max)
    components["Non-volatile status (per Pokemon)"] = nv_states
    components["Non-volatile status (12 Pokemon)"] = nv_states ** 12

    # Stat stages: 13^7 per active, 2 actives
    components["Stat stages (2 actives)"] = (13**7) ** 2

    # Terastallization: (1 + 6) per side = 7, squared
    components["Terastallization"] = 7 ** 2

    # Per-Pokemon persistent: PP (65^4 per Pokemon × 12), item state (3^12), ability change (2^12)
    components["PP (4 moves × 12 Pokemon)"] = 65 ** (4 * 12)
    components["Item state (12 Pokemon)"] = 3 ** 12
    components["Ability changed (12 Pokemon)"] = 2 ** 12

    # Total
    total_log10 = sum(math.log10(v) for v in components.values() if v > 0)

    return {
        "components": {k: f"10^{math.log10(v):.1f}" if v > 1 else "1" for k, v in components.items()},
        "total_log10": total_log10,
        "total_approx": f"~10^{total_log10:.0f}",
        "volatile_count": len(vol_detail),
        "volatile_detail": vol_detail,
        "pseudo_weather_detail": pw_detail,
        "side_condition_detail": sc_detail,
    }


def compute_gen1_ou(all_vars: list[StateVar]) -> dict:
    """Compute Gen 1 OU battle state space."""
    gen1 = [v for v in all_vars if v.gen_min <= 1 <= v.gen_max]

    components = {}

    # Team config: 10^57 per player
    components["Team config (2 players)"] = (10**57)**2

    # Active Pokemon
    components["Active Pokemon"] = 36

    # No weather, terrain, pseudo-weather, or side conditions in Gen 1
    # (Mist, Reflect, Light Screen are per-Pokemon volatiles in Gen 1)

    # Volatiles: Gen 1 specific
    vol = 1
    vol_detail = {}
    gen1_vols = [v for v in gen1 if v.category == "volatile" and v.scope == "per_active"
                 and not (v.gen_min > 1)]  # Only include things that exist in Gen 1
    for v in gen1_vols:
        vol *= v.num_values
        vol_detail[v.name] = v.num_values
    components["Volatiles (per active)"] = vol
    components["Volatiles (2 actives)"] = vol ** 2

    # HP
    components["HP (12 Pokemon)"] = 301 ** 12

    # Non-volatile: Gen 1 has different sleep (1-7 turns) and toxic (counter 1-15)
    # Healthy(1) + BRN(1) + FRZ(1) + PAR(1) + PSN(1) + Toxic(15) + SLP(7) = 27
    nv_per_poke = 27
    components["Non-volatile status (per Pokemon)"] = nv_per_poke
    components["Non-volatile status (12 Pokemon)"] = nv_per_poke ** 12

    # Stat stages
    components["Stat stages (2 actives)"] = (13**7) ** 2

    # PP
    components["PP (4 moves × 12 Pokemon)"] = 65 ** (4 * 12)

    total_log10 = sum(math.log10(v) for v in components.values() if v > 0)

    return {
        "components": {k: f"10^{math.log10(v):.1f}" if v > 1 else "1" for k, v in components.items()},
        "total_log10": total_log10,
        "total_approx": f"~10^{total_log10:.0f}",
        "volatile_count": len(vol_detail),
        "volatile_detail": vol_detail,
    }


def compute_gen9_vgc(all_vars: list[StateVar]) -> dict:
    """Compute Gen 9 VGC (doubles) battle state space."""
    gen9 = [v for v in all_vars if v.gen_min <= 9 <= v.gen_max and not v.singles_only]

    components = {}

    components["Team config (2 players)"] = (10**215)**2

    # Team preview: C(6,4) per side
    components["Team preview selection"] = 15 ** 2

    # Active positions: P(4,2) per side = 12, squared
    components["Active positions"] = 12 ** 2

    # Weather
    weather_states = 1
    for v in gen9:
        if v.category == "weather" and v.name != "No Weather":
            weather_states += v.num_values
    components["Weather"] = weather_states

    # Terrain
    terrain_states = 1
    for v in gen9:
        if v.category == "terrain" and v.name != "No Terrain":
            terrain_states += v.num_values
    components["Terrain"] = terrain_states

    # Pseudo-weather
    pw = 1
    for v in gen9:
        if v.category == "pseudo_weather":
            pw *= v.num_values
    components["Pseudo-weather"] = pw

    # Side conditions (including doubles-specific ones)
    sc = 1
    for v in gen9:
        if v.category == "side_condition":
            sc *= v.num_values
    components["Side conditions (both sides)"] = sc ** 2

    # Slot conditions (8 slots: 4 per side × 2 sides)
    slot = 1
    for v in gen9:
        if v.category == "slot_condition":
            slot *= v.num_values
    components["Slot conditions (all 8 slots)"] = slot ** 8

    # Volatiles (4 active Pokemon)
    vol = 1
    for v in gen9:
        if v.category == "volatile" and v.scope == "per_active":
            vol *= v.num_values
    components["Volatiles (4 actives)"] = vol ** 4

    # HP: 8 Pokemon participate
    components["HP (8 Pokemon)"] = 301 ** 8

    # Non-volatile status: 8 Pokemon
    nv_states = sum(v.num_values for v in enumerate_nonvolatile() if v.gen_min <= 9 <= v.gen_max)
    components["Non-volatile status (8 Pokemon)"] = nv_states ** 8

    # Stat stages: 4 actives
    components["Stat stages (4 actives)"] = (13**7) ** 4

    # Tera: (1 + 4) per side (bring 4 of 6)
    components["Terastallization"] = 5 ** 2

    # PP: 8 Pokemon × 4 moves
    components["PP (4 moves × 8 Pokemon)"] = 65 ** (4 * 8)

    # Item state, ability change: 8 Pokemon
    components["Item state (8 Pokemon)"] = 3 ** 8
    components["Ability changed (8 Pokemon)"] = 2 ** 8

    total_log10 = sum(math.log10(v) for v in components.values() if v > 0)

    return {
        "components": {k: f"10^{math.log10(v):.1f}" if v > 1 else "1" for k, v in components.items()},
        "total_log10": total_log10,
        "total_approx": f"~10^{total_log10:.0f}",
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("POKEMON SHOWDOWN BATTLE STATE ENUMERATION")
    print("=" * 80)

    # Collect all state variables
    all_vars = []
    all_vars.extend(enumerate_weather())
    all_vars.extend(enumerate_terrain())
    all_vars.extend(enumerate_pseudo_weather_from_source())
    all_vars.extend(enumerate_side_conditions_from_source())
    all_vars.extend(enumerate_slot_conditions())
    all_vars.extend(enumerate_volatiles_from_source())
    all_vars.extend(enumerate_nonvolatile())
    all_vars.extend(enumerate_per_pokemon())
    all_vars.extend(enumerate_per_active())
    all_vars.extend(enumerate_gen1_volatiles())

    # Print summary by category
    by_cat = defaultdict(list)
    for v in all_vars:
        by_cat[v.category].append(v)

    for cat in ["weather", "terrain", "pseudo_weather", "side_condition",
                "slot_condition", "volatile", "nonvolatile_status",
                "per_pokemon", "per_active", "stat_stage"]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        print(f"\n{'─' * 80}")
        print(f"  {cat.upper().replace('_', ' ')} ({len(items)} entries)")
        print(f"{'─' * 80}")
        for v in items:
            gen_str = f"Gen {v.gen_min}-{v.gen_max}" if v.gen_min != v.gen_max else f"Gen {v.gen_min}"
            dbl_str = " [doubles]" if v.doubles_only else ""
            print(f"  {v.name:<35s}  values={v.num_values:<6d}  {gen_str:<10s}{dbl_str}  {v.notes}")

    # Source extraction stats
    moves_ts = read("data/moves.ts")
    conditions_ts = read("data/conditions.ts")
    abilities_ts = read("data/abilities.ts")

    vol_from_moves = set(extract_quoted(r"volatileStatus:\s*'([^']+)'", moves_ts))
    vol_from_add = set(extract_quoted(r"addVolatile\('([^']+)'", moves_ts))
    vol_from_cond = set(extract_quoted(r"addVolatile\('([^']+)'", conditions_ts))
    vol_from_abil = set(extract_quoted(r"addVolatile\('([^']+)'", abilities_ts))
    all_source_vol = vol_from_moves | vol_from_add | vol_from_cond | vol_from_abil

    annotated_vol = {v.source for v in all_vars if v.category == "volatile" and v.source}
    unannotated = all_source_vol - annotated_vol - {v.name for v in all_vars if v.category == "volatile"}

    print(f"\n{'=' * 80}")
    print(f"SOURCE EXTRACTION STATS")
    print(f"{'=' * 80}")
    print(f"  Total volatiles found in source: {len(all_source_vol)}")
    print(f"  Annotated in our enumeration:    {len(annotated_vol)}")
    print(f"  Unannotated (possibly internal): {len(unannotated)}")
    if unannotated:
        print(f"  Unannotated names: {sorted(unannotated)}")

    # Compute state spaces
    print(f"\n{'=' * 80}")
    print(f"GEN 9 OU STATE SPACE")
    print(f"{'=' * 80}")
    gen9_ou = compute_gen9_ou(all_vars)
    for k, v in gen9_ou["components"].items():
        print(f"  {k:<45s}  {v}")
    print(f"\n  TOTAL: {gen9_ou['total_approx']}")
    print(f"  (Exact log10: {gen9_ou['total_log10']:.1f})")
    print(f"  Volatiles counted: {gen9_ou['volatile_count']}")

    print(f"\n{'=' * 80}")
    print(f"GEN 1 OU STATE SPACE")
    print(f"{'=' * 80}")
    gen1_ou = compute_gen1_ou(all_vars)
    for k, v in gen1_ou["components"].items():
        print(f"  {k:<45s}  {v}")
    print(f"\n  TOTAL: {gen1_ou['total_approx']}")
    print(f"  (Exact log10: {gen1_ou['total_log10']:.1f})")

    print(f"\n{'=' * 80}")
    print(f"GEN 9 VGC STATE SPACE")
    print(f"{'=' * 80}")
    gen9_vgc = compute_gen9_vgc(all_vars)
    for k, v in gen9_vgc["components"].items():
        print(f"  {k:<45s}  {v}")
    print(f"\n  TOTAL: {gen9_vgc['total_approx']}")
    print(f"  (Exact log10: {gen9_vgc['total_log10']:.1f})")

    # Comparison with current paper values
    print(f"\n{'=' * 80}")
    print(f"COMPARISON WITH CURRENT PAPER")
    print(f"{'=' * 80}")
    print(f"  {'Format':<20s} {'Paper':>12s} {'Script':>12s} {'Delta':>10s}")
    print(f"  {'─' * 56}")
    for name, paper_val, script_val in [
        ("Gen 1 OU", 183, gen1_ou["total_log10"]),
        ("Gen 9 OU", 518, gen9_ou["total_log10"]),
        ("Gen 9 VGC", 535, gen9_vgc["total_log10"]),
    ]:
        delta = script_val - paper_val
        print(f"  {name:<20s} {'10^' + str(paper_val):>12s} {'10^' + f'{script_val:.0f}':>12s} {'+' + f'{delta:.0f}' if delta >= 0 else f'{delta:.0f}':>10s}")

    # Save JSON
    output = {
        "all_variables": [asdict(v) for v in all_vars],
        "gen9_ou": gen9_ou,
        "gen1_ou": gen1_ou,
        "gen9_vgc": gen9_vgc,
    }
    out_path = Path(__file__).parent / "battle_state_variables.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
