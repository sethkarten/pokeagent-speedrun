"""
verify_state_space.py — Reproducible verification of Pokémon state-space derivations.

All numbers sourced from Pokémon Showdown data files in pokemon-showdown/data/.
Run from the project root:
    uv run --with numpy python figures/scripts/verify_state_space.py

Cross-references all values used in Appendix H of the paper.
"""

import re
import math
from pathlib import Path

DATA_DIR = Path("pokemon-showdown/data")


def c(n, k):
    return math.comb(n, k)

def l(x):
    return math.log10(x)


def parse_entries(filepath):
    """
    Parse top-level entries from a Showdown TypeScript data file.
    Returns list of dicts with parsed fields.
    """
    text = Path(filepath).read_text(encoding='utf-8')
    lines = text.split('\n')
    entries = []
    i = 0
    while i < len(lines):
        m = re.match(r'^\t(\w+):\s*\{', lines[i])
        if m:
            key = m.group(1)
            body_lines = [lines[i]]
            depth = lines[i].count('{') - lines[i].count('}')
            i += 1
            while i < len(lines) and depth > 0:
                body_lines.append(lines[i])
                depth += lines[i].count('{') - lines[i].count('}')
                i += 1
            body = '\n'.join(body_lines)
            entry = {'key': key, 'body': body}

            num_m = re.search(r'\bnum:\s*(-?\d+)', body)
            if num_m:
                entry['num'] = int(num_m.group(1))

            gen_m = re.search(r'\bgen:\s*(\d+)', body)
            if gen_m:
                entry['gen'] = int(gen_m.group(1))

            ns_m = re.search(r'isNonstandard:\s*["\']([^"\']+)', body)
            if ns_m:
                entry['isNonstandard'] = ns_m.group(1)

            name_m = re.search(r'\bname:\s*"([^"]+)"', body)
            if name_m:
                entry['name'] = name_m.group(1)

            forme_m = re.search(r'\bforme:\s*"([^"]+)"', body)
            if forme_m:
                entry['forme'] = forme_m.group(1)

            if 'baseSpecies' in body:
                entry['hasBase'] = True

            entries.append(entry)
        else:
            i += 1
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# 1. Species/formes per generation
# ─────────────────────────────────────────────────────────────────────────────

def count_species():
    """
    Count usable species+formes per generation from pokedex.ts.

    Gen 7 (USUM): all entries with num 1-807 (Gens 1-7), excluding Gen 8/9-only
    formes (Gigantamax, Galarian, Hisuian, Paldean, etc.).

    Gen 9: all entries with num > 0 and no isNonstandard flag.
    """
    entries = parse_entries(DATA_DIR / "pokedex.ts")
    valid = [e for e in entries if e.get('num', 0) > 0 and 'isNonstandard' not in e]

    # Gen 8/9-only forme keywords (not valid in USUM)
    gen89_keywords = ['gmax', 'galar', 'hisui', 'paldea', 'eternamax',
                      'totem', 'bloodmoon']

    def is_gen89_only(entry):
        forme = entry.get('forme', '')
        if not forme:
            return False
        f = forme.lower()
        return any(kw in f for kw in gen89_keywords)

    gen7 = [e for e in valid if e.get('num', 0) <= 807 and not is_gen89_only(e)]
    gen9 = valid  # all standard entries

    print(f"\n=== Species/Formes ===")
    print(f"Gen 7 (USUM) entries: {len(gen7)}")
    print(f"  Base Pokemon (num 1-807): {sum(1 for e in gen7 if not e.get('hasBase'))}")
    print(f"  Alternate formes: {sum(1 for e in gen7 if e.get('hasBase'))}")
    print(f"Gen 9 entries: {len(gen9)}")
    print(f"  Base Pokemon (num 1-1025): {sum(1 for e in gen9 if not e.get('hasBase'))}")
    print(f"  Alternate formes: {sum(1 for e in gen9 if e.get('hasBase'))}")
    print(f"  C({len(gen7)}, 6) = {c(len(gen7), 6):,}  log10={l(c(len(gen7), 6)):.2f}")
    print(f"  C({len(gen9)}, 6) = {c(len(gen9), 6):,}  log10={l(c(len(gen9), 6)):.2f}")
    return len(gen7), len(gen9)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Items per generation
# ─────────────────────────────────────────────────────────────────────────────

def count_items():
    """
    Count valid held items per generation from items.ts.

    Gen 7: entries with gen <= 7 and isNonstandard in (None, 'Past').
    'Past' items were valid in Gen 7 but have since been deprecated (Mega Stones,
    Z-Crystals, etc.).

    Gen 9: entries with no isNonstandard flag.
    """
    entries = parse_entries(DATA_DIR / "items.ts")
    valid = [e for e in entries if e.get('num', 0) > 0
             and e.get('isNonstandard') not in ('Unobtainable', 'CAP')]

    gen7 = [e for e in valid
            if e.get('gen', 99) <= 7
            and e.get('isNonstandard') in (None, 'Past')]
    gen9 = [e for e in valid if 'isNonstandard' not in e]

    print(f"\n=== Items ===")
    print(f"Gen 7 (USUM) items: {len(gen7)}")
    print(f"  Still-standard today: {sum(1 for e in gen7 if 'isNonstandard' not in e)}")
    print(f"  Deprecated 'Past' (Mega Stones, Z-Crystals, etc.): "
          f"{sum(1 for e in gen7 if e.get('isNonstandard') == 'Past')}")
    print(f"Gen 9 items: {len(gen9)}")
    print(f"  {len(gen7)}^6 = {len(gen7)**6:.4e}  log10={l(len(gen7)**6):.2f}")
    print(f"  {len(gen9)}^6 = {len(gen9)**6:.4e}  log10={l(len(gen9)**6):.2f}")
    return len(gen7), len(gen9)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mew's movepool per generation
# ─────────────────────────────────────────────────────────────────────────────

def count_mew_moves():
    """
    Count Mew's learnable moves per generation from learnsets.ts.
    Source code format: first character is the generation number.
    """
    text = (DATA_DIR / "learnsets.ts").read_text(encoding='utf-8')

    idx = text.find('mew: {')
    if idx < 0:
        print("ERROR: Mew's learnset not found")
        return None, None

    # Find the end of Mew's learnset block
    end = text.find('\n\t},', idx)
    mew_block = text[idx:end + 4]

    move_entries = re.findall(r'(\w+):\s*\[([^\]]+)\]', mew_block)

    gen7_moves = set()
    gen9_moves = set()

    for move_name, sources_str in move_entries:
        sources = re.findall(r'"([^"]+)"', sources_str)
        for src in sources:
            if src and src[0].isdigit():
                g = int(src[0])
                if g <= 7:
                    gen7_moves.add(move_name)
                if g <= 9:
                    gen9_moves.add(move_name)

    print(f"\n=== Mew's Movepool ===")
    print(f"Mew Gen 7 (USUM) moves: {len(gen7_moves)}")
    print(f"Mew Gen 9 moves: {len(gen9_moves)}")
    print(f"  C({len(gen7_moves)}, 4) = {c(len(gen7_moves), 4):,}")
    print(f"  C({len(gen7_moves)}, 4)^6 = {c(len(gen7_moves), 4)**6:.4e}  "
          f"log10={l(c(len(gen7_moves), 4)**6):.2f}")
    print(f"  C({len(gen9_moves)}, 4) = {c(len(gen9_moves), 4):,}")
    print(f"  C({len(gen9_moves)}, 4)^6 = {c(len(gen9_moves), 4)**6:.4e}  "
          f"log10={l(c(len(gen9_moves), 4)**6):.2f}")
    return len(gen7_moves), len(gen9_moves)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gen 1 total moves
# ─────────────────────────────────────────────────────────────────────────────

def count_gen1_moves():
    """
    Count all Generation 1 moves from moves.ts.
    Gen 1 moves have num in [1, 165] (original 165 moves from RBY).
    Includes moves now marked 'Past' (deprecated in current games but valid in Gen 1).
    """
    entries = parse_entries(DATA_DIR / "moves.ts")
    gen1 = [e for e in entries
            if 1 <= e.get('num', 0) <= 165
            and e.get('isNonstandard') not in ('Unobtainable', 'CAP', 'LGPE')]

    print(f"\n=== Gen 1 Moves ===")
    print(f"All Gen 1 moves (num 1-165): {len(gen1)}")
    print(f"  Still-standard: {sum(1 for e in gen1 if 'isNonstandard' not in e)}")
    print(f"  Deprecated 'Past': {sum(1 for e in gen1 if e.get('isNonstandard') == 'Past')}")
    print(f"  C({len(gen1)}, 4) = {c(len(gen1), 4):,}")
    print(f"  C({len(gen1)}, 4)^6 = {c(len(gen1), 4)**6:.4e}  log10={l(c(len(gen1), 4)**6):.2f}")
    return len(gen1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Natures
# ─────────────────────────────────────────────────────────────────────────────

def count_natures():
    entries = parse_entries(DATA_DIR / "natures.ts")
    # Filter to actual nature entries (exclude export lines etc.)
    natures = [e for e in entries if len(e['key']) > 2]
    print(f"\n=== Natures ===")
    print(f"Total natures: {len(natures)}")
    neutral = sum(1 for e in natures if 'plus' not in e['body'])
    print(f"Neutral natures: {neutral}")
    return len(natures)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EV exact count (mathematical, no data file needed)
# ─────────────────────────────────────────────────────────────────────────────

def ev_exact():
    """Exact EV spread count via inclusion-exclusion."""
    total = c(133, 6)
    one_over = 6 * c(69, 6)
    exact = total - one_over
    print(f"\n=== EV Spread Count ===")
    print(f"C(133,6) = {total:,}")
    print(f"6 * C(69,6) = {one_over:,}")
    print(f"Exact |E| = {exact:,}  log10={l(exact):.4f}")
    print(f"  |E|^6 = {exact**6:.4e}  log10={l(exact**6):.2f}")
    return exact


# ─────────────────────────────────────────────────────────────────────────────
# 7. Final team space calculations
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(sp7, sp9, mew7, mew9, it7, it9, gen1_moves, ev_count):
    """Compute and print all team space bounds."""
    natures = 25
    abilities = 3
    tera = 18
    iv_per_mon = 32**6

    print("\n" + "=" * 60)
    print("USUM (Gen 7) Team Space Upper Bound")
    print("=" * 60)
    comps7 = {
        f'Species C({sp7},6)': c(sp7, 6),
        f'Moves C({mew7},4)^6': c(mew7, 4)**6,
        'Abilities 3^6': abilities**6,
        'IVs 32^36': 32**36,
        f'EVs |E|^6': ev_count**6,
        'Natures 25^6': natures**6,
        f'Items {it7}^6': it7**6,
    }
    total7 = sum(l(v) for v in comps7.values())
    for name, val in comps7.items():
        print(f"  {name:25s}: log10 = {l(val):.2f}")
    print(f"  {'TOTAL':25s}: ~10^{total7:.1f}")

    print("\n" + "=" * 60)
    print("Gen 9 OU Team Space Upper Bound")
    print("=" * 60)
    comps9 = {
        f'Species C({sp9},6)': c(sp9, 6),
        f'Moves C({mew9},4)^6': c(mew9, 4)**6,
        'Abilities 3^6': abilities**6,
        'IVs 32^36': 32**36,
        f'EVs |E|^6': ev_count**6,
        'Natures 25^6': natures**6,
        f'Items {it9}^6': it9**6,
        'Tera 18^6': tera**6,
    }
    total9 = sum(l(v) for v in comps9.values())
    for name, val in comps9.items():
        print(f"  {name:25s}: log10 = {l(val):.2f}")
    print(f"  {'TOTAL':25s}: ~10^{total9:.1f}")

    print("\n" + "=" * 60)
    print("Gen 1 OU (RBY) Team Space Upper Bound")
    print("=" * 60)
    comps1 = {
        'Species C(151,6)': c(151, 6),
        f'Moves C({gen1_moves},4)^6': c(gen1_moves, 4)**6,
        'DVs 16^24': 16**24,
        'StatExp 256^30': 256**30,
    }
    total1 = sum(l(v) for v in comps1.values())
    for name, val in comps1.items():
        print(f"  {name:25s}: log10 = {l(val):.2f}")
    print(f"  {'TOTAL':25s}: ~10^{total1:.1f}")

    print("\n" + "=" * 60)
    print("Battle State Spaces")
    print("=" * 60)
    field = 36 * 33 * 2304 * 531441
    gen9_bs = (2*total9 + l(36) + l(300**12) + l(13**14) + l(field) + l(49))
    gen1_bs = (2*total1 + l(36) + l(300**12) + l(13**14) + l(27**12))
    print(f"Gen 9 OU battle state: ~10^{gen9_bs:.0f}")
    print(f"Gen 1 OU battle state: ~10^{gen1_bs:.0f}")
    print(f"Field exact: {field:,}  (log10={l(field):.2f})")

    return total7, total9, total1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Pokémon State-Space Verification")
    print("Source: pokemon-showdown/data/")
    print("=" * 60)

    sp7, sp9 = count_species()
    it7, it9 = count_items()
    mew7, mew9 = count_mew_moves()
    gen1_moves = count_gen1_moves()
    count_natures()
    ev = ev_exact()

    compute_all(sp7, sp9, mew7, mew9, it7, it9, gen1_moves, ev)
