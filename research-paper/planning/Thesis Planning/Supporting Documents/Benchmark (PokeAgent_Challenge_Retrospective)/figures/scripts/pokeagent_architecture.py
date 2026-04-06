#!/usr/bin/env python3
"""
PokeAgent Multi-Agent Architecture - NeurIPS Quality v9
Changes: Button icons instead of GBA, text left/image right for sub-agents,
simplified arrows, removed bottom loop arrow
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

C = {
    'env': ('#E3F2FD', '#1565C0'),
    'orch': ('#FFF3E0', '#E65100'),
    'agent': ('#E8F5E9', '#2E7D32'),
    'tool': ('#F3E5F5', '#7B1FA2'),
    'white': '#FFFFFF',
    'gray': '#F5F5F5',
    'border': '#78909C',
    'arrow': '#455A64',
    'text': '#263238',
}

SHOTS = "figures/assets/emerald_screenshots"

def img(ax, f, x, y, z=0.25):
    p = os.path.join(SHOTS, f)
    if os.path.exists(p):
        ab = AnnotationBbox(OffsetImage(mpimg.imread(p), zoom=z), (x, y),
                           frameon=True, bboxprops=dict(boxstyle='round,pad=0.003',
                           fc='white', ec='#888', lw=0.8))
        ax.add_artist(ab)

def box(ax, x, y, w, h, fc, ec, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                                facecolor=fc, edgecolor=ec, linewidth=lw))

def header(ax, x, y, w, label, colors):
    box(ax, x, y, w, 0.07, colors[1], colors[1], lw=0)
    ax.text(x + w/2, y + 0.035, label, ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')

def arr(ax, p1, p2, c=None, lw=2.5, rad=0, style='-', scale=14, outline=False):
    if outline:
        # Use simple arrow with white fill and thick black edge
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='simple,head_width=4,head_length=4',
                     connectionstyle=f"arc3,rad={rad}",
                     fc='white', ec='#263238', linewidth=3,
                     mutation_scale=4, zorder=2))
    else:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>',
                     connectionstyle=f"arc3,rad={rad}", linestyle=style,
                     color=c or C['arrow'], linewidth=lw, mutation_scale=scale))

def draw_gba_buttons(ax, x, y):
    """Draw GBA button layout with labels."""
    # D-pad section
    dpad_x, dpad_y = x + 0.08, y + 0.12
    # D-pad cross
    ax.add_patch(FancyBboxPatch((dpad_x - 0.022, dpad_y - 0.055), 0.044, 0.11,
                 boxstyle="round,pad=0.002,rounding_size=0.008",
                 fc='#546E7A', ec='#37474F', lw=1.5))
    ax.add_patch(FancyBboxPatch((dpad_x - 0.055, dpad_y - 0.022), 0.11, 0.044,
                 boxstyle="round,pad=0.002,rounding_size=0.008",
                 fc='#546E7A', ec='#37474F', lw=1.5))
    # D-pad arrows
    ax.text(dpad_x, dpad_y + 0.032, '▲', ha='center', va='center', fontsize=10, color='#B0BEC5')
    ax.text(dpad_x, dpad_y - 0.032, '▼', ha='center', va='center', fontsize=10, color='#B0BEC5')
    ax.text(dpad_x - 0.032, dpad_y, '◀', ha='center', va='center', fontsize=10, color='#B0BEC5')
    ax.text(dpad_x + 0.032, dpad_y, '▶', ha='center', va='center', fontsize=10, color='#B0BEC5')

    # A button (teal/green)
    ax.add_patch(Circle((x + 0.28, y + 0.14), 0.032, fc='#00897B', ec='#00695C', lw=2))
    ax.text(x + 0.28, y + 0.14, 'A', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # B button (red)
    ax.add_patch(Circle((x + 0.22, y + 0.08), 0.032, fc='#E53935', ec='#C62828', lw=2))
    ax.text(x + 0.22, y + 0.08, 'B', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Start button (gray pill)
    ax.add_patch(FancyBboxPatch((x + 0.20, y + 0.01), 0.06, 0.025,
                 boxstyle="round,pad=0.002,rounding_size=0.01",
                 fc='#78909C', ec='#546E7A', lw=1))
    ax.text(x + 0.23, y + 0.022, 'START', ha='center', va='center', fontsize=6, fontweight='bold', color='white')

    # Select button (gray pill)
    ax.add_patch(FancyBboxPatch((x + 0.12, y + 0.01), 0.06, 0.025,
                 boxstyle="round,pad=0.002,rounding_size=0.01",
                 fc='#78909C', ec='#546E7A', lw=1))
    ax.text(x + 0.15, y + 0.022, 'SELECT', ha='center', va='center', fontsize=6, fontweight='bold', color='white')

def main():
    # Content bounds: x=[0.02, 2.60], y=[0.05, 1.33]
    # Add equal margin of 0.04 on all sides
    margin = 0.04
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0.02 - margin, 2.60 + margin)  # -0.02 to 2.64
    ax.set_ylim(0.05 - margin, 1.33 + margin)  # 0.01 to 1.37
    ax.set_aspect('equal')
    ax.axis('off')

    # ========== COLUMN 1: GAME ENVIRONMENT ==========
    header(ax, 0.02, 1.26, 0.60, "Game Environment", C['env'])
    box(ax, 0.02, 0.05, 0.60, 1.19, C['env'][0], C['env'][1], lw=2)

    # Game screenshot - larger
    ax.text(0.07, 1.18, "Screen", ha='left', va='center', fontsize=14, fontweight='bold', color=C['text'])
    img(ax, "Pokemon_Emerald_(Eng)_012.png", 0.32, 0.97, z=0.68)

    # ASCII Map - centered
    ax.text(0.05, 0.70, "ASCII Map", ha='left', va='center', fontsize=14, fontweight='bold', color=C['text'])
    box(ax, 0.05, 0.42, 0.26, 0.26, C['gray'], C['border'], lw=1.2)
    ascii_map = "≈≈≈≈≈≈≈≈\n≈░░██░░≈\n≈░░░░░░≈\n≈░░@░░░≈\n≈░░░░██≈\n≈≈≈≈≈≈≈≈"
    ax.text(0.08, 0.65, ascii_map, ha='left', va='top', fontsize=12,
            fontfamily='monospace', color='#333', linespacing=0.95)
    ax.text(0.18, 0.44, "@ = player", ha='center', va='center', fontsize=11, color='#666')

    # Party Info - centered
    ax.text(0.34, 0.70, "Party", ha='left', va='center', fontsize=14, fontweight='bold', color=C['text'])
    box(ax, 0.34, 0.42, 0.26, 0.26, C['gray'], C['border'], lw=1.2)
    party_info = "Treecko Lv12\n██████░░ 85%\n\nWingull Lv10\n████░░░░ 50%\n\nRalts   Lv8\n████████ 100%"
    ax.text(0.36, 0.66, party_info, ha='left', va='top', fontsize=11,
            fontfamily='monospace', color='#333', linespacing=0.9)

    # Parsed state summary
    ax.text(0.32, 0.32, "Position: Route 106 (12, 8)", ha='center', va='center',
            fontsize=13, color='#555')
    ax.text(0.32, 0.22, "Badges: 0 | Money: $1,200", ha='center', va='center',
            fontsize=13, color='#555')
    ax.text(0.32, 0.10, "→ Multimodal input", ha='center', va='center',
            fontsize=12, style='italic', color='#666')

    # ========== COLUMN 2: ORCHESTRATOR ==========
    header(ax, 0.66, 1.26, 0.50, "Orchestrator", C['orch'])
    box(ax, 0.66, 0.05, 0.50, 1.19, C['orch'][0], C['orch'][1], lw=2)

    # Observe-Reason-Dispatch flow
    steps = [
        ("Observe", "Game state → VLM", 1.02),
        ("Reason", "Plan next action", 0.74),
        ("Dispatch", "→ Agent OR Tool", 0.46),
    ]

    for title, desc, y in steps:
        box(ax, 0.71, y, 0.40, 0.22, C['white'], C['border'], lw=1.5)
        ax.text(0.91, y + 0.15, title, ha='center', va='center',
                fontsize=16, fontweight='bold', color=C['text'])
        ax.text(0.91, y + 0.06, desc, ha='center', va='center',
                fontsize=13, color='#555')

    # Flow arrows between steps
    arr(ax, (0.91, 1.02), (0.91, 0.96), lw=2.5)
    arr(ax, (0.91, 0.74), (0.91, 0.68), lw=2.5)

    # Context note
    ax.text(0.91, 0.30, "100K token context", ha='center', va='center',
            fontsize=12, style='italic', color='#666')
    ax.text(0.91, 0.18, "with history compaction", ha='center', va='center',
            fontsize=12, style='italic', color='#666')

    # ========== COLUMN 3: SUB-AGENTS (text left, image right) ==========
    header(ax, 1.20, 1.26, 0.62, "Sub-Agents", C['agent'])

    agents = [
        ("Main", "General game\ncontrol", "Pokemon_Emerald_(Eng)_012.png", 1.03),
        ("Battle", "Move selection\n& switching", "Pokemon_Emerald_(Eng)_003.png", 0.84),
        ("Objective", "Plan game\npath/route", "Pokemon_Emerald_(Eng)_025.png", 0.65),
        ("Reflection", "Diagnose\nstuck states", "Pokemon_Emerald_(Eng)_070.png", 0.46),
        ("Verify", "Confirm goal\ncomplete", "Pokemon_Emerald_(Eng)_090.png", 0.27),
        ("Gym Puzzle", "Spatial\nreasoning", "Pokemon_Emerald_(Eng)_006.png", 0.08),
    ]

    for name, desc, shot, y in agents:
        box(ax, 1.20, y, 0.62, 0.18, C['agent'][0], C['agent'][1])
        # Name on left
        ax.text(1.24, y + 0.14, name, ha='left', va='center',
                fontsize=13, fontweight='bold', color=C['text'])
        # Description below name
        ax.text(1.24, y + 0.05, desc, ha='left', va='center',
                fontsize=11, color='#333', linespacing=0.85)
        # Image on right side
        img(ax, shot, 1.70, y + 0.09, z=0.38)

    # ========== COLUMN 4: MCP TOOLS ==========
    header(ax, 1.86, 1.26, 0.74, "MCP Tools", C['tool'])

    # GAME CONTROL - GBA buttons
    box(ax, 1.86, 0.90, 0.74, 0.34, C['tool'][0], C['tool'][1])
    ax.text(1.91, 1.19, "Game Control", ha='left', va='center',
            fontsize=14, fontweight='bold', color=C['text'])
    draw_gba_buttons(ax, 1.89, 0.92)
    # Function names
    ax.text(2.32, 1.14, "press_buttons()", ha='left', va='center',
            fontsize=10, fontfamily='monospace', color='#555')
    ax.text(2.32, 1.06, "navigate_to()", ha='left', va='center',
            fontsize=10, fontfamily='monospace', color='#555')

    # KNOWLEDGE BASE - Logged events
    box(ax, 1.86, 0.52, 0.74, 0.34, C['tool'][0], C['tool'][1])
    ax.text(1.91, 0.81, "Knowledge Base", ha='left', va='center',
            fontsize=14, fontweight='bold', color=C['text'])
    box(ax, 1.91, 0.55, 0.64, 0.22, C['gray'], C['border'], lw=1.2)
    kb_log = "► Beat Trainer: Bug Catcher\n► Found: Potion at Route 102\n► Wiped out on Route 104\n► Learned: Rock weak to Water"
    ax.text(1.94, 0.75, kb_log, ha='left', va='top', fontsize=11,
            fontfamily='monospace', color='#333', linespacing=1.15)

    # OBJECTIVES - 3 types
    box(ax, 1.86, 0.05, 0.74, 0.43, C['tool'][0], C['tool'][1])
    ax.text(1.91, 0.43, "Objectives (3 Types)", ha='left', va='center',
            fontsize=14, fontweight='bold', color=C['text'])

    # Story objectives
    ax.text(1.93, 0.35, "Story", ha='left', va='center', fontsize=13, fontweight='bold', color='#1565C0')
    ax.text(1.93, 0.31, "✓ Get starter\n○ Beat Gym 1", ha='left', va='top',
            fontsize=12, color='#555', linespacing=1.1)

    # Battling objectives
    ax.text(2.20, 0.35, "Battling", ha='left', va='center', fontsize=13, fontweight='bold', color='#C62828')
    ax.text(2.20, 0.31, "○ Train to Lv15\n○ Catch Ralts", ha='left', va='top',
            fontsize=12, color='#555', linespacing=1.1)

    # Dynamic objectives
    ax.text(1.93, 0.17, "Dynamic", ha='left', va='center', fontsize=13, fontweight='bold', color='#2E7D32')
    ax.text(1.93, 0.13, "○ Heal at Pokémon Center", ha='left', va='top',
            fontsize=12, color='#555')

    # ========== ARROWS (header to header) ==========
    header_y = 1.295  # center of headers (y=1.26, height=0.07)

    # Game Environment → Orchestrator
    arr(ax, (0.62, header_y), (0.66, header_y), lw=5, scale=28, outline=True)

    # Orchestrator → Sub-Agents
    arr(ax, (1.16, header_y), (1.20, header_y), lw=5, scale=28, outline=True)

    # Sub-Agents → MCP Tools
    arr(ax, (1.82, header_y), (1.86, header_y), lw=5, scale=28, outline=True)

    plt.tight_layout()
    plt.savefig('figures/pokeagent_architecture.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('figures/pokeagent_architecture.png', format='png', bbox_inches='tight', dpi=300)
    print("Saved figures/pokeagent_architecture.pdf and .png")

if __name__ == "__main__":
    main()
