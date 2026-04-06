#!/usr/bin/env python3
"""Generate the PokeAgent Architecture figure — Nature/Science journal style.

ViewBox 1000x430 optimized for NeurIPS linewidth (~5.5in).
At print: 1 SVG unit ~ 0.0055in. Font sizes designed for 6-7pt minimum.
"""

import os
import base64
from pathlib import Path

BLUE    = "#4C72B0"
BLUE_LT = "#EDF1F8"
ORANGE  = "#DD8452"
PURPLE  = "#8172B3"
PURPLE_LT = "#EFECF6"
GREEN   = "#55A868"
RED     = "#C44E52"
RED_LT  = "#FBEAEB"
DARK    = "#2D3748"
GRAY    = "#718096"
LGRAY   = "#E2E8F0"


def _b64(path):
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


def create_svg():
    d = Path(__file__).parent.parent
    ss = d / "track2_screenshots"
    battle_b64 = _b64(ss / "wild_battle.png")
    dialog_b64 = _b64(ss / "dialog.png")

    W, H = 1000, 430
    # Equalized 26px gutters: 160+26+274+26+148+26+274 = 934, margins: 38 left, 28 right
    cx = [38, 224, 524, 698]
    cw = [160, 274, 148, 274]
    top = 40
    # Per-column panel heights (hug content + 12px padding)
    ph1 = 270   # Col1: screenshots + parsed state + button input
    ph2 = 256   # Col2: prompt + VLM + ctx/stuck + loop
    ph3 = 362   # Col3: tallest (4 tool groups)
    ph4 = 248   # Col4: KB + obj mgr + self-reflect + verify agent

    # ── Styles & defs ──
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <defs>
    <style>
      .ct {{ font-family:'Helvetica Neue','Helvetica',Arial,sans-serif; font-weight:700; font-size:14px; letter-spacing:0.6px; text-transform:uppercase; }}
      .h  {{ font-family:'Helvetica Neue','Helvetica',Arial,sans-serif; font-weight:600; font-size:13px; }}
      .b  {{ font-family:'Helvetica Neue','Helvetica',Arial,sans-serif; font-weight:400; font-size:12px; }}
      .m  {{ font-family:'Monaco','Consolas',monospace; font-weight:500; font-size:12px; }}
      .p  {{ font-family:'Helvetica Neue','Helvetica',Arial,sans-serif; font-weight:600; font-size:11.5px; }}
      .al {{ font-family:'Helvetica Neue','Helvetica',Arial,sans-serif; font-weight:600; font-size:11.5px; }}
      .n  {{ font-family:'Helvetica Neue','Helvetica',Arial,sans-serif; font-weight:400; font-size:11.5px; font-style:italic; }}
      .lp {{ font-family:'Helvetica Neue','Helvetica',Arial,sans-serif; font-weight:600; font-size:12px; }}
    </style>
    <marker id="ah" markerWidth="8" markerHeight="7" refX="7" refY="3.5" orient="auto">
      <polygon points="0 0,8 3.5,0 7" fill="{GRAY}"/>
    </marker>
    <clipPath id="cb"><rect width="138" height="78" rx="4"/></clipPath>
    <clipPath id="cd"><rect width="138" height="52" rx="4"/></clipPath>
  </defs>

  <rect width="{W}" height="{H}" fill="white"/>

  <!-- Column panels (per-column heights) -->
  <rect x="{cx[0]}" y="{top}" width="{cw[0]}" height="{ph1}" rx="8" fill="{RED_LT}" opacity="0.45"/>
  <rect x="{cx[1]}" y="{top}" width="{cw[1]}" height="{ph2}" rx="8" fill="{BLUE_LT}" opacity="0.45"/>
  <rect x="{cx[2]}" y="{top}" width="{cw[2]}" height="{ph3}" rx="8" fill="{PURPLE_LT}" opacity="0.45"/>
  <rect x="{cx[3]}" y="{top}" width="{cw[3]}" height="{ph4}" rx="8" fill="#F2F3F6" opacity="0.6"/>

  <!-- Column titles -->
  <text x="{cx[0]+cw[0]//2}" y="28" class="ct" fill="{RED}" text-anchor="middle">Game Env.</text>
  <text x="{cx[1]+cw[1]//2}" y="28" class="ct" fill="{BLUE}" text-anchor="middle">Agent Core</text>
  <text x="{cx[2]+cw[2]//2}" y="28" class="ct" fill="{PURPLE}" text-anchor="middle">MCP Tools</text>
  <text x="{cx[3]+cw[3]//2}" y="28" class="ct" fill="{GRAY}" text-anchor="middle">Modules</text>


  <!-- ===== COLUMN 1: Game Environment ===== -->

  <!-- Battle screenshot -->
  <g transform="translate({cx[0]+11},{top+8})">
    <rect width="138" height="78" rx="4" fill="#111" stroke="{LGRAY}" stroke-width="0.8"/>
    <image href="{battle_b64}" width="138" height="78" clip-path="url(#cb)" preserveAspectRatio="xMidYMid slice"/>
    <rect x="3" y="3" width="78" height="16" rx="2.5" fill="{DARK}" opacity="0.75"/>
    <text x="8" y="15" class="p" fill="white">Game Frame</text>
  </g>

  <!-- Dialog screenshot -->
  <g transform="translate({cx[0]+11},{top+94})">
    <rect width="138" height="52" rx="4" fill="#111" stroke="{LGRAY}" stroke-width="0.8"/>
    <image href="{dialog_b64}" width="138" height="52" clip-path="url(#cd)" preserveAspectRatio="xMidYMid slice"/>
    <rect x="3" y="3" width="68" height="16" rx="2.5" fill="{DARK}" opacity="0.75"/>
    <text x="8" y="15" class="p" fill="white">Overworld</text>
  </g>

  <!-- Parsed State -->
  <g transform="translate({cx[0]+11},{top+154})">
    <rect width="138" height="50" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="42" rx="1.5" fill="{RED}"/>
    <text x="10" y="16" class="h" fill="{RED}">Parsed State</text>
    <text x="10" y="31" class="b" fill="{DARK}">Position, map, ASCII</text>
    <text x="10" y="44" class="b" fill="{DARK}">Party, HP, status</text>
  </g>

  <!-- Button Input -->
  <g transform="translate({cx[0]+11},{top+214})">
    <rect width="138" height="46" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="38" rx="1.5" fill="{RED}"/>
    <text x="10" y="16" class="h" fill="{RED}">Button Input</text>
    <text x="10" y="31" class="m" fill="{DARK}">A B Start Select</text>
    <text x="10" y="43" class="m" fill="{DARK}">Up Down Left Right</text>
  </g>


  <!-- ===== COLUMN 2: Agent Core ===== -->

  <!-- Prompt Builder -->
  <g transform="translate({cx[1]+10},{top+8})">
    <rect width="{cw[1]-20}" height="56" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="48" rx="1.5" fill="{BLUE}"/>
    <text x="12" y="16" class="h" fill="{BLUE}">Prompt Builder</text>
    <rect x="10" y="24" width="50" height="16" rx="3" fill="{BLUE}" opacity="0.08" stroke="{BLUE}" stroke-width="0.5"/>
    <text x="35" y="35" class="p" fill="{BLUE}" text-anchor="middle">System</text>
    <rect x="64" y="24" width="46" height="16" rx="3" fill="{RED}" opacity="0.08" stroke="{RED}" stroke-width="0.5"/>
    <text x="87" y="35" class="p" fill="{RED}" text-anchor="middle">State</text>
    <rect x="114" y="24" width="48" height="16" rx="3" fill="{ORANGE}" opacity="0.08" stroke="{ORANGE}" stroke-width="0.5"/>
    <text x="138" y="35" class="p" fill="{ORANGE}" text-anchor="middle">Goals</text>
    <rect x="166" y="24" width="52" height="16" rx="3" fill="{GRAY}" opacity="0.08" stroke="{GRAY}" stroke-width="0.5"/>
    <text x="192" y="35" class="p" fill="{GRAY}" text-anchor="middle">History</text>
    <rect x="222" y="24" width="32" height="16" rx="3" fill="{GREEN}" opacity="0.08" stroke="{GREEN}" stroke-width="0.5"/>
    <text x="238" y="35" class="p" fill="{GREEN}" text-anchor="middle">KB</text>
    <text x="12" y="50" class="b" fill="{GRAY}">Combines state, context, objectives into prompt</text>
  </g>

  <!-- Arrow down -->
  <line x1="{cx[1]+cw[1]//2}" y1="{top+66}" x2="{cx[1]+cw[1]//2}" y2="{top+74}" stroke="{GRAY}" stroke-width="1" marker-end="url(#ah)"/>

  <!-- VLM Backbone (clean inner diagram) -->
  <g transform="translate({cx[1]+10},{top+78})">
    <rect width="{cw[1]-20}" height="64" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="56" rx="1.5" fill="{BLUE}"/>
    <text x="12" y="16" class="h" fill="{BLUE}">Vision-Language Model</text>

    <!-- Input: Screen -->
    <rect x="10" y="26" width="44" height="28" rx="3" fill="{RED_LT}" stroke="{RED}" stroke-width="0.6"/>
    <rect x="15" y="30" width="34" height="14" rx="2" fill="{DARK}" opacity="0.1"/>
    <text x="32" y="50" class="p" fill="{RED}" text-anchor="middle">Screen</text>

    <!-- Input: Text -->
    <rect x="60" y="26" width="44" height="28" rx="3" fill="{BLUE_LT}" stroke="{BLUE}" stroke-width="0.6"/>
    <text x="82" y="44" class="p" fill="{BLUE}" text-anchor="middle">Text</text>

    <!-- Arrow -->
    <line x1="108" y1="40" x2="126" y2="40" stroke="{GRAY}" stroke-width="1" marker-end="url(#ah)"/>

    <!-- VLM box (prominent) -->
    <rect x="130" y="24" width="74" height="34" rx="6" fill="{BLUE}" opacity="0.12" stroke="{BLUE}" stroke-width="1.3"/>
    <text x="167" y="46" class="h" fill="{BLUE}" text-anchor="middle">Any VLM</text>

    <!-- Arrow -->
    <line x1="208" y1="40" x2="224" y2="40" stroke="{GRAY}" stroke-width="1" marker-end="url(#ah)"/>

    <!-- Output -->
    <rect x="228" y="24" width="26" height="34" rx="3" fill="{PURPLE_LT}" stroke="{PURPLE}" stroke-width="0.6"/>
    <text x="241" y="38" class="p" fill="{PURPLE}" text-anchor="middle">Tool</text>
    <text x="241" y="50" class="p" fill="{PURPLE}" text-anchor="middle">Call</text>
  </g>

  <!-- Arrow down (longer gap) -->
  <line x1="{cx[1]+cw[1]//2}" y1="{top+144}" x2="{cx[1]+cw[1]//2}" y2="{top+156}" stroke="{GRAY}" stroke-width="1" marker-end="url(#ah)"/>

  <!-- Context Manager + Stuck Detection -->
  <g transform="translate({cx[1]+10},{top+160})">
    <rect width="125" height="48" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="40" rx="1.5" fill="{BLUE}"/>
    <text x="10" y="15" class="h" fill="{BLUE}">Context Manager</text>
    <text x="10" y="29" class="b" fill="{DARK}">100K char compaction</text>
    <text x="10" y="42" class="b" fill="{GRAY}">Keep responses, trim state</text>
  </g>
  <g transform="translate({cx[1]+143},{top+160})">
    <rect width="125" height="48" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="40" rx="1.5" fill="{RED}"/>
    <text x="10" y="15" class="h" fill="{RED}">Stuck Detection</text>
    <text x="10" y="29" class="b" fill="{DARK}">Monitor recent coords</text>
    <text x="10" y="42" class="b" fill="{GRAY}">Auto-warn, try variance</text>
  </g>

  <!-- Agent Loop bar (prominent: higher opacity) -->
  <g transform="translate({cx[1]+10},{top+220})">
    <rect width="{cw[1]-20}" height="24" rx="12" fill="{BLUE}" opacity="0.15" stroke="{BLUE}" stroke-width="1" stroke-opacity="0.6"/>
    <text x="{(cw[1]-20)//2}" y="17" class="lp" fill="{BLUE}" text-anchor="middle">observe -- reason -- tool call -- execute -- repeat</text>
  </g>


  <!-- ===== COLUMN 3: MCP Tools ===== -->
'''

    tw = cw[2] - 18  # 130

    def tc(x, y, color, title, tools):
        h = 24 + len(tools) * 22 + 4
        s = f'''
  <g transform="translate({x},{y})">
    <rect width="{tw}" height="{h}" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="3" width="3" height="{h-6}" rx="1.5" fill="{color}"/>
    <text x="10" y="16" class="h" fill="{color}">{title}</text>'''
        for i, t in enumerate(tools):
            py = 22 + i * 22
            tl = min(len(t) * 7.5 + 14, tw - 14)
            s += f'''
    <rect x="8" y="{py}" width="{tl}" height="18" rx="3" fill="{color}" opacity="0.07" stroke="{color}" stroke-width="0.5" stroke-opacity="0.4"/>
    <text x="{8+tl//2}" y="{py+14}" class="m" fill="{color}" text-anchor="middle">{t}</text>'''
        s += '\n  </g>\n'
        return s

    tx = cx[2] + 9
    svg += tc(tx, top+8,   PURPLE, "Game Control", ["press_buttons", "navigate_to", "get_game_state"])
    svg += tc(tx, top+98,  GREEN,  "Knowledge",    ["add_knowledge", "search_knowledge", "get_summary"])
    svg += tc(tx, top+188, ORANGE, "Objectives",   ["complete_obj", "create_obj", "reflect", "gym_puzzle"])
    svg += tc(tx, top+298, PURPLE, "Verify",       ["verify_completion"])

    svg += f'''


  <!-- ===== COLUMN 4: Supporting Modules ===== -->

  <!-- Knowledge Base -->
  <g transform="translate({cx[3]+10},{top+8})">
    <rect width="{cw[3]-20}" height="62" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="54" rx="1.5" fill="{GREEN}"/>
    <text x="12" y="16" class="h" fill="{GREEN}">Knowledge Base</text>
    <text x="12" y="31" class="b" fill="{DARK}">Persistent JSON, importance-weighted retrieval</text>
    <text x="12" y="44" class="b" fill="{DARK}">Locations, NPCs, items, strategies</text>
    <text x="12" y="57" class="b" fill="{GRAY}">Survives across sessions</text>
  </g>

  <!-- Objective Manager -->
  <g transform="translate({cx[3]+10},{top+80})">
    <rect width="{cw[3]-20}" height="74" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="66" rx="1.5" fill="{ORANGE}"/>
    <text x="12" y="16" class="h" fill="{ORANGE}">Objective Manager</text>
    <rect x="10" y="24" width="72" height="16" rx="3" fill="{ORANGE}" opacity="0.08" stroke="{ORANGE}" stroke-width="0.5"/>
    <text x="46" y="35" class="p" fill="{ORANGE}" text-anchor="middle">Story</text>
    <rect x="86" y="24" width="80" height="16" rx="3" fill="{RED}" opacity="0.08" stroke="{RED}" stroke-width="0.5"/>
    <text x="126" y="35" class="p" fill="{RED}" text-anchor="middle">Battling</text>
    <rect x="170" y="24" width="86" height="16" rx="3" fill="{BLUE}" opacity="0.08" stroke="{BLUE}" stroke-width="0.5"/>
    <text x="213" y="35" class="p" fill="{BLUE}" text-anchor="middle">Dynamics</text>
    <text x="12" y="54" class="b" fill="{DARK}">Three parallel sequences for balanced progress</text>
    <text x="12" y="67" class="b" fill="{GRAY}">Milestones, team building, adaptive goals</text>
  </g>

  <!-- Self-Reflection -->
  <g transform="translate({cx[3]+10},{top+164})">
    <rect width="{cw[3]-20}" height="52" rx="4" fill="white" stroke="{LGRAY}" stroke-width="0.8"/>
    <rect x="0" y="4" width="3" height="44" rx="1.5" fill="{BLUE}"/>
    <text x="12" y="16" class="h" fill="{BLUE}">Self-Reflection</text>
    <text x="12" y="31" class="b" fill="{DARK}">Separate VLM analyzes stuck states</text>
    <text x="12" y="44" class="b" fill="{GRAY}">Compares against porymap ground truth</text>
  </g>

  <!-- Verification Agent (dashed border) -->
  <g transform="translate({cx[3]+10},{top+226})">
    <rect width="{cw[3]-20}" height="52" rx="4" fill="white" stroke="{BLUE}" stroke-width="0.8" stroke-dasharray="4,2"/>
    <rect x="0" y="4" width="3" height="44" rx="1.5" fill="{BLUE}"/>
    <text x="12" y="16" class="h" fill="{BLUE}">Verification Agent</text>
    <text x="12" y="31" class="b" fill="{DARK}">Independent VLM (no tools) validates completion</text>
    <text x="12" y="44" class="b" fill="{GRAY}">Prevents premature advancement</text>
  </g>


  <!-- ===== INTER-COLUMN ARROWS (all 1.5 stroke, rectilinear) ===== -->

  <!-- Col1 -> Col2: frame + state (elbow: horizontal then up) -->
  <path d="M {cx[0]+cw[0]},{top+60} L {cx[1]-6},{top+60} L {cx[1]-6},{top+36} L {cx[1]},{top+36}"
        stroke="{GRAY}" stroke-width="1.5" fill="none" marker-end="url(#ah)"/>
  <text x="{(cx[0]+cw[0]+cx[1])//2}" y="{top+54}" class="al" fill="{RED}" text-anchor="middle">frame + state</text>

  <!-- Col2 -> Col1: buttons (horizontal, left-pointing) -->
  <line x1="{cx[1]}" y1="{top+232}" x2="{cx[0]+cw[0]}" y2="{top+232}" stroke="{GRAY}" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="{(cx[0]+cw[0]+cx[1])//2}" y="{top+226}" class="al" fill="{BLUE}" text-anchor="middle">buttons</text>

  <!-- Col2 -> Col3: calls (horizontal at y=top+108) -->
  <line x1="{cx[1]+cw[1]}" y1="{top+108}" x2="{cx[2]}" y2="{top+108}" stroke="{GRAY}" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="{(cx[1]+cw[1]+cx[2])//2}" y="{top+102}" class="al" fill="{PURPLE}" text-anchor="middle">calls</text>

  <!-- Col3 -> Col2: results (horizontal at y=top+135, parallel) -->
  <line x1="{cx[2]}" y1="{top+135}" x2="{cx[1]+cw[1]}" y2="{top+135}" stroke="{GRAY}" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="{(cx[1]+cw[1]+cx[2])//2}" y="{top+148}" class="al" fill="{PURPLE}" text-anchor="middle">results</text>

  <!-- Col3 -> Col4: elbow connectors through vertical spine at gutter center -->
  <!-- Game Control -> Knowledge Base -->
  <path d="M {cx[2]+cw[2]},{top+44} L {cx[3]-4},{top+44} L {cx[3]-4},{top+38} L {cx[3]},{top+38}"
        stroke="{GRAY}" stroke-width="1.5" fill="none" marker-end="url(#ah)"/>
  <text x="{(cx[2]+cw[2]+cx[3])//2}" y="{top+38}" class="al" fill="{GRAY}" text-anchor="middle">record</text>

  <!-- Knowledge -> Objective Manager -->
  <path d="M {cx[2]+cw[2]},{top+130} L {cx[3]-4},{top+130} L {cx[3]-4},{top+114} L {cx[3]},{top+114}"
        stroke="{GRAY}" stroke-width="1.5" fill="none" marker-end="url(#ah)"/>
  <text x="{(cx[2]+cw[2]+cx[3])//2}" y="{top+124}" class="al" fill="{GRAY}" text-anchor="middle">update</text>

  <!-- Objectives -> Self-Reflection -->
  <path d="M {cx[2]+cw[2]},{top+230} L {cx[3]-4},{top+230} L {cx[3]-4},{top+190} L {cx[3]},{top+190}"
        stroke="{GRAY}" stroke-width="1.5" fill="none" marker-end="url(#ah)"/>
  <text x="{(cx[2]+cw[2]+cx[3])//2}" y="{top+224}" class="al" fill="{GRAY}" text-anchor="middle">check</text>

  <!-- Verify -> Verification Agent -->
  <path d="M {cx[2]+cw[2]},{top+320} L {cx[3]-4},{top+320} L {cx[3]-4},{top+252} L {cx[3]},{top+252}"
        stroke="{GRAY}" stroke-width="1.5" fill="none" marker-end="url(#ah)"/>
  <text x="{(cx[2]+cw[2]+cx[3])//2}" y="{top+290}" class="al" fill="{GRAY}" text-anchor="middle">confirm</text>

  <!-- Feedback loop (bottom) -->
  <path d="M {cx[3]+cw[3]//2},{top+ph4+2}
           L {cx[3]+cw[3]//2},{top+ph3+16}
           L {cx[0]+cw[0]//2},{top+ph3+16}
           L {cx[0]+cw[0]//2},{top+ph1+2}"
        stroke="{GRAY}" stroke-width="1.3" fill="none" stroke-dasharray="6,3" marker-end="url(#ah)"/>
  <rect x="{(cx[0]+cx[3]+cw[3])//2-46}" y="{top+ph3+9}" width="92" height="16" rx="3" fill="white"/>
  <text x="{(cx[0]+cx[3]+cw[3])//2}" y="{top+ph3+20}" class="al" fill="{GRAY}" text-anchor="middle">feedback loop</text>

</svg>'''

    return svg


def main():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(script_dir, "pokeagent_architecture.svg")

    print("Creating PokeAgent architecture figure (Nature style v6)...")
    svg_content = create_svg()

    with open(output_path, "w") as f:
        f.write(svg_content)
    print(f"Created: {output_path}")

    try:
        import cairosvg
        from PIL import Image

        pdf_path = os.path.join(script_dir, "pokeagent_architecture.pdf")
        png_path = os.path.join(script_dir, "pokeagent_architecture.png")

        print("Converting to PDF...")
        cairosvg.svg2pdf(url=output_path, write_to=pdf_path)
        print(f"Created: {pdf_path}")

        print("Converting to PNG (2x scale)...")
        cairosvg.svg2png(url=output_path, write_to=png_path, scale=2.0)
        print(f"Created: {png_path}")

        img = Image.open(png_path)
        print(f"PNG dimensions: {img.size[0]} x {img.size[1]} px")

    except ImportError:
        print("Note: Install cairosvg and pillow to auto-convert to PDF/PNG")


if __name__ == "__main__":
    main()
