#!/usr/bin/env python3
"""Generate Track 2 Task Diversity figure with game screenshots and Phosphor icons."""

import subprocess
import base64
from pathlib import Path

# Phosphor icon SVG paths (from assets/phosphor/*.svg)
ICONS = {
    'navigation': 'M216.49,184.49l-32,32a12,12,0,0,1-17-17L179,188H48a12,12,0,0,1,0-24H179l-11.52-11.51a12,12,0,0,1,17-17l32,32A12,12,0,0,1,216.49,184.49Zm-145-64a12,12,0,0,0,17-17L77,92H208a12,12,0,0,0,0-24H77L88.49,56.49a12,12,0,0,0-17-17l-32,32a12,12,0,0,0,0,17Z',
    'wild_battle': 'M192,28H64A36,36,0,0,0,28,64V192a36,36,0,0,0,36,36H192a36,36,0,0,0,36-36V64A36,36,0,0,0,192,28Zm12,164a12,12,0,0,1-12,12H64a12,12,0,0,1-12-12V64A12,12,0,0,1,64,52H192a12,12,0,0,1,12,12ZM104,88A16,16,0,1,1,88,72,16,16,0,0,1,104,88Zm40,40a16,16,0,1,1-16-16A16,16,0,0,1,144,128Zm40-40a16,16,0,1,1-16-16A16,16,0,0,1,184,88Zm-80,80a16,16,0,1,1-16-16A16,16,0,0,1,104,168Zm80,0a16,16,0,1,1-16-16A16,16,0,0,1,184,168Z',
    'trainer_battle': 'M216,28H152a12,12,0,0,0-9.33,4.45L79.5,110.51l-4.66-4.65a20,20,0,0,0-28.29,0L29.86,122.55a20,20,0,0,0,0,28.29h0L45,166,23.86,187.17a20,20,0,0,0,0,28.28l16.69,16.69a20,20,0,0,0,28.28,0L90,211l15.17,15.16a20,20,0,0,0,28.29,0l16.69-16.69a20,20,0,0,0,0-28.3l-4.65-4.65,78.06-63.17A12,12,0,0,0,228,104V40A12,12,0,0,0,216,28ZM54.69,212.34l-11-11L62,183l11,11Zm64.61-6L49.65,136.7l11.05-11,69.65,69.65ZM204,98.27l-75.58,61.17L121,152l47.51-47.5a12,12,0,0,0-17-17L104,135l-7.45-7.44L157.73,52H204Z',
    'training': 'M92,136a36,36,0,1,0-36-36A36,36,0,0,0,92,136Zm0-48a12,12,0,1,1-12,12A12,12,0,0,1,92,88Zm72,48a36,36,0,1,0-36-36A36,36,0,0,0,164,136Zm0-48a12,12,0,1,1-12,12A12,12,0,0,1,164,88ZM237.67,177.33,214.08,196l23.59,18.67a12,12,0,0,1-14.84,18.86l-24-19a44.06,44.06,0,0,1-72.38-1.73,44.06,44.06,0,0,1-72.38,1.73l-24,19A12,12,0,1,1,15.26,214.67L38.85,196,15.26,177.33a12,12,0,1,1,14.84-18.86l24,19a44,44,0,0,1,72.44,1.67,44,44,0,0,1,72.44-1.67l24-19a12,12,0,0,1,14.84,18.86ZM92,212a20,20,0,1,0-20-20A20,20,0,0,0,92,212Zm72,0a20,20,0,1,0-20-20A20,20,0,0,0,164,212Z',
    'menu': 'M224,48H32A16,16,0,0,0,16,64V192a16,16,0,0,0,16,16H224a16,16,0,0,0,16-16V64A16,16,0,0,0,224,48Zm0,144H32V64H224V192ZM48,136a8,8,0,0,1,8-8H200a8,8,0,0,1,0,16H56A8,8,0,0,1,48,136Zm0-32a8,8,0,0,1,8-8H200a8,8,0,0,1,0,16H56A8,8,0,0,1,48,104Zm0,64a8,8,0,0,1,8-8H200a8,8,0,0,1,0,16H56A8,8,0,0,1,48,168Z',
    'dialogue': 'M216,48H40A16,16,0,0,0,24,64V224a15.85,15.85,0,0,0,9.24,14.5A16.13,16.13,0,0,0,40,240a15.94,15.94,0,0,0,10.25-3.78l.09-.07L83,208H216a16,16,0,0,0,16-16V64A16,16,0,0,0,216,48ZM40,224h0ZM216,192H80a16,16,0,0,0-10.25,3.79l-.09.07L40,220.85V64H216Z',
    'puzzle': 'M208,76H180V56A52,52,0,0,0,76,56V76H48A20,20,0,0,0,28,96V208a20,20,0,0,0,20,20H208a20,20,0,0,0,20-20V96A20,20,0,0,0,208,76ZM100,56a28,28,0,0,1,56,0V76H100ZM204,204H52V100H204Z',
}

# Colors for each task type
COLORS = {
    'navigation': {'bg': '#e8f4fc', 'border': '#2980b9', 'icon': '#2980b9'},
    'wild_battle': {'bg': '#fff0f0', 'border': '#c0392b', 'icon': '#c0392b'},
    'trainer_battle': {'bg': '#fff0f0', 'border': '#c0392b', 'icon': '#c0392b'},
    'training': {'bg': '#e8fcf0', 'border': '#16a085', 'icon': '#16a085'},
    'menu': {'bg': '#e8f8e8', 'border': '#27ae60', 'icon': '#27ae60'},
    'dialogue': {'bg': '#fef5e8', 'border': '#e67e22', 'icon': '#e67e22'},
    'puzzle': {'bg': '#f5e8fc', 'border': '#8e44ad', 'icon': '#8e44ad'},
}


def load_image_as_base64(image_path):
    """Load an image file and return as base64 data URI."""
    if not image_path.exists():
        return None

    suffix = image_path.suffix.lower()
    mime_type = 'image/png' if suffix == '.png' else 'image/jpeg'

    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:{mime_type};base64,{data}"


def create_track2_tasks_svg():
    """Create SVG showing RPG subtask graph with game screenshots and icons."""

    script_dir = Path(__file__).parent.parent
    screenshots_dir = script_dir / "track2_screenshots"

    # Load screenshots
    screenshots = {
        'navigation': load_image_as_base64(screenshots_dir / "route119.png"),  # Complex route with weather
        'wild_battle': load_image_as_base64(screenshots_dir / "wild_battle.png"),
        'trainer_battle': load_image_as_base64(screenshots_dir / "battle.png"),
        'training': load_image_as_base64(screenshots_dir / "evolve.png"),
        'menu': load_image_as_base64(screenshots_dir / "menu.png"),
        'dialogue': load_image_as_base64(screenshots_dir / "dialog.png"),
        'puzzle': load_image_as_base64(screenshots_dir / "fortree_gym.png"),  # Gym 6 with rotation puzzle
    }

    # Task card dimensions - sized for GBA aspect ratio (240x160 = 3:2)
    card_width = 170
    card_height = 160
    img_height = 108  # maintains 3:2 ratio with width-8 padding

    def create_task_card(task_id, label, desc1, desc2, x, y):
        """Create a task card with screenshot and icon."""
        colors = COLORS[task_id]
        icon_path = ICONS[task_id]
        screenshot = screenshots.get(task_id)

        # Build the card
        card = f'''
    <!-- {label} Card -->
    <g transform="translate({x}, {y})">
      <!-- Card background -->
      <rect class="task-card" x="0" y="0" width="{card_width}" height="{card_height}"
            fill="{colors['bg']}" stroke="{colors['border']}" stroke-width="2.5" rx="8"/>

      <!-- Screenshot area with clip -->
      <defs>
        <clipPath id="clip-{task_id}">
          <rect x="4" y="4" width="{card_width-8}" height="{img_height}" rx="5"/>
        </clipPath>
      </defs>'''

        if screenshot:
            card += f'''
      <image href="{screenshot}" x="4" y="4" width="{card_width-8}" height="{img_height}"
             preserveAspectRatio="xMidYMid slice" clip-path="url(#clip-{task_id})"/>'''
        else:
            # Fallback: colored rectangle
            card += f'''
      <rect x="4" y="4" width="{card_width-8}" height="{img_height}" rx="5" fill="{colors['bg']}" opacity="0.5"/>'''

        # Icon badge in corner
        card += f'''
      <!-- Icon badge -->
      <circle cx="{card_width - 18}" cy="18" r="14" fill="white" stroke="{colors['border']}" stroke-width="1.5"/>
      <svg x="{card_width - 28}" y="8" width="20" height="20" viewBox="0 0 256 256">
        <path d="{icon_path}" fill="{colors['icon']}"/>
      </svg>

      <!-- Label -->
      <text class="task-label" x="{card_width/2}" y="{img_height + 22}">{label}</text>
      <text class="task-desc" x="{card_width/2}" y="{img_height + 36}">{desc1}</text>
      <text class="task-desc" x="{card_width/2}" y="{img_height + 48}">{desc2}</text>
    </g>'''

        return card

    # Build the full SVG - Circular wheel layout with Navigation at center
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 820 700">
  <defs>
    <style>
      .task-label {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 14px;
        font-weight: 700;
        fill: #2c3e50;
        text-anchor: middle;
      }
      .task-desc {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 10px;
        fill: #5d6d7e;
        text-anchor: middle;
      }
      .arrow {
        fill: none;
        stroke: #7f8c8d;
        stroke-width: 2.5;
        stroke-dasharray: 6,3;
      }
      .arrow-nav {
        fill: none;
        stroke: #3498db;
        stroke-width: 2.5;
      }
      .flow-label {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 10px;
        fill: #2c3e50;
        text-anchor: middle;
        font-weight: 600;
      }
      .center-hub {
        filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.15));
      }
    </style>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#7f8c8d"/>
    </marker>
    <marker id="arrowhead-nav" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#3498db"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="820" height="700" fill="white"/>
'''

    # Circular wheel layout - 6 tasks around Navigation center
    # Positions calculated for circle: center (410, 350), radius 210
    # Card centers at 60° intervals, with Training at top
    # Order (clockwise from top): Training, Wild Battle, Menu, Gym Puzzle, Dialogue, Trainer Battle
    # This puts both battles adjacent to Training

    # Center hub - Navigation (with emphasis)
    svg_content += '''
    <g class="center-hub">'''
    svg_content += create_task_card('navigation', 'Navigation', 'Overworld movement', 'Route traversal', 325, 270)
    svg_content += '''
    </g>'''

    # Circular positions (card top-left corners)
    # 0° (top): Training
    svg_content += create_task_card('training', 'Training', 'Level up, evolve', 'Strengthen team', 325, 20)

    # 60° (top-right): Wild Battle - adjacent to Training
    svg_content += create_task_card('wild_battle', 'Wild Battle', 'Random encounters', 'Catch and grind', 570, 120)

    # 120° (bottom-right): Menu
    svg_content += create_task_card('menu', 'Menu', 'Items, team, save', 'Resource management', 570, 370)

    # 180° (bottom): Gym Puzzle
    svg_content += create_task_card('puzzle', 'Gym Puzzle', 'Spatial reasoning', 'Unlock gym leader', 325, 520)

    # 240° (bottom-left): Dialogue
    svg_content += create_task_card('dialogue', 'Dialogue', 'NPC interaction', 'Story, hints, items', 80, 370)

    # 300° (top-left): Trainer Battle - adjacent to Training
    svg_content += create_task_card('trainer_battle', 'Trainer Battle', 'Fixed encounters', 'Story progression', 80, 120)

    # Add arrows - Circular wheel layout
    # Navigation center: (410, 350), edges: top=270, bottom=430, left=325, right=495
    # Card size: 170x160
    svg_content += '''
  <!-- Arrows from Navigation hub to each spoke -->

  <!-- Navigation -> Training (straight up) -->
  <path class="arrow-nav" d="M 410,270 L 410,180" marker-end="url(#arrowhead-nav)"/>

  <!-- Navigation -> Wild Battle (diagonal up-right) -->
  <path class="arrow-nav" d="M 495,290 L 570,260" marker-end="url(#arrowhead-nav)"/>

  <!-- Navigation -> Menu (diagonal down-right) -->
  <path class="arrow-nav" d="M 495,410 L 570,440" marker-end="url(#arrowhead-nav)"/>

  <!-- Navigation -> Gym Puzzle (straight down) -->
  <path class="arrow-nav" d="M 410,430 L 410,520" marker-end="url(#arrowhead-nav)"/>

  <!-- Navigation -> Dialogue (diagonal down-left) -->
  <path class="arrow-nav" d="M 325,410 L 250,440" marker-end="url(#arrowhead-nav)"/>

  <!-- Navigation -> Trainer Battle (diagonal up-left) -->
  <path class="arrow-nav" d="M 325,290 L 250,260" marker-end="url(#arrowhead-nav)"/>

  <!-- Outcome arrows (dashed): Battles -> Training -->

  <!-- Wild Battle -> Training (XP gain, from left side of Wild Battle into right side of Training) -->
  <path class="arrow" d="M 570,180 L 495,100" marker-end="url(#arrowhead)"/>
  <text class="flow-label" x="545" y="120">XP</text>

  <!-- Trainer Battle -> Training (XP gain, from right side of Trainer Battle into left side of Training) -->
  <path class="arrow" d="M 250,180 L 325,100" marker-end="url(#arrowhead)"/>
  <text class="flow-label" x="275" y="120">XP</text>

</svg>'''

    return svg_content


def main():
    svg_content = create_track2_tasks_svg()

    # Ensure we're in the figures directory
    script_dir = Path(__file__).parent.parent
    svg_path = script_dir / "track2_tasks.svg"
    png_path = script_dir / "track2_tasks.png"

    # Write SVG
    with open(svg_path, "w") as f:
        f.write(svg_content)
    print(f"Created {svg_path}")

    # Convert to PNG using cairosvg
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2.0)
        print(f"Created {png_path}")
    except ImportError:
        print("cairosvg not available, trying inkscape...")
        subprocess.run(["inkscape", str(svg_path), "--export-type=png",
                       f"--export-filename={png_path}", "--export-dpi=300"])


if __name__ == "__main__":
    main()
