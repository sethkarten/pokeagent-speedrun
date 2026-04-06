#!/usr/bin/env python3
"""
Generate NeurIPS-quality speedrun route figure for Pokemon Emerald Track 2.
Flowchart with map screenshots in snake layout + geographic side minimap.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

# Config
FIGURES_DIR = Path(__file__).parent.parent
SCREENSHOTS_DIR = FIGURES_DIR / "track2_screenshots"
SPRITES_DIR = SCREENSHOTS_DIR / "sprites"
OUTPUT_FILE = str(FIGURES_DIR / "speedrun_route.png")
DPI = 300

# Colors
BG_COLOR = "#ffffff"
CARD_BG = "#ffffff"
CARD_BORDER = "#aaaaaa"
START_COLOR = "#27ae60"
END_COLOR = "#e74c3c"
WP_COLOR = "#2980b9"
GREY_COLOR = "#888888"

WP_COLORS = {
    1:  START_COLOR,   # Littleroot — green
    4:  WP_COLOR,      # Oldale Town — blue
    8:  "#8e44ad",     # Petalburg — purple
    11: GREY_COLOR,    # Petalburg Woods — grey
    13: END_COLOR,     # Rustboro — red
    14: END_COLOR,     # Rustboro Gym — red
    15: END_COLOR,     # Roxanne — red
}
ARROW_COLOR = "#111111"
TEXT_COLOR = "#2c3e50"
LIGHT_TEXT = "#556270"

# Human speedrunner splits (cumulative, M:SS).
# Top speedrunner total: 17:27 (matches "~18 minutes" in paper text).
HUMAN_TIMES = {
    1:  "0:57",
    2:  "2:44",
    3:  "3:00",
    4:  "4:02",
    5:  "4:28",
    6:  "5:50",
    7:  "6:48",
    8:  "7:58",
    9:  "8:25",
    10: "10:34",
    11: "10:53",
    12: "12:25",
    13: "12:35",
    14: "12:42",
    15: "17:27",
}

THUMB_SIZE = 180


def load_map(filename, crop_box=None):
    """Load map image, optionally crop."""
    filepath = SCREENSHOTS_DIR / filename
    if not filepath.exists():
        filepath = SCREENSHOTS_DIR / filename.replace('.png', '_hq.png')
        if not filepath.exists():
            return None
    img = Image.open(filepath)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    if crop_box:
        img = img.crop(crop_box)
    return img


def make_square_thumb(img, size=THUMB_SIZE, align='center'):
    """Crop to square then resize. align: 'center', 'left', 'right'."""
    w, h = img.size
    s = min(w, h)
    if align == 'left':
        left = 0
    elif align == 'right':
        left = w - s
    else:
        left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def add_border(img, color=(180, 180, 180, 255), width=2):
    bordered = img.copy()
    draw = ImageDraw.Draw(bordered)
    w, h = bordered.size
    for i in range(width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    return bordered


def create_figure():
    fig = plt.figure(figsize=(16, 9.5), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)

    # Single axes spanning the full figure
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.96])
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    # ── Waypoint data: (label, map_file, crop_box_px, thumb_align) ──
    wp_data = {
        1:  ("Littleroot",           "littleroot_town.png",    None,               'center'),
        2:  ("Route 101",           "route101.png",           None,               'center'),
        3:  ("Starter",              "starter_chosen.png",     None,               'center'),
        4:  ("Oldale Town",         "oldale_town.png",        None,               'center'),
        5:  ("Rival Battle",        "rival_battle.png",       None,               'center'),
        6:  ("Birch Lab",           "birch_lab.png",          None,               'center'),
        7:  ("Route 102",           "route102.png",           (100, 0, 520, 320), 'center'),
        8:  ("Petalburg",            "petalburg_city.png",     (0, 50, 480, 380),  'center'),
        9:  ("Dad's Gym",           "dad_gym_gameplay.png",   (9, 82, 2469, 1622),'center'),
        10: ("Route 104 S",         "route104.png",           (0, 780, 640, 1100),'center'),
        11: ("Petalburg Woods",     "petalburg_woods.png",    (60, 40, 500, 380), 'center'),
        12: ("Route 104 N",         "route104.png",           (0, 80, 640, 400),  'center'),
        13: ("Rustboro",             "rustboro_city.png",      (40, 60, 520, 440), 'center'),
        14: ("Rustboro Gym",        "rustboro_gym.png",       None,               'center'),
        15: ("Roxanne",              "roxanne_defeated.png",   (229, 0, 2709, 1651),'left'),
    }

    fallbacks = {}

    # ── Snake layout: 5 cols × 3 rows ──
    cols = 5
    card_w = 0.140
    card_h = 0.212
    h_gap = (card_w + (0.195 - 0.140) / 2) * 1.1   # ~0.185
    v_gap = card_h + (0.310 - 0.212) / 2   # 0.261
    x0, y0 = 0.09, 0.84
    thumb_zoom = 0.92

    pos = {}
    for wp in range(1, 16):
        r = (wp - 1) // cols
        c = (wp - 1) % cols
        if r == 1:
            c = cols - 1 - c
        pos[wp] = (x0 + c * h_gap, y0 - r * v_gap)

    # ── Draw continuous snake path with smooth bezier bends ──
    hw = card_w / 2 + 0.008   # clearance from card edge
    br = 0.03                  # bend radius for smooth corners

    row_y  = [pos[1][1],  pos[6][1],  pos[11][1]]
    right_x = pos[5][0]  + hw + br   # right-side bend column
    left_x  = pos[10][0] - hw - br   # left-side bend column

    k = 0.5523  # cubic bezier factor for quarter-circle approximation

    def qbez(p0, p1, p2, n=32):
        """Cubic bezier quarter-circle arc from p0→p2, rounding corner p1."""
        # Derive cubic control points from the quadratic corner geometry
        dx0, dy0 = p1[0]-p0[0], p1[1]-p0[1]
        dx2, dy2 = p2[0]-p1[0], p2[1]-p1[1]
        c1 = (p0[0] + k*dx0, p0[1] + k*dy0)
        c2 = (p2[0] - k*dx2, p2[1] - k*dy2)
        t = np.linspace(0, 1, n)
        x = ((1-t)**3*p0[0] + 3*(1-t)**2*t*c1[0]
             + 3*(1-t)*t**2*c2[0] + t**3*p2[0])
        y = ((1-t)**3*p0[1] + 3*(1-t)**2*t*c1[1]
             + 3*(1-t)*t**2*c2[1] + t**3*p2[1])
        return list(zip(x, y))

    segs = []  # list of (x, y) points for the full path

    # Row 1: left → right
    segs.append((pos[1][0] + hw,  row_y[0]))
    segs.append((right_x - br,    row_y[0]))

    # Right-side bend: right-turn-down (2 quarter-arcs with straight between)
    segs += qbez((right_x - br, row_y[0]),
                 (right_x,      row_y[0]),
                 (right_x,      row_y[0] - br))
    segs.append((right_x, row_y[1] + br))
    segs += qbez((right_x,      row_y[1] + br),
                 (right_x,      row_y[1]),
                 (right_x - br, row_y[1]))

    # Row 2: right → left
    segs.append((pos[10][0] - hw + br, row_y[1]))
    segs.append((left_x + br,          row_y[1]))

    # Left-side bend: left-turn-down
    segs += qbez((left_x + br, row_y[1]),
                 (left_x,      row_y[1]),
                 (left_x,      row_y[1] - br))
    segs.append((left_x, row_y[2] + br))
    segs += qbez((left_x,      row_y[2] + br),
                 (left_x,      row_y[2]),
                 (left_x + br, row_y[2]))

    # Row 3: left → right, continuing past box 15 to flag
    flag_x = pos[15][0] + hw + 0.04   # flag position to the right of box 15
    segs.append((pos[11][0] + hw - br, row_y[2]))
    segs.append((flag_x - 0.01,        row_y[2]))

    px, py = zip(*segs)
    ax.plot(px, py, color=ARROW_COLOR, linewidth=4.5,
            solid_capstyle='round', solid_joinstyle='round', zorder=0)

    # Arrowhead pointing into the flag
    ax.annotate('', xy=(flag_x, row_y[2]),
                xytext=(flag_x - 0.018, row_y[2]),
                arrowprops=dict(arrowstyle='-|>', color=ARROW_COLOR,
                                lw=0, mutation_scale=28, fc=ARROW_COLOR),
                zorder=0)

    # Flag icon
    flag_img = Image.open(FIGURES_DIR / "assets/custom/flag.png").convert("RGBA")
    flag_size = card_h * 1.1
    flag_ib = OffsetImage(np.array(flag_img), zoom=flag_size * 0.28)
    flag_ab = AnnotationBbox(flag_ib, (flag_x + 0.025, row_y[2]),
                             frameon=False, zorder=5)
    ax.add_artist(flag_ab)
    flag_ab.set_clip_on(False)

    # ── Draw waypoint cards ──
    for wp, (name, mapfile, crop, align) in wp_data.items():
        cx, cy = pos[wp]

        # Card background
        # Define image region first, then draw card to match
        label_h  = 0.030
        top_pad  = 0.003
        img_w    = card_w          # image fills full card width
        img_h    = card_h - label_h - top_pad
        ix0 = cx - img_w / 2
        ix1 = cx + img_w / 2
        iy0 = cy - card_h / 2 + label_h
        iy1 = iy0 + img_h          # = cy + card_h/2 - top_pad

        # Card spans image width and image+label height, hugging the content
        c_x0 = ix0
        c_y0 = cy - card_h / 2
        c_w  = img_w
        c_h  = img_h + label_h

        # Drop shadow
        shadow_off = 0.007
        shadow = FancyBboxPatch(
            (c_x0 + shadow_off, c_y0 - shadow_off), c_w, c_h,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            facecolor='#555555', edgecolor='none',
            alpha=0.28, zorder=1
        )
        ax.add_patch(shadow)

        wp_color = WP_COLORS.get(wp, GREY_COLOR)
        card_border = wp_color if wp in WP_COLORS else CARD_BORDER
        card_lw = 5.5 if wp in WP_COLORS else 5.0
        card = FancyBboxPatch(
            (c_x0, c_y0), c_w, c_h,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            facecolor=CARD_BG, edgecolor=card_border,
            linewidth=card_lw, zorder=2
        )
        ax.add_patch(card)

        # Human time label above card
        human_time = HUMAN_TIMES.get(wp)
        if human_time:
            label = f"Human Record Pace\n{human_time}" if wp == 1 else human_time
            ax.text(cx, c_y0 + c_h + 0.013, label,
                    ha='center', va='bottom', fontsize=13,
                    color='black', zorder=8,
                    fontfamily='monospace')

        # Load screenshot — use fallback if file missing
        img = load_map(mapfile, crop_box=crop)
        if img is None and mapfile in fallbacks:
            fb_file, fb_crop = fallbacks[mapfile]
            img = load_map(fb_file, crop_box=fb_crop)

        if img:
            thumb = make_square_thumb(img, align=align)
            im = ax.imshow(
                np.array(thumb),
                extent=(ix0, ix1, iy0, iy1),
                aspect='auto', zorder=4,
                interpolation='lanczos',
            )
            # Clip to the card rectangle so the image never bleeds outside
            clip_rect = mpatches.Rectangle(
                (ix0, iy0), ix1 - ix0, iy1 - iy0,
                transform=ax.transData)
            im.set_clip_path(clip_rect)

        # Waypoint number badge — scatter ensures circular in display coords
        badge_x = c_x0 + 0.018
        badge_y = iy1 - 0.022
        bc = WP_COLORS.get(wp, GREY_COLOR)
        ax.scatter([badge_x], [badge_y], s=1200, c=bc,
                   edgecolors='black', linewidths=1.5, zorder=6)
        ax.text(badge_x, badge_y, str(wp), ha='center', va='center',
                fontsize=14, fontweight='bold', color='white', zorder=7)

        # Label below thumbnail
        label_color = WP_COLORS.get(wp, TEXT_COLOR)
        label_fontsize = 11 if wp == 11 else 14
        ax.text(cx, cy - card_h / 2 + 0.012, name,
                ha='center', va='center', fontsize=label_fontsize,
                fontweight='bold' if wp in WP_COLORS else 'medium',
                color=label_color, zorder=8,
                bbox=dict(boxstyle='round,pad=0.15', facecolor=CARD_BG,
                         edgecolor='none', alpha=0.9))

    # Title
    ax.text(0.50, 1.02, 'Speedrunning Route (Early Game)',
            fontsize=23, fontweight='bold', color=TEXT_COLOR,
            ha='center', va='top', transform=ax.transAxes)

    ax.set_xlim(-0.02, 1.50)
    ax.set_ylim(-0.05, 1.02)

    # ═══════════════════════════════════════════
    #  MINIMAP – drawn directly in ax so alignment is exact.
    #  Littleroot anchored to row-3 center (y0 - 2*v_gap).
    #  Rustboro anchored to row-1 center (y0).
    #  Game coords: x east+, y north+.
    # ═══════════════════════════════════════════

    MAP_X0      = 1.07                    # data-x where game_x=0 (western edge)
    MAP_Y0      = y0 - 2 * v_gap - 0.02  # data-y where game_y=0 (Littleroot)
    MAP_SCALE_X = 0.35 / 5.5             # data units per game-x unit
    MAP_SCALE_Y = (2 * v_gap + 0.10) / 5.5  # more vertical room

    def mgx(gx): return MAP_X0 + gx * MAP_SCALE_X
    def mgy(gy): return MAP_Y0 + gy * MAP_SCALE_Y

    geo = {
        "Littleroot":  (mgx(5.295), mgy(0)),
        "Route 101":   (mgx(5.295), mgy(0.634)),
        "Oldale":      (mgx(5.295), mgy(1.269)),
        "Route 103":   (mgx(5.295), mgy(2.086)),
        "Route 102":   (mgx(3.8),   mgy(1.269)),
        "Petalburg":   (mgx(2.0),   mgy(1.452)),
        "Route 104 S": (mgx(0),   mgy(1.8)),
        "P. Woods":    (mgx(0),   mgy(2.6)),
        "Route 104 N": (mgx(0),   mgy(3.4)),
        "Rustboro":    (mgx(0),   mgy(4.9)),
    }

    # ── Overlay the real emerald map image behind nodes ──
    # Hard-coded extent — DO NOT recompute from node positions.
    MAP_IMG_EXTENT = (1.03, 1.44, 0.26, 0.91)  # (left, right, bottom, top)
    emerald_map = Image.open(FIGURES_DIR / "assets/custom/emerald_map.png").convert("RGBA")
    map_im = ax.imshow(np.array(emerald_map),
                       extent=MAP_IMG_EXTENT,
                       aspect='auto', zorder=0, alpha=0.55, interpolation='lanczos')

    # Route path lines
    path = ["Littleroot", "Route 101", "Oldale", "Route 102", "Petalburg",
            "Route 104 S", "P. Woods", "Route 104 N", "Rustboro"]
    for i in range(len(path) - 1):
        x1, y1 = geo[path[i]]
        x2, y2 = geo[path[i + 1]]
        ax.plot([x1, x2], [y1, y2], color=ARROW_COLOR, lw=3, alpha=0.8,
                zorder=1, solid_capstyle='round')

    # Side branch to Route 103 (dashed)
    x1, y1 = geo["Oldale"]
    x2, y2 = geo["Route 103"]
    ax.plot([x1, x2], [y1, y2], color=ARROW_COLOR, lw=2, alpha=0.4,
            zorder=1, linestyle='--')

    # Route waypoints — scatter gives true circles in display space
    route_steps = {
        "Route 101":   "2",
        "Route 103":   "5",
        "Route 102":   "7",
        "Route 104 S": "10",
        "Route 104 N": "12",
    }
    for loc, step in route_steps.items():
        lx, ly = geo[loc]
        ax.scatter([lx], [ly], s=600, c='#999999',
                   edgecolors='white', linewidths=1.5, zorder=12)
        ax.text(lx, ly, step, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=13)

    # Town markers
    # (waypoint_num, label_dx, label_dy, label_ha)
    towns = {
        "Littleroot":  (1,  -0.038,  0.000, 'right'),
        "Oldale":      (4,   0.038,  0.000, 'left'),
        "Petalburg":   (8,   0.000, -0.028, 'center'),
        "P. Woods":    (11,  0.038,  0.000, 'left'),
        "Rustboro":    (13,  0.038,  0.000, 'left'),
    }
    for loc, (wn, ldx, ldy, lha) in towns.items():
        lx, ly = geo[loc]
        c = WP_COLORS.get(wn, GREY_COLOR)
        ax.scatter([lx], [ly], s=800, c=c,
                   edgecolors='white', linewidths=2.0, zorder=13)
        ax.text(lx, ly, str(wn), ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', zorder=14)



    map_cx = (MAP_X0 + mgx(5.5)) / 2
    ax.text(map_cx, y0 + card_h / 2 - 0.33, 'World Map',
            fontsize=18, fontweight='bold', color=TEXT_COLOR,
            ha='center', va='bottom', zorder=14)

    return fig, map_im


def main():
    print("Creating speedrun route figure...")
    fig, map_im = create_figure()

    # Version WITH map
    fig.savefig(OUTPUT_FILE, dpi=DPI, facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved to {OUTPUT_FILE}")
    fig.savefig(OUTPUT_FILE.replace('.png', '.pdf'), dpi=DPI, facecolor=BG_COLOR,
                bbox_inches='tight', pad_inches=0.05)
    print(f"Saved to {OUTPUT_FILE.replace('.png', '.pdf')}")

    # Version WITHOUT map (map opacity → 0)
    map_im.set_alpha(0)
    out_no_map = OUTPUT_FILE.replace('.png', '_without_map.png')
    fig.savefig(out_no_map, dpi=DPI, facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved to {out_no_map}")
    fig.savefig(out_no_map.replace('.png', '.pdf'), dpi=DPI, facecolor=BG_COLOR,
                bbox_inches='tight', pad_inches=0.05)
    print(f"Saved to {out_no_map.replace('.png', '.pdf')}")

    plt.close(fig)


if __name__ == "__main__":
    main()
