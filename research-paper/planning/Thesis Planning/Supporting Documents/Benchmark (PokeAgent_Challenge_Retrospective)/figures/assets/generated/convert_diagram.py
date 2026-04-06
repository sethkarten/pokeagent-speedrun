#!/usr/bin/env python3
"""Convert SVG diagrams to PDF and PNG formats for NeurIPS publication."""

import subprocess
import sys
import os

def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    try:
        import cairosvg
        return True
    except ImportError:
        print("Installing cairosvg...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg", "-q"])
        return True

def convert_svg(svg_name="battle_loop_diagram"):
    """Convert SVG to PDF and PNG."""
    import cairosvg

    # Paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    svg_path = os.path.join(script_dir, f"{svg_name}.svg")
    pdf_path = os.path.join(script_dir, f"{svg_name}.pdf")
    png_path = os.path.join(script_dir, f"{svg_name}.png")

    if not os.path.exists(svg_path):
        print(f"Error: {svg_path} not found")
        return

    print(f"Converting {svg_path}...")

    # Convert to PDF (vector)
    print("Creating PDF (vector)...")
    cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
    print(f"  -> {pdf_path}")

    # Convert to PNG at 2x scale for high DPI
    print("Creating PNG (2x scale for 300 DPI)...")
    cairosvg.svg2png(url=svg_path, write_to=png_path, scale=2.0)
    print(f"  -> {png_path}")

    print("\nDone! Files created:")
    print(f"  - {pdf_path} (vector, for LaTeX)")
    print(f"  - {png_path} (raster)")

    # Verify dimensions
    try:
        from PIL import Image
        img = Image.open(png_path)
        print(f"\nPNG dimensions: {img.size[0]} x {img.size[1]} px")
    except ImportError:
        pass

if __name__ == "__main__":
    check_and_install_dependencies()

    # Convert specified diagram or default to battle_loop_diagram
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            convert_svg(name)
    else:
        # Default: convert all known SVG diagrams
        diagrams = ["battle_loop_diagram", "pokechamp_architecture", "pokeagent_architecture"]
        for name in diagrams:
            convert_svg(name)
            print("\n" + "="*50 + "\n")
