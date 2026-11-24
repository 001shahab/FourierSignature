"""Command line interface: ``fourier-signature INPUT [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .fourier import FourierSeries
from .path import image_to_path, load_points, points_to_path
from .render import THEMES, render_animation, render_poster, render_svg

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fourier-signature",
        description=(
            "Redraw a signature with rotating circles (epicycles) and export it "
            "as an animated GIF, MP4, animated SVG or poster."
        ),
        epilog="Designed and developed by Prof. Shahab Anbarjafari (shb@3sholding.com).",
    )
    p.add_argument("input", type=Path, help="signature image (PNG, JPG, ...) or points file (JSON, CSV)")
    p.add_argument("-o", "--output", type=Path, help="animation file (.gif or .mp4)")
    p.add_argument("--svg", type=Path, help="also write an animated SVG")
    p.add_argument("--poster", type=Path, help="also write a high resolution PNG poster")
    p.add_argument("-n", "--circles", type=int, default=150, help="number of circles (default 150)")
    p.add_argument("--svg-circles", type=int, default=80, help="circles in the SVG (default 80)")
    p.add_argument("--frames", type=int, default=240, help="animation frames (default 240)")
    p.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    p.add_argument("--width", type=int, default=900, help="animation width in pixels (default 900)")
    p.add_argument("--duration", type=float, default=10.0, help="SVG loop length in seconds (default 10)")
    p.add_argument("--theme", choices=sorted(THEMES), default="midnight", help="colour theme")
    p.add_argument(
        "--closing", choices=["auto", "retrace", "jump"], default="auto",
        help="how the path returns to its start (default auto)",
    )
    p.add_argument("--points", type=int, default=3000, help="samples along the path (default 3000)")
    p.add_argument("--smooth", type=float, default=3.0, help="smoothing strength, 0 disables (default 3)")
    p.add_argument("--caption", help="small caption at the bottom of the animation and poster")
    p.add_argument("--no-circles", action="store_true", help="hide the circles, show only the ink")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src: Path = args.input
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 2
    if not (args.output or args.svg or args.poster):
        args.output = src.with_name(src.stem + "_fourier.gif")

    if src.suffix.lower() in IMAGE_EXT:
        z = image_to_path(src, n_points=args.points, closing=args.closing, smooth=args.smooth)
    else:
        z = points_to_path(load_points(src), n_points=args.points, closing=args.closing,
                           smooth=args.smooth)

    full = FourierSeries.from_points(z)
    series = full.truncate(min(args.circles + 1, len(full)))
    kept = series.energy_captured(z)
    print(f"{len(series) - 1} circles capture {kept:.2%} of the signature's energy")

    if args.output:
        render_animation(series, args.output, frames=args.frames, fps=args.fps, width=args.width,
                         theme=args.theme, caption=args.caption, show_circles=not args.no_circles)
        print(f"animation  -> {args.output}")
    if args.svg:
        render_svg(full, args.svg, n_circles=args.svg_circles, duration=args.duration,
                   width=args.width, theme=args.theme, show_circles=not args.no_circles)
        print(f"svg        -> {args.svg}")
    if args.poster:
        render_poster(series, args.poster, theme=args.theme, caption=args.caption)
        print(f"poster     -> {args.poster}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
