"""Rendering: animated GIF / MP4, animated SVG and still posters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fourier import FourierSeries

PathLike = str | Path


# -------------------------------------------------------------------- themes
@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    circle: str
    arm: str
    trace: str
    glow: str
    tip: str
    text: str


THEMES: dict[str, Theme] = {
    "midnight": Theme(
        name="midnight",
        background="#0b1020",
        circle="#7aa2ff",
        arm="#c9d6ff",
        trace="#ffd27a",
        glow="#ff9f43",
        tip="#ffffff",
        text="#8b97b8",
    ),
    "paper": Theme(
        name="paper",
        background="#f7f3ea",
        circle="#6b7a99",
        arm="#3c4660",
        trace="#1d2a6b",
        glow="#5468ff",
        tip="#c0392b",
        text="#7a7466",
    ),
    "neon": Theme(
        name="neon",
        background="#07070c",
        circle="#00e5ff",
        arm="#9bf6ff",
        trace="#ff4fd8",
        glow="#ff4fd8",
        tip="#ffffff",
        text="#6c6c8a",
    ),
}


def get_theme(theme: str | Theme) -> Theme:
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r}; choose from {', '.join(THEMES)}") from None


# ------------------------------------------------------------------- helpers
_UNIT = np.exp(2j * np.pi * np.linspace(0, 1, 64))


def _bounds(series: FourierSeries, pad: float = 0.12) -> tuple[float, float, float, float]:
    c = series.curve(3000)
    x0, x1, y0, y1 = c.real.min(), c.real.max(), c.imag.min(), c.imag.max()
    w, h = x1 - x0, y1 - y0
    p = pad * max(w, h)
    return x0 - p, x1 + p, y0 - p, y1 + p


def _figure(size_px: tuple[int, int], theme: Theme, bounds, dpi: int = 100):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = size_px
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=theme.background)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(theme.background)
    ax.set_axis_off()
    x0, x1, y0, y1 = bounds
    # Fit the drawing in the canvas while keeping the aspect ratio.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = x1 - x0, y1 - y0
    if bw / bh > w / h:
        bh = bw * h / w
    else:
        bw = bh * w / h
    ax.set_xlim(cx - bw / 2, cx + bw / 2)
    ax.set_ylim(cy - bh / 2, cy + bh / 2)
    ax.set_aspect("equal")
    return fig, ax


def _canvas_size(bounds, width: int) -> tuple[int, int]:
    x0, x1, y0, y1 = bounds
    ratio = (y1 - y0) / (x1 - x0)
    height = int(round(width * min(max(ratio, 0.42), 1.0) / 2) * 2)
    return width, height


def _circle_segments(joints: np.ndarray, radii: np.ndarray) -> list[np.ndarray]:
    segs = []
    for c, r in zip(joints[:-1], radii):
        ring = c + r * _UNIT
        segs.append(np.column_stack([ring.real, ring.imag]))
    return segs


def _circle_colors(theme: Theme, n: int, strength: float = 0.32):
    from matplotlib.colors import to_rgba

    base = np.array(to_rgba(theme.circle))
    alphas = strength * np.linspace(1.0, 0.25, max(n, 1)) ** 1.5
    cols = np.tile(base, (n, 1))
    cols[:, 3] = alphas[:n]
    return cols


# ----------------------------------------------------------------- animation
def render_animation(
    series: FourierSeries,
    out: PathLike,
    frames: int = 240,
    fps: int = 30,
    width: int = 900,
    theme: str | Theme = "midnight",
    hold_seconds: float = 1.5,
    caption: str | None = None,
    show_circles: bool = True,
) -> Path:
    """Render the epicycles drawing the signature to a GIF or MP4 file."""
    from matplotlib.collections import LineCollection

    th = get_theme(theme)
    out = Path(out)
    bounds = _bounds(series)
    size = _canvas_size(bounds, width)
    fig, ax = _figure(size, th, bounds)

    ts = np.linspace(0.0, 1.0, frames, endpoint=False)
    # Densely sample the trace so fast strokes stay smooth between frames.
    sub = 8
    dense_t = np.linspace(0.0, 1.0, frames * sub + 1)
    dense = series.evaluate(dense_t)

    radii = series.radii[1:]  # skip the constant term: it has no circle
    lw_scale = width / 900

    circles = LineCollection([], linewidths=0.8 * lw_scale, colors=_circle_colors(th, len(radii)))
    ax.add_collection(circles)
    (arms,) = ax.plot([], [], color=th.arm, lw=0.9 * lw_scale, alpha=0.75, solid_capstyle="round")
    (glow,) = ax.plot([], [], color=th.glow, lw=7 * lw_scale, alpha=0.12, solid_capstyle="round")
    (trace,) = ax.plot([], [], color=th.trace, lw=2.2 * lw_scale, solid_capstyle="round")
    (tip,) = ax.plot([], [], "o", color=th.tip, ms=4.5 * lw_scale)
    (tip_glow,) = ax.plot([], [], "o", color=th.glow, ms=14 * lw_scale, alpha=0.25)
    if caption:
        ax.text(
            0.5, 0.04, caption, transform=ax.transAxes, ha="center", va="bottom",
            color=th.text, fontsize=10 * lw_scale, family="DejaVu Sans",
        )

    images = []
    for i, t in enumerate(ts):
        j = series.joints(t)
        if show_circles:
            circles.set_segments(_circle_segments(j[1:], radii))
            arms.set_data(j.real[1:], j.imag[1:])
        upto = dense[: i * sub + 1]
        trace.set_data(upto.real, upto.imag)
        glow.set_data(upto.real, upto.imag)
        tip.set_data([j[-1].real], [j[-1].imag])
        tip_glow.set_data([j[-1].real], [j[-1].imag])
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    # Final frame: the finished signature, circles faded out.
    circles.set_segments([])
    arms.set_data([], [])
    tip.set_data([], [])
    tip_glow.set_data([], [])
    trace.set_data(dense.real, dense.imag)
    glow.set_data(dense.real, dense.imag)
    fig.canvas.draw()
    final = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    images.extend([final] * max(1, int(hold_seconds * fps)))

    import matplotlib.pyplot as plt

    plt.close(fig)
    _write_frames(images, out, fps)
    return out


def _write_frames(images: list[np.ndarray], out: Path, fps: int) -> None:
    suffix = out.suffix.lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".gif":
        from PIL import Image

        # One shared palette keeps colours stable and the file small.
        pil = [Image.fromarray(a) for a in images]
        palette_src = pil[len(pil) // 2]
        pal = palette_src.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in pil]
        frames[0].save(
            out, save_all=True, append_images=frames[1:], duration=int(1000 / fps),
            loop=0, optimize=True, disposal=1,
        )
    elif suffix in (".mp4", ".mov", ".webm"):
        import imageio.v2 as imageio

        with imageio.get_writer(out, fps=fps, quality=8, macro_block_size=2) as w:
            for a in images:
                w.append_data(a)
    else:
        raise ValueError("animation output must be .gif, .mp4, .mov or .webm")


# -------------------------------------------------------------------- poster
def render_poster(
    series: FourierSeries,
    out: PathLike,
    width: int = 2400,
    theme: str | Theme = "midnight",
    at: float = 0.37,
    caption: str | None = None,
) -> Path:
    """A still image: the signature partly drawn with the circle chain."""
    from matplotlib.collections import LineCollection

    th = get_theme(theme)
    out = Path(out)
    bounds = _bounds(series)
    fig, ax = _figure(_canvas_size(bounds, width), th, bounds, dpi=200)
    s = width / 900 / 2

    full = series.curve(6000)
    ax.plot(full.real, full.imag, color=th.trace, lw=1.0 * s, alpha=0.18)
    t = np.linspace(0, at, 4000)
    drawn = series.evaluate(t)
    ax.plot(drawn.real, drawn.imag, color=th.glow, lw=8 * s, alpha=0.12, solid_capstyle="round")
    ax.plot(drawn.real, drawn.imag, color=th.trace, lw=2.4 * s, solid_capstyle="round")

    j = series.joints(at)
    radii = series.radii[1:]
    ax.add_collection(
        LineCollection(_circle_segments(j[1:], radii), linewidths=0.8 * s,
                       colors=_circle_colors(th, len(radii), 0.45))
    )
    ax.plot(j.real[1:], j.imag[1:], color=th.arm, lw=0.9 * s, alpha=0.8)
    ax.plot([j[-1].real], [j[-1].imag], "o", color=th.glow, ms=16 * s, alpha=0.3)
    ax.plot([j[-1].real], [j[-1].imag], "o", color=th.tip, ms=5 * s)
    if caption:
        ax.text(0.5, 0.04, caption, transform=ax.transAxes, ha="center", va="bottom",
                color=th.text, fontsize=9 * s * 2, family="DejaVu Sans")

    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=th.background)
    plt.close(fig)
    return out


# --------------------------------------------------------------- animated SVG
def render_svg(
    series: FourierSeries,
    out: PathLike,
    n_circles: int = 80,
    duration: float = 10.0,
    width: int = 900,
    theme: str | Theme = "midnight",
    show_circles: bool = True,
) -> Path:
    """Write a self contained animated SVG (SMIL, no JavaScript).

    Each circle is a nested ``<g>`` rotating relative to its parent, so the
    browser composes the epicycles itself: the file stays small and loops
    forever. It plays in browsers, in GitHub READMEs and in most email
    clients that render SVG.
    """
    th = get_theme(theme)
    out = Path(out)
    s = series.truncate(min(n_circles + 1, len(series)))
    x0, x1, y0, y1 = _bounds(s)
    w_units, h_units = x1 - x0, y1 - y0
    height = int(round(width * h_units / w_units))
    sw = w_units / width  # one pixel in drawing units

    # Trace path, revealed with a dash offset synchronised to the tip.
    m = 1600
    t = np.linspace(0, 1, m + 1)
    curve = s.evaluate(t)
    d = "M" + " L".join(f"{p.real:.4f},{p.imag:.4f}" for p in curve)
    seg = np.abs(np.diff(curve))
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    keys = np.linspace(0, m, 201).astype(int)
    key_times = ";".join(f"{t[k]:.4f}" for k in keys)
    offsets = ";".join(f"{total - cum[k]:.4f}" for k in keys)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
    )
    parts.append(
        "<title>Fourier Signature</title>"
        "<desc>Epicycles redrawing a signature. Made with Fourier Signature by "
        "Prof. Shahab Anbarjafari.</desc>"
    )
    parts.append(f'<rect width="100%" height="100%" fill="{th.background}"/>')
    # Map drawing units to pixels with y pointing up.
    parts.append(
        f'<g transform="translate(0 {height}) scale({1 / sw:.6f} {-1 / sw:.6f}) '
        f'translate({-x0:.6f} {-y0:.6f})">'
    )
    parts.append(
        f'<path d="{d}" fill="none" stroke="{th.glow}" stroke-opacity="0.14" '
        f'stroke-width="{7 * sw:.5f}" stroke-linecap="round" stroke-linejoin="round" '
        f'stroke-dasharray="{total:.4f}" stroke-dashoffset="{total:.4f}">'
        f'<animate attributeName="stroke-dashoffset" dur="{duration}s" repeatCount="indefinite" '
        f'calcMode="linear" keyTimes="{key_times}" values="{offsets}"/></path>'
    )
    parts.append(
        f'<path d="{d}" fill="none" stroke="{th.trace}" stroke-width="{2.2 * sw:.5f}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'stroke-dasharray="{total:.4f}" stroke-dashoffset="{total:.4f}">'
        f'<animate attributeName="stroke-dashoffset" dur="{duration}s" repeatCount="indefinite" '
        f'calcMode="linear" keyTimes="{key_times}" values="{offsets}"/></path>'
    )

    # Epicycle chain. The constant term just offsets the origin.
    c0 = s.coefficients[0] if s.frequencies[0] == 0 else 0j
    coeffs = s.coefficients[1:] if s.frequencies[0] == 0 else s.coefficients
    freqs = s.frequencies[1:] if s.frequencies[0] == 0 else s.frequencies
    alphas = 0.38 * np.linspace(1.0, 0.25, max(len(coeffs), 1)) ** 1.5

    parts.append(f'<g transform="translate({c0.real:.5f} {c0.imag:.5f})">')
    prev_angle, prev_f = 0.0, 0
    for k, (c, f) in enumerate(zip(coeffs, freqs)):
        r = abs(c)
        a0 = np.degrees(np.angle(c)) - prev_angle
        df = int(f) - prev_f
        parts.append("<g>")
        parts.append(
            f'<animateTransform attributeName="transform" type="rotate" '
            f'from="{a0:.3f}" to="{a0 + 360 * df:.3f}" dur="{duration}s" '
            f'repeatCount="indefinite"/>'
        )
        if show_circles:
            parts.append(
                f'<circle r="{r:.5f}" fill="none" stroke="{th.circle}" '
                f'stroke-opacity="{alphas[k]:.3f}" stroke-width="{0.8 * sw:.5f}"/>'
                f'<line x2="{r:.5f}" stroke="{th.arm}" stroke-opacity="0.7" '
                f'stroke-width="{0.9 * sw:.5f}"/>'
            )
        parts.append(f'<g transform="translate({r:.5f} 0)">')
        prev_angle, prev_f = np.degrees(np.angle(c)), int(f)
    parts.append(
        f'<circle r="{7 * sw:.5f}" fill="{th.glow}" fill-opacity="0.3"/>'
        f'<circle r="{2.5 * sw:.5f}" fill="{th.tip}"/>'
    )
    parts.append("</g></g>" * len(coeffs))
    parts.append("</g></g></svg>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts))
    return out
