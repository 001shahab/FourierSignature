"""
Fourier Signature
=================

Turn a handwritten signature into a chain of rotating circles (epicycles)
that redraw it, and export the result as an animated GIF, MP4, animated SVG
or a high resolution poster.

Designed and developed by Prof. Shahab Anbarjafari (shb@3sholding.com).
"""

from .fourier import FourierSeries
from .path import image_to_path, load_points, resample, close_path
from .render import Theme, THEMES, render_animation, render_poster, render_svg

__all__ = [
    "FourierSeries",
    "image_to_path",
    "load_points",
    "resample",
    "close_path",
    "Theme",
    "THEMES",
    "render_animation",
    "render_poster",
    "render_svg",
]

__version__ = "1.0.0"
__author__ = "Prof. Shahab Anbarjafari"
__email__ = "shb@3sholding.com"
