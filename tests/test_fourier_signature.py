import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from fourier_signature import FourierSeries, image_to_path, load_points, render_svg
from fourier_signature.cli import main
from fourier_signature.path import close_path, points_to_path, resample


def circle(n=256, r=1.0, k=1):
    t = np.linspace(0, 1, n, endpoint=False)
    return r * np.exp(2j * np.pi * k * t)


def test_single_circle_is_one_epicycle():
    s = FourierSeries.from_points(circle(r=2.0))
    assert s.frequencies[0] == 0
    assert s.frequencies[1] == 1
    assert s.radii[1] == pytest.approx(2.0)
    assert np.all(s.radii[2:] < 1e-9)


def test_full_series_reconstructs_samples_exactly():
    rng = np.random.default_rng(0)
    z = rng.normal(size=64) + 1j * rng.normal(size=64)
    s = FourierSeries.from_points(z)
    t = np.arange(64) / 64
    np.testing.assert_allclose(s.evaluate(t), z, atol=1e-9)


def test_truncation_keeps_largest_and_energy_is_monotone():
    z = circle(k=1) + 0.3 * circle(k=-3) + 0.05 * circle(k=7)
    full = FourierSeries.from_points(z)
    two = full.truncate(3)
    assert set(two.frequencies.tolist()) == {0, 1, -3}
    assert full.truncate(2).energy_captured(z) < two.energy_captured(z) <= 1.0


def test_joints_end_on_curve():
    s = FourierSeries.from_points(circle() + 0.2 * circle(k=5))
    j = s.joints(0.3)
    assert j[0] == 0
    assert j[-1] == pytest.approx(s.evaluate(0.3)[0])


def test_resample_is_uniform_by_arc_length():
    z = np.array([0, 1, 1 + 3j])
    r = resample(z, 5)
    assert np.allclose(np.abs(np.diff(r)), 1.0)


def test_close_path_modes():
    z = np.array([0, 1, 2, 3], dtype=complex)
    assert np.allclose(close_path(z, "retrace"), [0, 1, 2, 3, 2, 1])
    assert np.allclose(close_path(z, "jump"), z)
    loop = circle(32)[:-1]
    assert np.allclose(close_path(loop, "auto"), loop)
    with pytest.raises(ValueError):
        close_path(z, "spiral")


@pytest.fixture
def signature_png(tmp_path):
    img = Image.new("L", (400, 160), 255)
    d = ImageDraw.Draw(img)
    d.line([(30, 120), (90, 30), (150, 120), (210, 30), (270, 120)], fill=0, width=9)
    d.line([(40, 145), (360, 135)], fill=0, width=6)
    p = tmp_path / "sig.png"
    img.save(p)
    return p


def test_image_to_path_is_normalised(signature_png):
    z = image_to_path(signature_png, n_points=800)
    assert z.size >= 790
    assert max(np.ptp(z.real), np.ptp(z.imag)) == pytest.approx(2.0, rel=0.05)
    assert abs((z.real.max() + z.real.min()) / 2) < 1e-6


def test_dark_background_images_work(tmp_path):
    img = Image.new("L", (200, 100), 0)
    ImageDraw.Draw(img).ellipse((40, 20, 160, 80), outline=255, width=6)
    p = tmp_path / "dark.png"
    img.save(p)
    z = image_to_path(p, n_points=400)
    assert np.isfinite(z).all()


def test_load_points_json_strokes(tmp_path):
    p = tmp_path / "pad.json"
    p.write_text(json.dumps({"y_down": True, "strokes": [[[0, 0], [10, 5]], [[20, 0]]]}))
    pts = load_points(p)
    assert pts.shape == (3, 2)
    assert pts[1, 1] == -5
    z = points_to_path(pts, n_points=100)
    assert np.isfinite(z).all()


def test_load_points_csv(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text("x,y\n0,0\n1,0\n1,1\n")
    assert load_points(p).shape == (3, 2)


def test_svg_is_well_formed(tmp_path):
    import xml.etree.ElementTree as ET

    s = FourierSeries.from_points(circle() + 0.3 * circle(k=-2))
    out = render_svg(s, tmp_path / "a.svg", n_circles=10)
    root = ET.parse(out).getroot()
    assert root.tag.endswith("svg")
    assert out.read_text().count("animateTransform") == len(s.truncate(11)) - 1


def test_cli_end_to_end(signature_png, tmp_path):
    gif, svg, png = tmp_path / "a.gif", tmp_path / "a.svg", tmp_path / "a.png"
    code = main([str(signature_png), "-o", str(gif), "--svg", str(svg), "--poster", str(png),
                 "--circles", "40", "--frames", "12", "--width", "320"])
    assert code == 0
    assert Image.open(gif).n_frames > 1
    assert svg.stat().st_size > 1000
    assert Image.open(png).size[0] > 1000


@pytest.mark.parametrize("closing", ["retrace", "jump"])
def test_cli_honours_circles_above_points(signature_png, tmp_path, capsys, closing):
    code = main([str(signature_png), "--poster", str(tmp_path / "a.png"), "--circles", "501",
                 "--points", "200", "--closing", closing])
    assert code == 0
    assert capsys.readouterr().out.startswith("501 circles")
