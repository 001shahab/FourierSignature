"""Turn signatures (images or point lists) into one ordered, closed path.

A Fourier series needs a single continuous curve, but a signature is made of
pixels or of several pen strokes. The pipeline here is:

1. Ink extraction: grayscale, automatic polarity, Otsu threshold, speck removal.
2. Skeletonisation: thin every stroke down to a one pixel wide centre line.
3. Ordering: walk the skeleton graph stroke by stroke with an Euler tour,
   so the pen follows the ink and only lifts between separate strokes.
4. Resampling: redistribute the points uniformly by arc length, so the
   epicycles move at a steady speed along the signature.
5. Closing: a Fourier series is periodic, so the path must return to its
   start. ``retrace`` walks back along the signature (invisible), ``jump``
   draws a straight line home, ``auto`` picks ``jump`` when home is close.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter1d, label
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

PathLike = str | Path


# ---------------------------------------------------------------- utilities
def resample(z: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline uniformly by arc length to ``n`` points."""
    z = np.asarray(z, dtype=complex).ravel()
    if z.size < 2:
        raise ValueError("a path needs at least two points")
    seg = np.abs(np.diff(z))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.full(n, z[0])
    target = np.linspace(0.0, s[-1], n)
    return np.interp(target, s, z.real) + 1j * np.interp(target, s, z.imag)


def close_path(z: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Make an open path periodic.

    ``auto`` uses ``jump`` when the path already ends near its start (within
    a quarter of its size) and ``retrace`` otherwise. ``retrace`` appends the path in reverse so the pen returns over its own
    ink (no visible closing line). ``jump`` leaves the path as is, which the
    Fourier series closes with a straight segment.
    """
    z = np.asarray(z, dtype=complex).ravel()
    if mode == "auto":
        mode = "jump" if _end_gap(z) <= 0.25 else "retrace"
    if mode == "retrace":
        return np.concatenate([z, z[-2:0:-1]])
    if mode == "jump":
        return z
    raise ValueError(f"unknown closing mode {mode!r}; use 'auto', 'retrace' or 'jump'")


def _end_gap(z: np.ndarray) -> float:
    """Distance from the last point to the first, relative to the path size."""
    span = max(np.ptp(z.real), np.ptp(z.imag)) or 1.0
    return float(abs(z[-1] - z[0]) / span)


def normalise(z: np.ndarray) -> np.ndarray:
    """Centre the path on the origin and scale its larger side to 2."""
    z = np.asarray(z, dtype=complex)
    lo = complex(z.real.min(), z.imag.min())
    hi = complex(z.real.max(), z.imag.max())
    centre = (lo + hi) / 2
    span = max(hi.real - lo.real, hi.imag - lo.imag) or 1.0
    return (z - centre) * (2.0 / span)


def order_points(points: np.ndarray, start: int | None = None) -> np.ndarray:
    """Order 2D points into a nearest neighbour tour (returns complex)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        raise ValueError("no points to order")
    tree = cKDTree(pts)
    visited = np.zeros(n, dtype=bool)
    # Start from the leftmost point: signatures are written left to right.
    current = int(np.argmin(pts[:, 0])) if start is None else start
    order = np.empty(n, dtype=int)
    for i in range(n):
        order[i] = current
        visited[current] = True
        if i == n - 1:
            break
        nxt = -1
        k = 8
        while k <= n:
            _, idx = tree.query(pts[current], k=min(k, n))
            idx = np.atleast_1d(idx)
            free = idx[~visited[idx]]
            if free.size:
                nxt = int(free[0])
                break
            if k == n:
                break
            k = min(k * 4, n)
        if nxt < 0:  # fallback: brute force over the remaining points
            remaining = np.flatnonzero(~visited)
            d = np.sum((pts[remaining] - pts[current]) ** 2, axis=1)
            nxt = int(remaining[np.argmin(d)])
        current = nxt
    ordered = pts[order]
    return ordered[:, 0] + 1j * ordered[:, 1]


# ------------------------------------------------------------------- images
def extract_ink(image: Image.Image, max_side: int = 900) -> np.ndarray:
    """Return a boolean mask of the ink in a signature image."""
    img = image
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        # Transparent PNGs: composite onto white so ink stays dark.
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    gray = ImageOps.exif_transpose(img).convert("L")
    if max(gray.size) > max_side:
        gray.thumbnail((max_side, max_side), Image.LANCZOS)
    a = np.asarray(gray, dtype=float) / 255.0
    # Ink is the minority colour: flip if the background is dark.
    if np.median(a) < 0.5:
        a = 1.0 - a
    try:
        t = threshold_otsu(a)
    except ValueError:  # flat image
        t = 0.5
    mask = a < t
    return _remove_specks(mask, min_size=max(8, int(mask.sum() * 0.002)))


def _remove_specks(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected blobs smaller than ``min_size`` pixels (dust, noise)."""
    labels, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def image_to_path(
    source: PathLike | Image.Image,
    n_points: int = 3000,
    closing: str = "auto",
    smooth: float = 3.0,
) -> np.ndarray:
    """Convert a signature image into a closed, normalised complex path."""
    image = source if isinstance(source, Image.Image) else Image.open(source)
    mask = extract_ink(image)
    if not mask.any():
        raise ValueError("no ink found in the image")
    skel = skeletonize(mask)
    stroke_width = mask.sum() / max(skel.sum(), 1)
    skel = _prune_spurs(skel, max_len=max(4, int(1.2 * stroke_width)))
    z = trace_skeleton(skel)
    return points_to_path(
        np.column_stack([z.real, z.imag]), n_points=n_points, closing=closing, smooth=smooth
    )


# ----------------------------------------------------------- skeleton graph
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _adjacency(skel: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    rows, cols = np.nonzero(skel)
    index = -np.ones(skel.shape, dtype=int)
    index[rows, cols] = np.arange(rows.size)
    h, w = skel.shape
    adj: list[list[int]] = [[] for _ in range(rows.size)]
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in _NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and index[rr, cc] >= 0:
                adj[i].append(int(index[rr, cc]))
    return np.column_stack([rows, cols]), adj


def _prune_spurs(skel: np.ndarray, max_len: int) -> np.ndarray:
    """Remove short side branches that skeletonisation leaves on thick ink."""
    skel = skel.copy()
    pix, adj = _adjacency(skel)
    deg = np.array([len(a) for a in adj])
    for end in np.flatnonzero(deg == 1):
        branch = [end]
        prev, cur = -1, end
        while len(branch) <= max_len:
            nxt = [n for n in adj[cur] if n != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
            if deg[cur] >= 3:  # reached a junction: this is a spur
                rr, cc = pix[branch].T
                skel[rr, cc] = False
                break
            branch.append(cur)
    return skel


def trace_skeleton(skel: np.ndarray) -> np.ndarray:
    """Walk a skeleton like a pen, returning one ordered complex path.

    Every connected stroke is covered by a depth first Euler tour that starts
    where the pen is closest and ends at the far side of the stroke: side
    branches are drawn and retraced, so the pen never leaves the ink inside a
    stroke. Strokes are visited left to right; long horizontal strokes
    (underlines, flourishes) are drawn last, as people usually do.
    """
    pix, adj = _adjacency(skel)
    n = len(pix)
    if n == 0:
        raise ValueError("empty skeleton")
    xy = np.column_stack([pix[:, 1], -pix[:, 0]]).astype(float)  # y axis up

    # Connected components.
    comp = -np.ones(n, dtype=int)
    comps: list[np.ndarray] = []
    for seed in range(n):
        if comp[seed] >= 0:
            continue
        stack, members = [seed], []
        comp[seed] = len(comps)
        while stack:
            v = stack.pop()
            members.append(v)
            for u in adj[v]:
                if comp[u] < 0:
                    comp[u] = len(comps)
                    stack.append(u)
        comps.append(np.array(members))

    total_w = np.ptp(xy[:, 0]) or 1.0
    tiny = max(5, int(0.01 * n))

    def sort_key(members: np.ndarray):
        x = xy[members, 0]
        is_flourish = np.ptp(x) > 0.5 * total_w
        return (is_flourish, x.min())

    comps = [c for c in comps if c.size >= tiny] or comps
    comps.sort(key=sort_key)

    walk: list[int] = []
    pen = None
    for members in comps:
        if pen is None:
            root = int(members[np.argmin(xy[members, 0])])
        else:  # start where the pen is closest, like lifting it to the next stroke
            root = int(members[np.argmin(np.sum((xy[members] - pen) ** 2, axis=1))])
        target = int(members[np.argmax(np.abs(xy[members, 0] - xy[root, 0]))])
        tour = _euler_tour(root, target, adj, xy)
        walk.extend(tour)
        pen = xy[tour[-1]]
    pts = xy[walk]
    return pts[:, 0] + 1j * pts[:, 1]


def _euler_tour(root: int, target: int, adj: list[list[int]], xy: np.ndarray) -> list[int]:
    # Breadth first spanning tree from the root.
    parent = {root: -1}
    queue = [root]
    for v in queue:
        for u in adj[v]:
            if u not in parent:
                parent[u] = v
                queue.append(u)
    children: dict[int, list[int]] = {v: [] for v in parent}
    for v, p in parent.items():
        if p >= 0:
            children[p].append(v)

    # The root to target path is walked last, so the tour ends at the target.
    spine = set()
    v = target
    while v >= 0:
        spine.add(v)
        v = parent[v]
    for v, ch in children.items():
        ch.sort(key=lambda u: (u in spine, xy[u, 0]))

    out = [root]
    stack = [(root, iter(children[root]))]
    while stack:
        v, it = stack[-1]
        u = next(it, None)
        if u is None:
            stack.pop()
            if stack and v not in spine:
                out.append(stack[-1][0])  # step back to the parent
            continue
        out.append(u)
        stack.append((u, iter(children[u])))
    return out


def points_to_path(
    points: np.ndarray,
    n_points: int = 3000,
    closing: str = "auto",
    smooth: float = 3.0,
    ordered: bool = True,
) -> np.ndarray:
    """Turn 2D points into a closed, smooth, normalised complex path."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    z = pts[:, 0] + 1j * pts[:, 1] if ordered else order_points(pts)
    if closing == "auto":
        closing = "jump" if _end_gap(z) <= 0.25 else "retrace"
    half = n_points // 2 if closing == "retrace" else n_points
    z = resample(z, max(half, 4))
    z = close_path(z, closing)
    if smooth > 0:
        z = gaussian_filter1d(z.real, smooth, mode="wrap") + 1j * gaussian_filter1d(
            z.imag, smooth, mode="wrap"
        )
    return normalise(z)


# ------------------------------------------------------------------- points
def load_points(path: PathLike) -> np.ndarray:
    """Load an ordered point list from JSON or CSV.

    JSON may be ``[[x, y], ...]``, ``{"points": [[x, y], ...]}`` or the
    stroke format exported by the web pad: ``{"strokes": [[[x, y], ...], ...]}``.
    Screen coordinates (y down) are expected from the pad and are flipped
    when ``"y_down": true`` is present. CSV must have two numeric columns.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        y_down = False
        if isinstance(data, dict):
            y_down = bool(data.get("y_down", False))
            if "strokes" in data:
                pts = [p for stroke in data["strokes"] for p in stroke]
            else:
                pts = data["points"]
        else:
            pts = data
        arr = np.asarray(pts, dtype=float)
        if y_down:
            arr[:, 1] *= -1
        return arr
    rows = []
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            try:
                rows.append([float(row[0]), float(row[1])])
            except (ValueError, IndexError):
                continue  # header or blank line
    if not rows:
        raise ValueError(f"no numeric points found in {path}")
    return np.asarray(rows)
