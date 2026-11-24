"""Discrete Fourier series of a closed planar curve.

A curve is treated as a sequence of complex numbers z_n = x_n + i*y_n,
sampled uniformly in time t in [0, 1). Its discrete Fourier transform

    c_k = (1/N) * sum_n z_n * exp(-2*pi*i*k*n/N)

gives one rotating vector (an epicycle) per frequency k. Adding the vectors
tip to tail reproduces the curve:

    z(t) = sum_k c_k * exp(2*pi*i*k*t)

Keeping only the largest |c_k| gives the best approximation (in the least
squares sense) for a given number of circles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourierSeries:
    """A truncated Fourier series, ordered from the largest circle down.

    Attributes
    ----------
    coefficients:
        Complex amplitudes c_k. ``abs(c)`` is the radius of each circle and
        ``angle(c)`` its starting phase.
    frequencies:
        Integer frequencies k (turns per full period). Negative values spin
        clockwise.
    """

    coefficients: np.ndarray
    frequencies: np.ndarray

    # ------------------------------------------------------------------ build
    @classmethod
    def from_points(cls, z: np.ndarray, n_terms: int | None = None) -> "FourierSeries":
        """Fit a series to complex samples ``z`` (uniform in time).

        The constant term (k = 0, the centre of mass) is always kept first so
        the chain of circles is anchored. The remaining terms are sorted by
        decreasing radius and truncated to ``n_terms`` circles in total.
        """
        z = np.asarray(z, dtype=complex).ravel()
        n = z.size
        if n < 2:
            raise ValueError("need at least two points to build a Fourier series")

        coeffs = np.fft.fft(z) / n
        freqs = np.fft.fftfreq(n, d=1.0 / n).round().astype(int)

        dc = np.flatnonzero(freqs == 0)
        rest = np.flatnonzero(freqs != 0)
        rest = rest[np.argsort(-np.abs(coeffs[rest]), kind="stable")]
        order = np.concatenate([dc, rest])

        if n_terms is not None:
            if n_terms < 1:
                raise ValueError("n_terms must be at least 1")
            order = order[:n_terms]

        return cls(coefficients=coeffs[order], frequencies=freqs[order])

    # --------------------------------------------------------------- evaluate
    def __len__(self) -> int:
        return int(self.coefficients.size)

    def truncate(self, n_terms: int) -> "FourierSeries":
        """Return a copy that keeps only the first ``n_terms`` circles."""
        return FourierSeries(self.coefficients[:n_terms], self.frequencies[:n_terms])

    def vectors(self, t: float | np.ndarray) -> np.ndarray:
        """Rotating vectors at time ``t``; shape ``(len(t), n_terms)``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        phase = np.exp(2j * np.pi * np.outer(t, self.frequencies))
        return phase * self.coefficients[None, :]

    def joints(self, t: float) -> np.ndarray:
        """Positions of every joint in the chain at time ``t``.

        Returns ``n_terms + 1`` complex points: the origin, then the tip of
        each successive circle. The last point lies on the curve.
        """
        v = self.vectors(t)[0]
        return np.concatenate([[0j], np.cumsum(v)])

    def evaluate(self, t: float | np.ndarray) -> np.ndarray:
        """Point(s) on the reconstructed curve at time(s) ``t``."""
        return self.vectors(t).sum(axis=1)

    def curve(self, samples: int = 2000) -> np.ndarray:
        """The full reconstructed closed curve, ``samples`` points long."""
        return self.evaluate(np.linspace(0.0, 1.0, samples, endpoint=False))

    @property
    def radii(self) -> np.ndarray:
        return np.abs(self.coefficients)

    def energy_captured(self, z: np.ndarray) -> float:
        """Fraction of the curve's (mean-removed) energy kept by this series.

        By Parseval's theorem the energy of a signal equals the sum of the
        squared magnitudes of its Fourier coefficients.
        """
        z = np.asarray(z, dtype=complex).ravel()
        total = np.mean(np.abs(z - z.mean()) ** 2)
        if total == 0:
            return 1.0
        kept = np.sum(np.abs(self.coefficients[self.frequencies != 0]) ** 2)
        return float(min(kept / total, 1.0))
