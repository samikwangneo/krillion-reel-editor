"""Cam <-> screen sync.

Convention everywhere: cam_offset = cam_time - screen_time for the same real instant,
i.e. cam_t = screen_t + cam_offset.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

ENV_RATE = 100  # Hz for onset envelopes


def onset_envelope(x: np.ndarray, rate: int = 16000) -> np.ndarray:
    """Band-passed (200-4000 Hz) log-energy onset envelope at ENV_RATE.

    Onsets (positive energy changes) match across very different mics/rooms far better than
    raw waveforms do.
    """
    sos = signal.butter(4, [200, 4000], btype="band", fs=rate, output="sos")
    y = signal.sosfiltfilt(sos, x)
    hop = rate // ENV_RATE
    n = len(y) // hop
    e = np.log1p(1e4 * (y[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    on = np.maximum(np.diff(e, prepend=e[0]), 0)
    on -= on.mean()
    sd = on.std()
    return on / sd if sd > 0 else on


def xcorr_offset(cam: np.ndarray, screen: np.ndarray, rate: int = 16000,
                 max_lag_s: float | None = None) -> tuple[float, float]:
    """Return (cam_offset_seconds, confidence). confidence = main peak / best peak >1s away."""
    a, b = onset_envelope(cam, rate), onset_envelope(screen, rate)
    c = signal.correlate(a, b, mode="full", method="fft")
    lags = np.arange(-len(b) + 1, len(a))  # lag k: a[n + k] ~ b[n]
    if max_lag_s is not None:
        keep = np.abs(lags) <= max_lag_s * ENV_RATE
        c, lags = c[keep], lags[keep]
    i = int(np.argmax(c))
    far = np.abs(lags - lags[i]) > ENV_RATE
    second = float(c[far].max()) if far.any() else 0.0
    conf = float(c[i] / second) if second > 0 else float("inf")
    # parabolic sub-sample refinement
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
    else:
        d = 0.0
    return float((lags[i] + d) / ENV_RATE), conf


def windowed_check(cam: np.ndarray, screen: np.ndarray, offset: float, rate: int = 16000,
                   win_s: float = 40.0) -> list[float]:
    """Re-estimate the offset on several windows of the screen audio to detect drift / bad matches."""
    out = []
    dur = len(screen) / rate
    for start in np.linspace(5, max(5, dur - win_s - 5), 3):
        s0, s1 = int(start * rate), int((start + win_s) * rate)
        c0 = int((start + offset - 8) * rate)
        c1 = int((start + win_s + offset + 8) * rate)
        if c0 < 0 or c1 > len(cam) or s1 > len(screen):
            continue
        off, _ = xcorr_offset(cam[c0:c1], screen[s0:s1], rate, max_lag_s=10)
        out.append(round(off + (c0 - s0) / rate, 3))
    return out


def reaction_offset(cam: np.ndarray, reveal_times: list[float], rate: int = 16000,
                    search: tuple[float, float] = (-120, 120), react_delay: float = 0.4) -> tuple[float, float]:
    """Fallback when the screen has no usable audio.

    Builds an impulse train at each on-screen reveal (+ a typical human reaction delay) and finds
    the cam offset where cam loudness onsets best line up with it. Returns (offset, confidence).
    """
    env = onset_envelope(cam, rate)
    env = np.maximum(env, 0)
    # smooth: reactions are ~0.5-1.5s bursts
    k = np.hanning(int(0.6 * ENV_RATE))
    env = np.convolve(env, k / k.sum(), mode="same")
    scores = []
    lags = np.arange(int(search[0] * ENV_RATE), int(search[1] * ENV_RATE))
    for lag in lags:
        idx = [int((t + react_delay) * ENV_RATE) + lag for t in reveal_times]
        if min(idx) < 0 or max(idx) >= len(env):
            scores.append(-np.inf)
            continue
        scores.append(float(np.mean(env[idx])))
    scores = np.asarray(scores)
    i = int(np.argmax(scores))
    best = lags[i] / ENV_RATE
    far = np.isfinite(scores) & (np.abs(lags - lags[i]) > 2 * ENV_RATE)
    second = float(scores[far].max()) if far.any() else 0.0
    conf = float(scores[i] / second) if second > 0 else float("inf")
    return float(best), conf
