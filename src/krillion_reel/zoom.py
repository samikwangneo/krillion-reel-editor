"""Step 4 - ZOOMS: per-frame zoom factor for each cam.

* Punch-in on that player's own reveal: ease-out-cubic up over `ease`, hold, ease back out.
  Strength depends on the tier (bigger for rarer answers).
* Any stretch longer than `idle_after` without a punch gets a slow, subtle push-in-and-out so the
  picture keeps moving.
`zoom.enabled: false` returns all ones; `zoom.strength` scales every zoom amount.
"""
from __future__ import annotations

import numpy as np

REACT_DELAY = 0.1  # start the punch just after the card lands


def _ease_out_cubic(u):
    return 1 - (1 - u) ** 3


def _ease_in_out_cubic(u):
    return np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2)


def curve(n_frames: int, fps: float, reveals: list[tuple[float, str | None]], cfg: dict) -> np.ndarray:
    z = np.ones(n_frames, np.float32)
    zc = cfg["zoom"]
    if not zc["enabled"] or n_frames == 0:
        return z
    k = float(zc["strength"])
    t = np.arange(n_frames) / fps
    busy = np.zeros(n_frames, bool)
    tiers = zc["tiers"]
    default_punch = tiers.get("rare", 1.15)
    for rt, tier in reveals:
        peak = 1 + (float(tiers.get(tier, default_punch)) - 1) * k
        t0 = rt + REACT_DELAY
        t1, t2, t3 = t0 + zc["ease"], t0 + zc["ease"] + zc["hold"], t0 + zc["ease"] + zc["hold"] + zc["ease_out"]
        up = (t >= t0) & (t < t1)
        z[up] = np.maximum(z[up], 1 + (peak - 1) * _ease_out_cubic((t[up] - t0) / zc["ease"]))
        hold = (t >= t1) & (t < t2)
        z[hold] = np.maximum(z[hold], peak)
        down = (t >= t2) & (t < t3)
        z[down] = np.maximum(z[down], peak - (peak - 1) * _ease_in_out_cubic((t[down] - t2) / zc["ease_out"]))
        busy |= (t >= t0) & (t < t3)
    # idle stretches
    idle_amp = (float(zc["idle_strength"]) - 1) * k
    if idle_amp > 0:
        i = 0
        while i < n_frames:
            if busy[i]:
                i += 1
                continue
            j = i
            while j < n_frames and not busy[j]:
                j += 1
            if (j - i) / fps > zc["idle_after"]:
                u = np.linspace(0, 1, j - i)
                z[i:j] = 1 + idle_amp * np.sin(np.pi * u) ** 2
            i = j
    return z
