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


def sfx_offset(cam: np.ndarray, screen: np.ndarray, rate: int = 16000, n_events: int = 25,
               search: tuple[float, float] = (-300.0, 300.0), tol: float = 0.03,
               min_votes: int = 4, peaks: int = 5) -> tuple[float | None, list[dict]]:
    """Find the game's own sound effects (from the screen recording) inside the cam audio.

    Each of the loudest `n_events` 0.6s chunks of screen audio is used as a template and matched
    (normalised waveform cross-correlation) against the cam audio. The cam mic usually hears the
    laptop speakers faintly; envelope xcorr misses that under room speech, but the exact waveform
    of a sound effect still stands out. Each match votes for an offset; returns (median offset of
    the biggest cluster within `tol`, per-event evidence), or (None, evidence) if fewer than
    `min_votes` independent events agree.
    """
    sos = signal.butter(4, [300, 6000], btype="band", fs=rate, output="sos")
    c = signal.sosfiltfilt(sos, cam.astype(np.float64))
    s = signal.sosfiltfilt(sos, screen.astype(np.float64))
    hop = rate // 100
    e = np.sqrt((s[: len(s) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1))
    picked: list[int] = []
    for i in np.argsort(e)[::-1]:
        if e[i] <= 0 or len(picked) >= n_events:
            break
        if all(abs(int(i) - j) > 150 for j in picked):  # events >= 1.5s apart
            picked.append(int(i))
    ev = []
    L = int(0.6 * rate)
    # one FFT of the whole cam track, reused for every template; window energy from a running sum
    nfft = 1 << int(np.ceil(np.log2(len(c) + L)))
    C = np.fft.rfft(c, nfft)
    csq = np.concatenate(([0.0], np.cumsum(c ** 2)))
    en = np.sqrt(np.maximum(csq[L:] - csq[:-L], 1e-9))  # energy of c[k:k+L]
    for i in sorted(picked):
        t0 = max(0.0, i / 100 - 0.15)
        tpl = s[int(t0 * rate):int(t0 * rate) + L]
        if len(tpl) < L or not np.any(tpl):
            continue
        cc = np.fft.irfft(C * np.conj(np.fft.rfft(tpl, nfft)), nfft)[: len(en)]  # cc[k] = sum c[k+j] tpl[j]
        ncc = np.abs(cc) / en / np.linalg.norm(tpl)
        lo = max(0, int((t0 + search[0]) * rate)); hi = min(len(ncc), int((t0 + search[1]) * rate))
        if hi <= lo:
            continue
        # each event votes for its top `peaks` matches: under loud room speech the true match is
        # not always the single best one, but it is almost always among the top few
        w = ncc[lo:hi].copy()
        for _ in range(peaks):
            k = int(np.argmax(w))
            if w[k] <= 0:
                break
            ev.append({"screen_t": round(t0, 2), "offset": round((lo + k) / rate - t0, 4), "ncc": round(float(w[k]), 3)})
            w[max(0, k - rate // 2):k + rate // 2] = 0
    best: list[dict] = []
    for a in ev:
        cl = [b for b in ev if abs(b["offset"] - a["offset"]) <= tol]
        cl = list({b["screen_t"]: b for b in sorted(cl, key=lambda b: b["ncc"])}.values())  # 1 vote per event
        if len(cl) > len(best) or (len(cl) == len(best) and sum(b["ncc"] for b in cl) > sum(b["ncc"] for b in best)):
            best = cl
    for b in ev:
        b["agrees"] = b in best
    ev = [b for b in ev if b["agrees"]] + [b for b in ev if not b["agrees"]][:25]
    if len(best) < min_votes:
        return None, ev
    return float(np.median([b["offset"] for b in best])), ev
