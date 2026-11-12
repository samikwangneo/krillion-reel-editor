"""Synthetic checks: known offsets must be recovered to within one video frame (1/30 s)."""
import numpy as np

from krillion_reel import audiosync, timeline, zoom
from krillion_reel.config import DEFAULTS

RATE = 16000


def _bursts(dur, times, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.002, int(dur * RATE)).astype(np.float32)
    for t in times:
        i = int(t * RATE)
        x[i:i + 4000] += rng.normal(0, 0.3, 4000) * np.hanning(4000)
    return x


def test_xcorr_recovers_offset():
    events = [3.1, 7.4, 12.0, 15.2, 21.7, 26.3, 33.9, 40.5]
    screen = _bursts(45, events, 1)
    true_off = 12.345  # cam_t = screen_t + off
    cam = _bursts(70, [t + true_off for t in events], 2)
    off, conf = audiosync.xcorr_offset(cam, screen)
    assert abs(off - true_off) < 1 / 30, off
    assert conf > 2


def test_reaction_fallback():
    reveals = [10.0, 31.5, 55.2, 80.1, 99.9, 120.4, 140.0]
    true_off = -4.2
    cam = _bursts(160, [r + true_off + 0.5 for r in reveals], 3)
    off, _ = audiosync.reaction_offset(cam, reveals, search=(-30, 30))
    assert abs(off - true_off) < 0.25, off


def test_timeline_aligns_prompts_and_holds_faster_player():
    q = lambda c, s, r, e: {"clock_start": c, "submit": s, "reveal": r, "tier_end": e, "tier": "rare"}  # noqa
    sync = {"cam_offset": {"p1": {"seconds": 1.0}, "p2": {"seconds": -2.0}},
            "questions": [{"n": 1, "p1": q(10, 20, 24, 30), "p2": q(50, 55, 58, 62)}],
            "results": {}}
    segs = timeline.build(sync, DEFAULTS)
    s = segs[0]
    assert abs(s.length - max(30 - 0.25 - 6.5, 62 - 0.25 - 46.5)) < 1e-9
    # both prompts line up at r = lead
    assert s.tracks["p1"].screen_t(3.5, 30) == 10 and s.tracks["p2"].screen_t(3.5, 30) == 50
    # p2 finishes first -> its screen holds, its cam keeps rolling
    assert s.tracks["p2"].screen_t(s.length, 30) < 62 - 0.25
    assert s.tracks["p2"].cam_t(s.length) == 46.5 + s.length - 2.0


def test_zoom_bigger_for_rarer_tiers_and_disable():
    cfg = {**DEFAULTS, "zoom": dict(DEFAULTS["zoom"])}
    z_rare = zoom.curve(300, 30, [(2.0, "rare")], cfg).max()
    z_top = zoom.curve(300, 30, [(2.0, "one in a krillion")], cfg).max()
    assert 1.14 < z_rare < 1.16 and z_top > z_rare
    cfg["zoom"]["enabled"] = False
    assert zoom.curve(300, 30, [(2.0, "rare")], cfg).max() == 1.0
