"""3+ players: one featured member per group, a quick look at the others' cards, and the 2:59 fit."""
import numpy as np

from krillion_reel import edl
from krillion_reel.config import DEFAULTS
from krillion_reel.timeline import Segment, Track


class FakeSig:
    words = {p: [] for p in ("p1", "p2", "p3")}
    db = {"p1": np.full(3000, -40.0), "p2": np.full(3000, -40.0), "p3": np.full(3000, -40.0)}

    def talking(self, p, t):
        return False

    def reacting(self, p, t):
        return False


def _seg(tiers):
    tr = lambda tier: Track(start=0.0, length=36.0, reveal=33.0, tier=tier, offset=0.0, submit=30.0)  # noqa
    return Segment(1, 0.0, 36.0, {p: tr(t) for p, t in tiers.items()})


def _cfg():
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    cfg["layout"] = {**DEFAULTS["layout"], "answering": "highlights"}
    cfg["groups"] = [["p1", "p2"]]
    return cfg


def test_better_roommate_is_featured_and_other_gets_a_flash_in_tier_order():
    seg = _seg({"p1": "plankton", "p2": "rare", "p3": "schooler"})
    shown, flashed, _ = edl.split_teams(seg, FakeSig(), _cfg())
    assert shown == ["p2", "p3"] and flashed == ["p1"]
    clips = edl.plan_clips([seg], FakeSig(), _cfg())
    assert {c["team"] for c in clips} == {"p1", "p2", "p3"}
    p1 = [c for c in clips if c["team"] == "p1"]
    assert len(p1) == 1 and p1[0]["r0"] == 32.7 and p1[0]["r1"] == 34.7  # 0.3s before -> 1.7s after the card
    # reveal shots in tier order: p1 plankton flash, p3 schooler, p2 rare
    reveals = [c["team"] for c in clips if c["r0"] <= 33.0 < c["r1"]]
    assert reveals == ["p1", "p3", "p2"], reveals


def test_featured_override_from_picks():
    seg = _seg({"p1": "plankton", "p2": "rare", "p3": "schooler"})
    shown, flashed, why = edl.split_teams(seg, FakeSig(), _cfg(), {"1": {"featured": "p1"}})
    assert shown == ["p1", "p3"] and flashed == ["p2"] and why["p1"] == "picked in picks.json"


def test_fit_lands_exactly_on_target_frames():
    cfg = _cfg()
    keeps = [{"r0": 0.0, "r1": 100.0, "speed": 1.25}, {"r0": 0.0, "r1": 140.0, "speed": 1.25, "flex": True}]
    fit = edl.fit_to_target(keeps, cfg)
    assert fit["status"] == "exact"
    assert sum(edl._frames(k, 30) for k in keeps) == 179 * 30
    assert all(1.25 < k["speed"] <= 1.5 for k in keeps if k.get("flex"))
    too_long = [{"r0": 0.0, "r1": 300.0, "speed": 1.25}, {"r0": 0.0, "r1": 10.0, "speed": 1.25, "flex": True}]
    assert edl.fit_to_target(too_long, cfg)["status"] == "long"
