"""Claude's picks (answering: model) are checked against the short-cut rules."""
import numpy as np

from krillion_reel import edl
from krillion_reel.config import DEFAULTS
from krillion_reel.timeline import Segment, Track


class FakeSig:
    """Nobody talks, nothing is loud: no reading detected, no extended reactions."""
    words = {"p1": [], "p2": []}
    db = {"p1": np.full(3000, -40.0), "p2": np.full(3000, -40.0)}

    def talking(self, p, t):
        return False

    def reacting(self, p, t):
        return False


def _seg():
    # both teams: segment starts at screen 0, submit at r 30, card at r 33, DESCEND at r 36
    tr = lambda tier: Track(start=0.0, length=36.0, reveal=33.0, tier=tier, offset=0.0, submit=30.0)  # noqa
    return Segment(1, 0.0, 36.0, {"p1": tr("rare"), "p2": tr("schooler")})


def _cfg():
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    # the rules as these tests were written for: 5s of thinking each, the last 10s before submitting
    hl = {**DEFAULTS["layout"]["highlights"], "think_after_read": 5.0, "think_after_read_other": 5.0,
          "end_before_submit": 10.0}
    cfg["layout"] = {**DEFAULTS["layout"], "answering": "model", "highlights": hl}
    return cfg


def _covered(clips, team, r0, r1):
    return sum(max(0.0, min(r1, c["r1"]) - max(r0, c["r0"])) for c in clips if c["team"] == team) / (r1 - r0)


def test_picks_get_missing_rules_added_and_end_is_never_split():
    cfg, seg = _cfg(), _seg()
    rev = {"p1": 33.0, "p2": 33.0}
    # the model only picks one middle moment for p2 - no reading, no first thinking
    picks = [{"team": "p2", "r0": 12.0, "r1": 15.0, "why": "joke", "speed": 1.0}]
    clips = edl.apply_picks(seg, FakeSig(), cfg, rev, 0, "Name a thing", picks)
    a = edl.anchors(seg, FakeSig(), cfg, rev, 0, "Name a thing")
    first = a["first"]
    assert clips[0]["team"] == first and clips[0]["r0"] == 0.0 and "rule:" in clips[0]["why"]
    for team, r0, r1, _ in edl.must_keep(a, cfg):
        assert _covered(clips, team, r0, r1) > 0.99, (team, r0, r1)
    # the model's pick survived, with its speed
    assert any(c["team"] == "p2" and c["r0"] <= 12.0 and c["r1"] >= 15.0 and c.get("speed") == 1.0 for c in clips)
    # the final shot (3s before submit -> reaction) is one uncut clip per team, at the end
    for p in ("p1", "p2"):
        fin = [c for c in clips if c["team"] == p and c["r0"] <= 27.0 + 1e-6 < c["r1"]]
        assert len(fin) == 1 and fin[0]["r1"] >= 33.0
    assert [c["team"] for c in clips[-2:]] == a["order"]


def test_picks_never_replay_and_stop_before_the_end_phase():
    cfg, seg = _cfg(), _seg()
    rev = {"p1": 33.0, "p2": 33.0}
    picks = [{"team": "p1", "r0": 0.0, "r1": 8.0, "why": "reads"},
             {"team": "p1", "r0": 5.0, "r1": 25.0, "why": "overlaps itself and runs into the end phase"}]
    clips = edl.apply_picks(seg, FakeSig(), cfg, rev, 0, None, picks)
    for p in ("p1", "p2"):
        shown = sorted((c["r0"], c["r1"]) for c in clips if c["team"] == p)
        assert all(b0 >= a1 - 1e-6 for (_, a1), (b0, _) in zip(shown, shown[1:])), shown
    model = [c for c in clips if "model:" in c["why"]]
    assert all(c["r1"] <= 20.0 + 1e-6 for c in model)  # end phase starts at submit - 10 = 20


def test_model_mode_without_picks_is_the_highlights_cut():
    cfg, seg = _cfg(), _seg()
    hl = {**cfg, "layout": {**cfg["layout"], "answering": "highlights"}}
    assert edl.plan_clips([seg], FakeSig(), cfg, {1: None}, None) == edl.plan_clips([seg], FakeSig(), hl, {1: None})


def test_rules_fill_gaps_without_overriding_picks_or_going_backwards():
    cfg, seg = _cfg(), _seg()
    rev = {"p1": 33.0, "p2": 33.0}
    # picks skip the reading; p1's pick sits inside its own "starts thinking" window, at 1x
    picks = [{"team": "p2", "r0": 8.0, "r1": 12.0, "why": "back-and-forth"},
             {"team": "p1", "r0": 6.0, "r1": 9.0, "why": "joke", "speed": 1.0}]
    clips = edl.apply_picks(seg, FakeSig(), cfg, rev, 0, None, picks)
    assert any(c["team"] == "p1" and c["r0"] <= 6.0 and c["r1"] >= 9.0 and c.get("speed") == 1.0 for c in clips)
    for p in ("p1", "p2"):  # every team's footage only moves forward
        mine = [c for c in clips if c["team"] == p]
        assert all(b["r0"] >= a["r1"] - 1e-6 for a, b in zip(mine, mine[1:])), mine


def test_reveal_never_holds_into_next_question():
    """Day 6: a deep cut reaction ran past DESCEND, so the card froze while the cam was already
    reading the next prompt (which then played again). The reveal shot must stop at the next part."""
    cfg = DEFAULTS
    # card at r 16, DESCEND at r 17.35 (track end), next question's segment starts at screen 20.6
    tr = Track(start=3.4, length=17.35, reveal=19.4, tier="deep cut", offset=0.0, submit=15.0, hold_until=20.6)
    seg = Segment(1, 0.0, 30.0, {"p1": tr})
    end = edl._reveal_end(seg, "p1", 16.0, None, cfg)
    assert end <= 17.35 + 1e-6, end
    # with a gap before the next part, the card may still hold a little past DESCEND (day 5)
    tr.hold_until = 25.0
    assert edl._reveal_end(seg, "p1", 16.0, None, cfg) > 17.35
