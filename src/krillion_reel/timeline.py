"""Master timeline: question segments that line both players up at each prompt.

Segment N starts `lead` seconds before each player's clock start for prompt N (so both prompt cards
appear together) and runs until the slower player clicks DESCEND. The faster player's *screen* holds
its last frame (the static tier card); their *cam* keeps rolling so faces never freeze.
An optional ending segment aligns both results screens.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import PLAYERS


@dataclass
class Track:
    start: float      # screen time at segment start (may be < 0: held first frame)
    length: float     # this player's own length; beyond it the screen holds
    reveal: float | None  # screen time of the reveal
    tier: str | None
    offset: float     # cam_offset (cam_t = screen_t + offset)
    submit: float | None = None  # screen time the answer was submitted (DIVE)
    # screen time this player's next part starts (next prompt's segment, or the results screen).
    # A held tier card may only run until here: past it their cam is already on the next question.
    hold_until: float | None = None

    def screen_t(self, r: float, fps: float) -> float:
        return self.start + min(r, max(0.0, self.length - 1.0 / fps))

    def cam_t(self, r: float) -> float:
        return self.start + r + self.offset


@dataclass
class Segment:
    n: int             # question number (0 = results ending, -1 = intro)
    master_start: float
    length: float
    tracks: dict[str, Track]

    @property
    def master_end(self) -> float:
        return self.master_start + self.length

    def reveal_master(self, p: str) -> float | None:
        t = self.tracks[p]
        return None if t.reveal is None else self.master_start + (t.reveal - t.start)


def build(sync: dict, cfg: dict) -> list[Segment]:
    lead = cfg["timeline"]["lead"]
    after = cfg["cut"]["protect_after"]
    drop = set(cfg.get("drop_questions") or [])
    offs = {p: float(sync["cam_offset"][p]["seconds"]) for p in PLAYERS}
    segs: list[Segment] = []
    m = 0.0
    intro = cfg["timeline"].get("intro")
    first_q = next((q for q in sync["questions"] if q["n"] not in drop and q.get(intro or "")), None)
    if intro in PLAYERS and first_q:
        # the host's intro: their cam (and idle game screen) from the start of the cam recording until
        # just before the first prompt. Segment time r = the intro team's cam time.
        start = -offs[intro]
        end = first_q[intro]["clock_start"] - lead
        if end - start > 1.0:
            segs.append(Segment(-1, m, end - start, {intro: Track(start=start, length=end - start, reveal=None,
                                                                  tier=None, offset=offs[intro])}))
            m += end - start
    res = sync.get("results") or {}
    qs = sync["questions"]
    for i, q in enumerate(qs):
        if q["n"] in drop or not all(q.get(p) for p in PLAYERS):
            continue
        tracks = {}
        for p in PLAYERS:
            e = q[p]
            # next part for this player, even if that question is dropped (its talk still isn't ours)
            nxt = next((x[p]["clock_start"] - lead for x in qs[i + 1:] if x.get(p)), None)
            if nxt is None and res.get(p) is not None:
                nxt = res[p] - 0.3
            start = e["clock_start"] - lead
            end = e.get("tier_end")
            if end is None:  # no card found: keep through the would-be reveal window
                end = (e.get("reveal") or e["submit"] + 4.0) + after + 1.0
            else:
                # stop just before the DESCEND click so the next prompt never flashes in; anything
                # shown past this holds the (static) tier card while the cam keeps rolling
                end -= 0.25
            if e.get("reveal") is not None:
                end = max(end, e["reveal"] + 0.35)  # card fully faded in
            tracks[p] = Track(start=start, length=end - start, reveal=e.get("reveal"), tier=e.get("tier"),
                              offset=offs[p], submit=e.get("submit"), hold_until=nxt)
        length = max(t.length for t in tracks.values())
        segs.append(Segment(q["n"], m, length, tracks))
        m += length
    if cfg["timeline"]["include_results"] and all(res.get(p) is not None for p in PLAYERS):
        hold = cfg["timeline"]["ending_hold"]
        # sync.json "results_window": {team: [screen_t0, screen_t1]} widens the ending so picks.json
        # "results" clips can keep post-game banter (the results screen holds while the cam rolls)
        win = sync.get("results_window") or {}
        tracks = {}
        for p in PLAYERS:
            a, b = win.get(p) or (res[p] - 0.3, res[p] + hold)
            tracks[p] = Track(start=a, length=b - a, reveal=None, tier=None, offset=offs[p])
        segs.append(Segment(0, m, max(t.length for t in tracks.values()), tracks))
    return segs


def find(segs: list[Segment], m: float) -> Segment:
    for s in segs:
        if s.master_start <= m < s.master_end:
            return s
    return segs[-1]
