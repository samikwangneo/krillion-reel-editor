"""Step 2 - CLIPS + CUT DEAD SPACE: edl.json.

Only one team is on screen at a time, so the reel is a sequence of CLIPS, each a continuous run of
one team's footage (their cam + their game), in segment-relative time r (r = 0 is `lead` seconds
before that question's clock starts, for both teams). Per question, with layout.answering:

interleave (default) - cut between teams while they think, but each team keeps its own place, so
  going back to a team picks up exactly where it was left and no thinking is lost. A shot lasts at
  least min_shot and until it has caught up with where the other team was, then cuts right after the
  team on screen finishes a line. Then both reveals, lower tier first.
sequential - one team's whole think, then the other's, then both reveals.

alternate - both teams share one clock, so each team's thinking is only half shown:
  1. prompt + answering - both teams on a shared clock, cutting between them by who is talking
     (hold >= min_shot, force a switch after max_shot), up to the first reveal
  2. the first team's reveal: [reveal - reveal_before, reveal + reveal_after]
  3. the second team's reveal - continuing their footage if they were still answering, or rewinding
     a few seconds if both cards landed at nearly the same moment, so both reveals always play in full
Results: each team's final score as its own short clip.

Then dead space is cut inside each clip where that team is:
  * silent  (no WhisperX word and cam level below noise floor + N dB - laughs/gasps aren't
    transcribed, so loudness counts as "not silent" too)
  * on a static game screen (motion below threshold, or holding a finished question)
  * outside [reveal - protect_before, reveal + protect_after]
  * not still thinking (prompt card up to the answer being submitted) if cut.protect_thinking
for >= min_gap; each cut keeps `padding` on both sides. Every clip and cut is written with its
reason so it can be reviewed before rendering.

Finally every kept piece gets a playback speed: `speed.boring` (1.25x) for everything except the
reveal of a `speed.normal_tiers` card (deep cut / one in a krillion), which plays at 1x from just
before the card through the reaction (around_reveal, extended while they're still loud, up to
reaction_max). slow_submit / slow_reactions / slow_results bring back 1x for those too. Fast
stretches shorter than `min_fast` stay at normal speed so the pace doesn't flicker.
"""
from __future__ import annotations

import json

import numpy as np

from . import media, screen, timeline
from .config import Session
from .sync import PLAYERS, cam_track, cam_wav

GRID = 10  # Hz


def level_db(wav: np.ndarray, rate: int) -> np.ndarray:
    """Cam level in dB per 1/GRID second."""
    hop = rate // GRID
    m = len(wav) // hop
    rms = np.sqrt((wav[: m * hop].reshape(m, hop) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms)


def speech_activity(words: list[dict], db: np.ndarray, db_above: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-1/GRID-second masks in cam time: (has_word, loud)."""
    n = len(db) + 1
    has_word = np.zeros(n, bool)
    for w in words:
        has_word[max(0, int((w["start"] - 0.1) * GRID)):int((w["end"] + 0.1) * GRID) + 1] = True
    loud = np.zeros(n, bool)
    loud[:len(db)] = db > np.percentile(db, 10) + db_above
    return has_word, loud


def reaction_mask(db: np.ndarray, pct: float, pad: tuple[float, float]) -> np.ndarray:
    """True where a loud reaction is happening, about to, or just did (cam time, GRID Hz).
    Loud = smoothed level in this team's own top (100 - pct)%, so mic gain doesn't matter."""
    sm = np.convolve(np.pad(db, 2, mode="edge"), np.ones(5) / 5, "valid")
    hot = sm > np.percentile(sm, pct)
    before, after = int(pad[0] * GRID), int(pad[1] * GRID)
    # t counts if a hot sample lies in [t - after, t + before]
    full = np.convolve(hot.astype(float), np.ones(before + after + 1), "full")
    return np.append(full[before:before + len(hot)] > 0, False)


def _at(mask: np.ndarray, t: float) -> bool:
    i = int(t * GRID)
    return bool(mask[i]) if 0 <= i < len(mask) else False


class Signals:
    def __init__(self, sess: Session, cfg: dict):
        c = cfg["cut"]
        sp = cfg["speed"]
        self.speech, self.motion, self.react, self.words, self.db = {}, {}, {}, {}, {}
        win = int(0.3 * screen.PROXY_FPS)
        for p in PLAYERS:
            words = json.loads((sess.work / f"{p}_words.json").read_text(encoding="utf-8"))
            self.words[p] = words
            wav = media.audio_track(sess.files[f"{p}_cam"], cam_track(sess, cfg, p), cam_wav(sess, cfg, p))
            db = level_db(wav, 16000)
            self.db[p] = db
            self.speech[p] = speech_activity(words, db, c["speech_db_above_floor"])
            self.react[p] = reaction_mask(db, sp["reaction_percentile"], sp["reaction_pad"])
            mo = np.load(sess.work / f"{p}_screen_signals.npz")["motion"]
            # a pixel change counts for +-0.3s so single quiet frames inside activity don't cut
            self.motion[p] = np.array([mo[max(0, i - win):i + win + 1].max() for i in range(len(mo))])

    def talking(self, p: str, cam_t: float) -> bool:
        w, loud = self.speech[p]
        return _at(w, cam_t) or _at(loud, cam_t)

    def reacting(self, p: str, cam_t: float) -> bool:
        return _at(self.react[p], cam_t)

    def screen_motion(self, p: str, screen_t: float) -> float:
        i = int(screen_t * screen.PROXY_FPS)
        return float(self.motion[p][i]) if 0 <= i < len(self.motion[p]) else 0.0


def _alternate(seg: timeline.Segment, sig: Signals, cfg: dict, r_end: float, start: int) -> list[tuple]:
    """Shared-clock answering phase [0, r_end): returns [(team_idx, r0, r1, why)]."""
    lay = cfg["layout"]
    fps = cfg["output"]["fps"]
    n = int(np.ceil(r_end * GRID))
    if n <= 0:
        return []
    team = np.zeros(n, int)
    why = [""] * n
    cur, shot_start, reason = start, 0.0, "prompt"
    for i in range(n):
        r = i / GRID
        other = 1 - cur
        held = {j: r >= seg.tracks[PLAYERS[j]].length - 1.0 / fps for j in (0, 1)}
        talk = {j: sig.talking(PLAYERS[j], seg.tracks[PLAYERS[j]].cam_t(r)) for j in (0, 1)}
        dur = r - shot_start
        if held[cur] and not held[other]:
            cur, shot_start, reason = other, r, "other team's question is over"
        elif dur >= lay["min_shot"] and talk[other] and not talk[cur] and not held[other]:
            cur, shot_start, reason = other, r, "following who's talking"
        elif dur >= lay["max_shot"] and not held[other]:
            cur, shot_start, reason = other, r, f"alternating after {lay['max_shot']:.0f}s"
        team[i], why[i] = cur, reason
    # fold blink shots (< 1s) into the previous shot
    i = 0
    while i < n:
        j = i
        while j < n and team[j] == team[i]:
            j += 1
        if j - i < GRID and i > 0:
            team[i:j] = team[i - 1]
            i = 0
            continue
        i = j
    out, i = [], 0
    while i < n:
        j = i
        while j < n and team[j] == team[i]:
            j += 1
        out.append((int(team[i]), i / GRID, min(r_end, j / GRID), why[i]))
        i = j
    return out


def _think_split(seg: timeline.Segment, rev: dict, cfg: dict, start: int) -> tuple[list[str], dict]:
    """Reveal order (lower tier first so the better card lands last; on a tie the first team
    alternates per question) and where each team's thinking ends (just after they submit)."""
    lay, tiers = cfg["layout"], cfg["tiers"]
    rank = {p: tiers.index(seg.tracks[p].tier) if seg.tracks[p].tier in tiers else -1 for p in PLAYERS}
    order = sorted(PLAYERS, key=lambda p: (rank[p], PLAYERS.index(p) != start))
    split = {}
    for p in PLAYERS:
        tr = seg.tracks[p]
        reveal_in = rev[p] - lay["reveal_before"]
        sub = (tr.submit - tr.start + cfg["speed"]["around_submit"][1]) if tr.submit is not None else reveal_in
        split[p] = max(0.0, min(sub, reveal_in))
    return order, split


def _reveal_end(seg: timeline.Segment, p: str, rev: float, sig: Signals | None, cfg: dict) -> float:
    """Normal tiers: reveal_after past the card (or the DESCEND click, whichever is first). Deep cut /
    one in a krillion: at least big_reveal_after, and on while they're still loud (up to
    speed.reaction_max) - past DESCEND too, holding the tier card while the cam keeps rolling."""
    lay, sp = cfg["layout"], cfg["speed"]
    tr = seg.tracks[p]
    if tr.tier not in sp.get("normal_tiers", []) or sig is None:
        return min(tr.length, rev + lay["reveal_after"])
    end = rev + lay.get("big_reveal_after", 4.0)
    while end < rev + sp.get("reaction_max", 6.0) and sig.reacting(p, tr.cam_t(end)):
        end += 1 / GRID
    return end


def _reveal_clips(seg: timeline.Segment, rev: dict, cfg: dict, order: list[str], split: dict,
                  sig: Signals | None = None) -> list[dict]:
    return [{"question": seg.n, "team": p, "r0": round(split[p], 2),
             "r1": round(_reveal_end(seg, p, rev[p], sig, cfg), 2),
             "why": f"{p} reveal ({seg.tracks[p].tier})"} for p in order]


def _sequential(seg: timeline.Segment, rev: dict, cfg: dict, start: int, sig: Signals | None = None) -> list[dict]:
    """Each team's whole think, one team after the other, then both reveals back to back."""
    order, split = _think_split(seg, rev, cfg, start)
    clips = [{"question": seg.n, "team": p, "r0": 0.0, "r1": round(split[p], 2),
              "why": f"{p} thinking (prompt -> answer submitted)"} for p in order]
    return clips + _reveal_clips(seg, rev, cfg, order, split, sig)


def _read_end(seg: timeline.Segment, sig: Signals, p: str, prompt: str | None, cfg: dict) -> tuple[float, bool]:
    """When team p finishes reading the prompt aloud (segment r), and whether they read it at all.
    Reading = a run of words that are in the prompt, starting within read_max of the segment start."""
    import re
    hl = cfg["layout"]["highlights"]
    tr = seg.tracks[p]
    lead = cfg["timeline"]["lead"]
    fallback = lead + hl["read_fallback"]
    if not prompt:
        return fallback, False

    def norm(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", x.lower())

    pset = {norm(x) for x in prompt.replace("-", " ").split() if norm(x)}
    base = tr.start + tr.offset  # cam_t = base + r
    ws = [(w["start"] - base, w["end"] - base, norm(w["word"])) for w in sig.words[p]]
    ws = [w for w in ws if 0.0 <= w[0] <= lead + hl["read_max"]]
    for i, (s0, e0, t0) in enumerate(ws):
        if t0 not in pset or sum(w[2] in pset for w in ws[i:i + 4]) < 3:
            continue
        end, j = min(e0, s0 + 1.0), i + 1
        while j < len(ws) and ws[j][0] - end <= 1.0 and (
                ws[j][2] in pset or (j + 1 < len(ws) and ws[j + 1][2] in pset)):
            end = max(end, min(ws[j][1], ws[j][0] + 1.0))
            j += 1
        return end + 0.3, True
    return fallback, False


def _liveliest(seg: timeline.Segment, sig: Signals, p: str, a: float, b: float,
               length: float) -> tuple[float, float] | None:
    """[x, x + length) inside [a, b) where team p is loudest and talking most."""
    if b - a < length:
        return None
    tr = seg.tracks[p]
    db = sig.db[p]
    floor = float(np.median(db))
    n = int(length * GRID)
    best = None
    for k in range(int(round((b - a - length) * GRID)) + 1):
        x = a + k / GRID
        idx = [int(tr.cam_t(x + i / GRID) * GRID) for i in range(n)]
        lv = [db[i] - floor for i in idx if 0 <= i < len(db)]
        talk = sum(sig.talking(p, tr.cam_t(x + i / GRID)) for i in range(n)) / n
        score = (float(np.mean(np.clip(lv, 0, None))) if lv else 0.0) + 5.0 * talk
        if best is None or score > best[1]:
            best = (x, score)
    return (best[0], best[0] + length)


def anchors(seg: timeline.Segment, sig: Signals, cfg: dict, rev: dict, start: int,
            prompt: str | None) -> dict:
    """The fixed moments every short cut is built around (segment time r):
      first        team shown reading the prompt (one that read it aloud if any did)
      read_end     {team: when they finish reading (or the fallback), read aloud?}
      end_start    {team: start of their last end_before_submit seconds}
      final_start  {team: start of their uncut final shot (no_cut_before_submit before submit)}
      reveal_end   {team: end of their reveal + reaction}
      order        reveal order (lower tier first)"""
    hl = cfg["layout"]["highlights"]
    order, split = _think_split(seg, rev, cfg, start)
    sub = {}
    for p in PLAYERS:
        tr = seg.tracks[p]
        sub[p] = (tr.submit - tr.start) if tr.submit is not None else split[p]
    reads = {p: _read_end(seg, sig, p, prompt, cfg) for p in PLAYERS}
    first = sorted(PLAYERS, key=lambda p: (not reads[p][1], PLAYERS.index(p) != start))[0]
    end_start = {p: max(0.0, sub[p] - hl["end_before_submit"]) for p in PLAYERS}
    return {
        "first": first,
        "other": PLAYERS[1 - PLAYERS.index(first)],
        "read_end": {p: (min(reads[p][0], end_start[p]), reads[p][1]) for p in PLAYERS},
        "submit": sub,
        "end_start": end_start,
        "final_start": {p: max(0.0, sub[p] - hl["no_cut_before_submit"]) for p in PLAYERS},
        "reveal_end": {p: _reveal_end(seg, p, rev[p], sig, cfg) for p in PLAYERS},
        "order": order,
    }


def must_keep(a: dict, cfg: dict) -> list[tuple[str, float, float, str]]:
    """Windows the short cut always shows, in play order: (team, r0, r1, what)."""
    hl = cfg["layout"]["highlights"]
    first, other = a["first"], a["other"]
    r_end, o_end = a["read_end"][first][0], a["read_end"][other][0]
    win = [(first, 0.0, r_end, "reading the prompt" if a["read_end"][first][1] else "prompt"),
           (first, r_end, min(r_end + hl["think_after_read"], a["end_start"][first]), "starts thinking"),
           (other, o_end, min(o_end + hl["think_after_read"], a["end_start"][other]),
            "starts thinking (just finished reading)")]
    win += [(p, a["end_start"][p], a["final_start"][p], "last seconds before submitting") for p in a["order"]]
    win += [(p, a["final_start"][p], a["reveal_end"][p], "submits -> reveal") for p in a["order"]]
    return [w for w in win if w[2] - w[1] >= 0.5]


def _highlights(seg: timeline.Segment, sig: Signals, cfg: dict, rev: dict, start: int,
                prompt: str | None) -> list[dict]:
    """Shorter cut of a question:
      1. one team reading the prompt aloud (from the start until they finish),
      2. think_after_read seconds of each team right after they finish reading,
      3. the liveliest `middle` seconds of each team's discussion in between (the rest is cut),
      4. both teams' last end_before_submit seconds, interleaved; each team's final shot runs
         uninterrupted from no_cut_before_submit before their submit through the card + reaction.
    Overlapping pieces of a team (short thinks) are trimmed, so nothing plays twice."""
    hl = cfg["layout"]["highlights"]
    a = anchors(seg, sig, cfg, rev, start, prompt)
    order, first, other = a["order"], a["first"], a["other"]
    clips: list[dict] = []
    used: dict[str, list] = {p: [] for p in PLAYERS}

    def add(p, a, b, why):
        a = max(0.0, a)
        for u0, u1 in sorted(used[p]):  # drop the parts of [a, b) this team already has
            if a < u1 and b > u0:
                if a >= u0:
                    a = u1
                else:
                    b = min(b, u0)
        if b - a < 0.5:
            return
        used[p].append((a, b))
        clips.append({"question": seg.n, "team": p, "r0": round(a, 2), "r1": round(b, 2), "why": f"{p} {why}"})

    end_start, final_start = a["end_start"], a["final_start"]
    for p in PLAYERS:  # reserve the end phase so the earlier phases stop before it
        used[p].append((end_start[p], 1e9))
    # 1 + 2: reading, then each team's first seconds of thinking
    r_end, o_end = a["read_end"][first][0], a["read_end"][other][0]
    add(first, 0.0, r_end, "reading the prompt" if a["read_end"][first][1] else "prompt")
    add(first, r_end, r_end + hl["think_after_read"], "starts thinking")
    add(other, o_end, o_end + hl["think_after_read"], "starts thinking (just finished reading)")
    # 3: liveliest bit of each team's middle, in time order
    mids = []
    for p, e in ((first, r_end), (other, o_end)):
        w = _liveliest(seg, sig, p, e + hl["think_after_read"], end_start[p], hl["middle"])
        if w:
            mids.append((w[0], p, w))
    for _, p, w in sorted(mids):
        add(p, w[0], w[1], "liveliest moment of the discussion")
    # 4: last seconds before submitting, interleaved (lower tier first, like the reveals)
    for p in PLAYERS:
        used[p] = [u for u in used[p] if u[1] < 1e8]
    for p in order:
        add(p, end_start[p], final_start[p], "last seconds before submitting")
    for p in order:
        add(p, final_start[p], a["reveal_end"][p], f"submits -> reveal ({seg.tracks[p].tier})")
    _close_gaps(clips)
    return clips


def _close_gaps(clips: list[dict]) -> None:
    """Back-to-back clips of one team with a tiny gap would be a jarring micro-jump: play through it."""
    for prev, cur in zip(clips, clips[1:]):
        if prev["team"] == cur["team"] and 0 < cur["r0"] - prev["r1"] <= 1.5:
            cur["r0"] = prev["r1"]


def _minus(a: float, b: float, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of [a, b) not covered by spans."""
    out, pos = [], a
    for s0, s1 in sorted(spans):
        if s1 <= pos or s0 >= b:
            continue
        if s0 > pos:
            out.append((pos, s0))
        pos = max(pos, s1)
    if pos < b:
        out.append((pos, b))
    return out


def apply_picks(seg: timeline.Segment, sig: Signals, cfg: dict, rev: dict, start: int,
                prompt: str | None, picks: list[dict]) -> list[dict]:
    """Clips chosen by Claude (out/<session>/picks.json) for one question, checked against the rules.

    The picks cover the opening and the middle; the model's choices win, and the validator only:
      * clamps each pick to the question and stops it before that team's end phase,
      * fills in the parts of the must-keep windows (reading, the first seconds of thinking) that the
        picks leave out - each placed before that team's next later clip, so a team never jumps
        backwards, and the question always opens on the prompt,
      * never plays a moment twice and never moves a team back in time (later overlaps are trimmed),
      * always appends both teams' last seconds + uncut submit -> reveal shots, in reveal order,
      * closes tiny same-team gaps.
    Every clip says where it came from: "model: ..." or "rule: ...". A pick may carry
    "speed": 1.0 to keep that moment at normal speed (big reveals are always 1x anyway)."""
    a = anchors(seg, sig, cfg, rev, start, prompt)
    end_start = a["end_start"]
    seq: list[dict] = []
    for c in picks:
        p = c.get("team")
        if p not in PLAYERS:
            continue
        r0, r1 = max(0.0, float(c.get("r0", 0))), min(float(c.get("r1", 0)), end_start[p])
        if r1 - r0 >= 0.5:
            seq.append({"team": p, "r0": r0, "r1": r1, "why": f"model: {c.get('why', '')}".rstrip(": "),
                        "speed": c.get("speed")})
    must = must_keep(a, cfg)
    opening = [w for w in must if w[3] not in ("last seconds before submitting", "submits -> reveal")]
    for i, (p, w0, w1, what) in enumerate(opening):
        have = [(c["r0"], c["r1"]) for c in seq if c["team"] == p]
        for g0, g1 in _minus(w0, w1, have):
            if g1 - g0 < 0.5:
                continue
            rule = {"team": p, "r0": g0, "r1": g1, "why": f"rule: {what} (missing from picks)", "speed": None}
            if i == 0 and g0 <= 0.5:  # the prompt reading always opens the question
                seq.insert(0, rule)
                continue
            later = [k for k, c in enumerate(seq) if c["team"] == p and c["r0"] >= g1 - 1e-6]
            seq.insert(later[0] if later else len(seq), rule)
    out: list[dict] = []
    shown_to = {p: 0.0 for p in PLAYERS}  # each team only moves forward in time

    def add(p, r0, r1, why, speed=None):
        r0 = max(float(r0), shown_to[p] if any(c["team"] == p for c in out) else 0.0)
        if r1 - r0 < 0.5:
            return
        shown_to[p] = r1
        c = {"question": seg.n, "team": p, "r0": round(r0, 2), "r1": round(r1, 2), "why": f"{p} {why}"}
        if speed is not None:
            c["speed"] = float(speed)
        out.append(c)

    for c in seq:
        add(c["team"], c["r0"], c["r1"], c["why"], c["speed"])
    for p in a["order"]:
        add(p, max(end_start[p], shown_to[p]), a["final_start"][p], "rule: last seconds before submitting")
    for p in a["order"]:
        add(p, a["final_start"][p], a["reveal_end"][p], f"rule: submits -> reveal ({seg.tracks[p].tier})")
    _close_gaps(out)
    return out


def _pause_after(sig: Signals, p: str, tr: timeline.Track, lo: float, hi: float) -> tuple[float, str]:
    """Best place in [lo, hi] to cut away from team p, from their WhisperX words: just after the
    first sentence end (or a 0.5s+ breath), else in the biggest gap between words, else hi."""
    base = tr.start + tr.offset  # r = cam_t - base
    ws = [((w["start"] - base), min(w["end"], w["start"] + 1.0) - base, w["word"]) for w in sig.words[p]]
    ws = [w for w in ws if lo - 2.0 <= w[1] and w[0] <= hi + 2.0]
    if not any(lo <= w[1] <= hi or lo <= w[0] <= hi for w in ws):
        return lo, "at a quiet moment"
    gaps = []  # (cut point, gap length, sentence end)
    for (s0, e0, t0), (s1, _, _) in zip(ws, ws[1:]):
        cut = e0 + min(0.15, max(0.0, s1 - e0) / 2)
        if lo <= cut <= hi:
            gaps.append((cut, s1 - e0, t0.rstrip()[-1:] in ".?!"))
    if ws and lo <= ws[-1][1] + 0.15 <= hi:
        gaps.append((ws[-1][1] + 0.15, 9.0, True))
    for cut, gap, end in gaps:
        if (end and gap >= 0.2) or gap >= 0.5:
            return cut, "after they finish a line"
    if gaps and max(g[1] for g in gaps) >= 0.1:
        cut = max(gaps, key=lambda g: g[1])[0]
        return cut, "between words"
    return hi, "no pause found"


def _interleave(seg: timeline.Segment, sig: Signals, cfg: dict, rev: dict, start: int) -> list[dict]:
    """Cut between teams while both think, but each team keeps its own place: going back to a team
    resumes exactly where it was left, so no thinking is skipped. A shot runs at least min_shot and
    until it has caught up with where the other team was left, then ends right after the team on
    screen finishes a line (searching up to pause_search further). Both reveals follow."""
    lay = cfg["layout"]
    order, split = _think_split(seg, rev, cfg, start)
    pos = {p: 0.0 for p in PLAYERS}
    cur = PLAYERS[start]
    clips = []
    while any(pos[p] < split[p] - 0.05 for p in PLAYERS):
        oth = PLAYERS[1 - PLAYERS.index(cur)]
        if pos[cur] >= split[cur] - 0.05:
            cur = oth
            continue
        if pos[oth] >= split[oth] - 0.05:
            end, why = split[cur], "rest of their thinking (other team has submitted)"
        else:
            lo = max(pos[cur] + lay["min_shot"], pos[oth])
            if lo >= split[cur] - lay["min_shot"]:  # don't leave a tiny leftover
                end, why = split[cur], "thinking up to their submit"
            else:
                end, why = _pause_after(sig, cur, seg.tracks[cur], lo,
                                        min(lo + lay["pause_search"], split[cur]))
                if split[cur] - end < lay["min_shot"]:
                    end, why = split[cur], "thinking up to their submit"
        label = "prompt" if pos[cur] == 0 else "picks up where they were"
        clips.append({"question": seg.n, "team": cur, "r0": round(pos[cur], 2), "r1": round(end, 2),
                      "why": f"{cur} {label}, cut {why}"})
        pos[cur] = end
        cur = oth
    return clips + _reveal_clips(seg, rev, cfg, order, split, sig)


def plan_clips(segs: list[timeline.Segment], sig: Signals, cfg: dict,
               prompts: dict | None = None, picks: dict | None = None) -> list[dict]:
    lay = cfg["layout"]
    clips: list[dict] = []
    start = 0
    for seg in segs:
        if seg.n == 0:
            for p in PLAYERS:
                clips.append({"question": 0, "team": p, "r0": 0.0, "r1": round(seg.tracks[p].length, 2),
                              "why": "final score"})
            continue
        rev = {}
        for p in PLAYERS:
            tr = seg.tracks[p]
            rev[p] = (tr.reveal - tr.start) if tr.reveal is not None else tr.length - lay["reveal_after"]
        mode = lay.get("answering", "interleave")
        q_picks = (picks or {}).get(str(seg.n), {}).get("clips") if mode == "model" else None
        if q_picks:
            clips += apply_picks(seg, sig, cfg, rev, start, (prompts or {}).get(seg.n), q_picks)
            start = 1 - start
            continue
        if mode in ("highlights", "model"):  # model without picks for this question -> the rule cut
            clips += _highlights(seg, sig, cfg, rev, start, (prompts or {}).get(seg.n))
            start = 1 - start
            continue
        if mode in ("interleave", "sequential"):
            clips += (_interleave(seg, sig, cfg, rev, start) if mode == "interleave"
                      else _sequential(seg, rev, cfg, start, sig))
            start = 1 - start
            continue
        a, b = sorted(PLAYERS, key=lambda p: rev[p])
        a_start = max(0.0, rev[a] - lay["reveal_before"])
        for j, r0, r1, why in _alternate(seg, sig, cfg, a_start, start):
            clips.append({"question": seg.n, "team": PLAYERS[j], "r0": round(r0, 2), "r1": round(r1, 2), "why": why})
        a_end = min(seg.tracks[a].length, rev[a] + lay["reveal_after"])
        clips.append({"question": seg.n, "team": a, "r0": round(a_start, 2), "r1": round(a_end, 2),
                      "why": f"{a} reveal ({seg.tracks[a].tier})"})
        b_start = rev[b] - lay["reveal_before"]
        if b_start > a_end:
            b_start, why = a_end, f"{b} still answering, then reveal ({seg.tracks[b].tier})"
        else:
            why = f"{b} reveal ({seg.tracks[b].tier})" + (" - replayed from just before their card"
                                                          if b_start < a_end else "")
        b_end = min(seg.tracks[b].length, rev[b] + lay["reveal_after"])
        clips.append({"question": seg.n, "team": b, "r0": round(max(0.0, b_start), 2), "r1": round(b_end, 2),
                      "why": why})
        start = 1 - PLAYERS.index(b)  # open the next question on the team we didn't just end on
    # merge adjacent clips of the same team that are continuous
    merged: list[dict] = []
    for c in clips:
        m = merged[-1] if merged else None
        if (m and m["question"] == c["question"] and m["team"] == c["team"] and abs(m["r1"] - c["r0"]) < 1e-6
                and m.get("speed") == c.get("speed")):
            m["r1"] = c["r1"]
            m["why"] += " + " + c["why"]
        else:
            merged.append(dict(c))
    return [c for c in merged if c["r1"] - c["r0"] > 0.05]


def load_picks(sess: Session) -> dict | None:
    """out/<session>/picks.json -> {"<question>": {"clips": [...]}}, or None if there isn't one."""
    f = sess.out / "picks.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    return {str(k): v for k, v in (d.get("questions") or {}).items()}


def build(sess: Session, sync: dict, cfg: dict, log=print) -> dict:
    c = cfg["cut"]
    fps = cfg["output"]["fps"]
    segs = {s.n: s for s in timeline.build(sync, cfg)}
    sig = Signals(sess, cfg)
    picks = load_picks(sess) if cfg["layout"].get("answering") == "model" else None
    clips = plan_clips(list(segs.values()), sig, cfg, {q['n']: q.get('prompt') for q in sync['questions']}, picks)
    keeps, cuts = [], []
    out_t = 0.0
    for cl in clips:
        seg, p = segs[cl["question"]], cl["team"]
        tr = seg.tracks[p]
        rr = (tr.reveal - tr.start) if tr.reveal is not None else None
        sr = (tr.submit - tr.start) if tr.submit is not None else None
        r0, r1 = cl["r0"], cl["r1"]
        n = int(round((r1 - r0) * GRID))
        cut_ok = np.zeros(n, bool)
        state = []
        for i in range(n):
            r = r0 + i / GRID
            held = r >= tr.length - 1.0 / fps
            static = held or sig.screen_motion(p, tr.screen_t(r, fps)) <= c["static_motion"]
            silent = not sig.talking(p, tr.cam_t(r))
            protected = rr is not None and rr - c["protect_before"] <= r < rr + c["protect_after"]
            thinking = c["protect_thinking"] and sr is not None and r < sr + 0.5
            cut_ok[i] = silent and static and not protected and not thinking and cl["question"] != 0
            state.append("held" if held else "static")
        spans, pos = [], r0
        for s, e in screen.runs(cut_ok, GRID, min_len=c["min_gap"], max_gap=0.0):
            a = r0 + s + (0 if s == 0 else c["padding"])          # clip edges are cuts already
            b = r0 + e - (0 if r0 + e >= r1 - 0.05 else c["padding"])
            if b - a < 0.2:
                continue
            if a > pos:
                spans.append((pos, a))
            cuts.append({"question": cl["question"], "team": p, "r0": round(a, 3), "r1": round(b, 3),
                         "reason": f"Q{cl['question'] or 'results'}: {p} silent {e - s:.1f}s, game {state[int(s * GRID)]}"})
            pos = b
        if r1 > pos:
            spans.append((pos, r1))
        for a, b in spans:
            for x0, x1, speed, why in _speed_runs(a, b, cl, tr, rr, sr, sig, cfg):
                if cl.get("speed") is not None and why == "boring":
                    speed, why = float(cl["speed"]), "picked speed"
                keeps.append({"question": cl["question"], "team": p, "r0": round(x0, 3), "r1": round(x1, 3),
                              "speed": speed, "speed_why": why, "out_start": round(out_t, 3)})
                out_t += (x1 - x0) / speed

    per_q: dict = {}
    for k in keeps:
        per_q[k["question"]] = per_q.get(k["question"], 0) + (k["r1"] - k["r0"]) / k["speed"]
    fast = [k for k in keeps if k["speed"] != 1.0]
    return {
        "fps": fps,
        "total_seconds": round(out_t, 2),
        "kept_seconds": round(sum(k["r1"] - k["r0"] for k in keeps), 2),
        "sped_up_seconds": round(sum(k["r1"] - k["r0"] for k in fast), 2),
        "clips_seconds": round(sum(cl["r1"] - cl["r0"] for cl in clips), 2),
        "per_question_seconds": {str(k): round(v, 2) for k, v in per_q.items()},
        "clips": clips,
        "keep": keeps,
        "cuts": cuts,
    }


def _speed_runs(a: float, b: float, cl: dict, tr: timeline.Track, rr: float | None, sr: float | None,
                sig: Signals, cfg: dict) -> list[tuple[float, float, float, str]]:
    """Split kept span [a, b) into runs of (r0, r1, speed, why)."""
    sp = cfg["speed"]
    fast = float(sp["boring"])
    if not sp["enabled"] or fast == 1.0:
        return [(a, b, 1.0, "normal")]
    if cl["question"] == 0:
        return [(a, b, 1.0, "results") if sp.get("slow_results") else (a, b, fast, "boring")]
    big = tr.tier in sp.get("normal_tiers", [])  # deep cut / one in a krillion: their reveal plays at 1x
    n = max(1, int(round((b - a) * GRID)))
    why = []
    for i in range(n):
        r = a + (i + 0.5) / GRID
        if big and rr is not None and rr - sp["around_reveal"][0] <= r < rr + sp["around_reveal"][1]:
            why.append("reveal")
        elif big and rr is not None and rr <= r < rr + sp.get("reaction_max", 6.0) and sig.reacting(cl["team"], tr.cam_t(r)):
            why.append("reaction")  # keep the big reaction at 1x while it lasts
        elif sp.get("slow_submit") and sr is not None and sr - sp["around_submit"][0] <= r < sr + sp["around_submit"][1]:
            why.append("submitting")
        elif sp.get("slow_reactions") and sig.reacting(cl["team"], tr.cam_t(r)):
            why.append("reaction")
        else:
            why.append("")

    def runs():
        i = 0
        while i < n:
            j = i
            while j < n and (why[j] == "") == (why[i] == ""):
                j += 1
            yield i, j
            i = j

    for i, j in list(runs()):  # sub-0.5s edge-of-a-reaction blips aren't worth slowing down for
        if set(why[i:j]) == {"reaction"} and (j - i) / GRID < 0.5:
            why[i:j] = [""] * (j - i)
    for i, j in list(runs()):  # fast stretches too short to be worth it stay normal
        if why[i] == "" and (j - i) / GRID < sp["min_fast"]:
            why[i:j] = ["short"] * (j - i)
    out = []
    for i, j in runs():
        x0, x1 = a + i / GRID, (b if j == n else a + j / GRID)
        if why[i] == "":
            out.append((x0, x1, fast, "boring"))
        else:
            out.append((x0, x1, 1.0, "+".join(sorted(set(why[i:j])))))
    return out


def suggest_drops(sync: dict, edl: dict, cfg: dict) -> list[dict]:
    """Rank questions for dropping when the reel is too long: lowest combined tier first (least
    exciting reveals), ties broken by longest duration (dropping it saves the most)."""
    tiers = cfg["tiers"]
    rows = []
    for q in sync["questions"]:
        n = q["n"]
        dur = edl["per_question_seconds"].get(str(n))
        if dur is None:
            continue
        rank = sum(tiers.index(q[p]["tier"]) if q.get(p) and q[p].get("tier") in tiers else 0 for p in PLAYERS)
        rows.append({"q": n, "seconds": dur, "tier_rank": rank,
                     "tiers": [q[p].get("tier") if q.get(p) else None for p in PLAYERS],
                     "answers": [q[p].get("answer") if q.get(p) else None for p in PLAYERS],
                     "prompt": q.get("prompt")})
    rows.sort(key=lambda r: (r["tier_rank"], -r["seconds"]))
    over = edl["total_seconds"] - cfg["output"]["max_seconds"]
    picked, saved = [], 0.0
    for r in rows:
        if saved >= over:
            break
        picked.append(r)
        saved += r["seconds"]
    return picked


def load_or_build(sess: Session, sync: dict, cfg: dict, rebuild: bool = True, log=print) -> dict:
    if sess.edl_json.exists() and not rebuild:
        return json.loads(sess.edl_json.read_text(encoding="utf-8"))
    edl = build(sess, sync, cfg, log=log)
    sess.edl_json.write_text(json.dumps(edl, indent=2), encoding="utf-8")
    log(f"[cut] wrote {sess.edl_json}: {len(edl['clips'])} clips, {len(edl['cuts'])} cuts, "
        f"{edl['clips_seconds']:.1f}s -> {edl['total_seconds']:.1f}s")
    return edl
