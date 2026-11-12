"""Step 2 - CLIPS + CUT DEAD SPACE: edl.json.

Only one team is on screen at a time, so the reel is a sequence of CLIPS, each a continuous run of
one team's footage (their cam + their game), in segment-relative time r (r = 0 is `lead` seconds
before that question's clock starts, for both teams). Per question:

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
for >= min_gap; each cut keeps `padding` on both sides. Every clip and cut is written with its
reason so it can be reviewed before rendering.
"""
from __future__ import annotations

import json

import numpy as np

from . import media, screen, timeline
from .config import Session
from .sync import PLAYERS, cam_track, cam_wav

GRID = 10  # Hz


def speech_activity(words: list[dict], wav: np.ndarray, rate: int, db_above: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-1/GRID-second masks in cam time: (has_word, loud)."""
    n = int(len(wav) / rate * GRID) + 1
    has_word = np.zeros(n, bool)
    for w in words:
        has_word[max(0, int((w["start"] - 0.1) * GRID)):int((w["end"] + 0.1) * GRID) + 1] = True
    hop = rate // GRID
    m = len(wav) // hop
    rms = np.sqrt((wav[: m * hop].reshape(m, hop) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 10)
    loud = np.zeros(n, bool)
    loud[:m] = db > floor + db_above
    return has_word, loud


def _at(mask: np.ndarray, t: float) -> bool:
    i = int(t * GRID)
    return bool(mask[i]) if 0 <= i < len(mask) else False


class Signals:
    def __init__(self, sess: Session, cfg: dict):
        c = cfg["cut"]
        self.speech, self.motion = {}, {}
        win = int(0.3 * screen.PROXY_FPS)
        for p in PLAYERS:
            words = json.loads((sess.work / f"{p}_words.json").read_text(encoding="utf-8"))
            wav = media.audio_track(sess.files[f"{p}_cam"], cam_track(sess, cfg, p), cam_wav(sess, cfg, p))
            self.speech[p] = speech_activity(words, wav, 16000, c["speech_db_above_floor"])
            mo = np.load(sess.work / f"{p}_screen_signals.npz")["motion"]
            # a pixel change counts for +-0.3s so single quiet frames inside activity don't cut
            self.motion[p] = np.array([mo[max(0, i - win):i + win + 1].max() for i in range(len(mo))])

    def talking(self, p: str, cam_t: float) -> bool:
        w, loud = self.speech[p]
        return _at(w, cam_t) or _at(loud, cam_t)

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


def plan_clips(segs: list[timeline.Segment], sig: Signals, cfg: dict) -> list[dict]:
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
        if m and m["question"] == c["question"] and m["team"] == c["team"] and abs(m["r1"] - c["r0"]) < 1e-6:
            m["r1"] = c["r1"]
            m["why"] += " + " + c["why"]
        else:
            merged.append(dict(c))
    return [c for c in merged if c["r1"] - c["r0"] > 0.05]


def build(sess: Session, sync: dict, cfg: dict, log=print) -> dict:
    c = cfg["cut"]
    fps = cfg["output"]["fps"]
    segs = {s.n: s for s in timeline.build(sync, cfg)}
    sig = Signals(sess, cfg)
    clips = plan_clips(list(segs.values()), sig, cfg)
    keeps, cuts = [], []
    out_t = 0.0
    for cl in clips:
        seg, p = segs[cl["question"]], cl["team"]
        tr = seg.tracks[p]
        rr = (tr.reveal - tr.start) if tr.reveal is not None else None
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
            cut_ok[i] = silent and static and not protected and cl["question"] != 0
            state.append("held" if held else "static")
        pos = r0
        for s, e in screen.runs(cut_ok, GRID, min_len=c["min_gap"], max_gap=0.0):
            a = r0 + s + (0 if s == 0 else c["padding"])          # clip edges are cuts already
            b = r0 + e - (0 if r0 + e >= r1 - 0.05 else c["padding"])
            if b - a < 0.2:
                continue
            if a > pos:
                keeps.append({"question": cl["question"], "team": p, "r0": round(pos, 3), "r1": round(a, 3),
                              "out_start": round(out_t, 3)})
                out_t += a - pos
            cuts.append({"question": cl["question"], "team": p, "r0": round(a, 3), "r1": round(b, 3),
                         "reason": f"Q{cl['question'] or 'results'}: {p} silent {e - s:.1f}s, game {state[int(s * GRID)]}"})
            pos = b
        if r1 > pos:
            keeps.append({"question": cl["question"], "team": p, "r0": round(pos, 3), "r1": round(r1, 3),
                          "out_start": round(out_t, 3)})
            out_t += r1 - pos

    per_q: dict = {}
    for k in keeps:
        per_q[k["question"]] = per_q.get(k["question"], 0) + k["r1"] - k["r0"]
    return {
        "fps": fps,
        "total_seconds": round(out_t, 2),
        "clips_seconds": round(sum(cl["r1"] - cl["r0"] for cl in clips), 2),
        "per_question_seconds": {str(k): round(v, 2) for k, v in per_q.items()},
        "clips": clips,
        "keep": keeps,
        "cuts": cuts,
    }


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
