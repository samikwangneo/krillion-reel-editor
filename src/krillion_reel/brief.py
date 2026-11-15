"""`krillion-reel brief`: everything Claude needs to pick the clips for a session, as one Markdown file.

Workflow (layout.answering: model):
  1. krillion-reel brief raw/dayN   -> out/dayN/brief.md (+ brief_frames/qN.jpg)
  2. ask Claude (in Claude Code) to "edit dayN": it reads the brief and writes out/dayN/picks.json
  3. krillion-reel render raw/dayN  -> the picks are checked against the rules (edl.apply_picks)

All times in the brief and in picks.json are segment time r: seconds into the question, the same
"r" that edl.json and `krillion-reel cut` print.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import edl as E
from . import ff, timeline
from .config import Session
from .screen import OCR_FPS
from .sync import PLAYERS

INSTRUCTIONS = """\
# Edit brief: {name}

You are choosing what to keep from each question of a Krillion 1v1 for an Instagram Reel. Write your
picks to `out/{name}/picks.json`, then run `krillion-reel render raw/{name}` (or `cut` to review).

## What to pick
For each question, list the clips for the **opening and the middle**, in play order. Each clip is
one team's continuous footage from `r0` to `r1`, in segment time (seconds into the question).
* Open on a team **reading the prompt** (from r = 0 until they finish reading).
* Then **~5s of each team thinking**, right after each one finishes reading.
* From each team's free middle, keep only what is worth watching. That means jokes, roasts, trash
  talk, near-misses, a good argument about the answer, or a funny reaction. Cut repetition, mumbling
  and dead air. A few seconds per team is typical. Skip a team's middle entirely if nothing in it lands.
* Start and end clips on line boundaries from the transcript, so no clip cuts someone off mid-word.
* Interleave the teams to keep it moving. A clip can't replay a moment already shown.

**The end of each question is automatic.** Both teams' last {end:.0f}s before submitting, then
each team's uncut shot from {nocut:.0f}s before submit through the card and their reaction. Don't
pick anything at or after a team's `end phase` time; it gets clipped off.

**The rules are enforced afterwards.** If the picks leave out the reading or the first seconds of
thinking, those clips get added back, marked "rule: ..." in edl.json.

**Speed.** Everything plays at {fast}x except deep cut / one in a krillion reveals. Add
`"speed": 1.0` to a pick to keep a moment at normal speed, for example a punchline that needs it.

**Length: {target}.** The reel must land on {target} (Instagram only pushes reels up to 2:59).
Set `"drop_questions": [4]` in picks.json to drop a question for this session when even the
must-haves don't fit. Your picks without a `"speed"` and the answer sinking (submit -> card) are *filler*: they get
sped up between {fast}x and {filler_max}x to land exactly on the target. `krillion-reel cut` prints
the fit: "too long" means drop picks, "short" means add picks (the funniest moments first).

**Intro.** The reel opens on {intro}'s intro (their cam before the first prompt). Pick the part
worth keeping (~10-15s) as `"intro": {{"clips": [{{"r0": .., "r1": .., "why": ..}}]}}`, r = their
cam time. Without it the whole intro plays.

**3+ players.** sync.json `"groups"` lists sides, e.g. `[["p1", "p2"]]` (roommates) vs p3. Each
question shows one member of each group (the featured one) plus everyone ungrouped, so it plays
like a 1v1. The others get a ~2s look at their card, placed among the reveals in tier order. The
featured member is the better tier (ties: louder reaction); set `"featured": "p2"` on a question to
override. Only pick clips of the two teams shown.

## picks.json
```json
{{
  "questions": {{
    "1": {{"clips": [
      {{"team": "p1", "r0": 0.0, "r1": 9.6, "why": "reads the prompt, starts guessing"}},
      {{"team": "p2", "r0": 4.2, "r1": 9.2, "why": "first thoughts"}},
      {{"team": "p2", "r0": 11.0, "r1": 14.5, "why": "roasts their teammate's answer", "speed": 1.0}}
    ]}}
  }}
}}
```
Any question left out of picks.json uses the automatic highlights cut.

"""


def _lines(words: list[dict], base: float, r0: float, r1: float) -> list[tuple[float, float, str]]:
    """Group words into lines (break on a pause, a sentence end, or a long line) in segment time."""
    ws = [(w["start"] - base, min(w["end"], w["start"] + 1.0) - base, w["word"].strip()) for w in words]
    ws = [w for w in ws if r0 <= w[0] < r1 and w[2]]
    out, cur = [], []
    for w in ws:
        if cur and (w[0] - cur[-1][1] > 0.7 or len(cur) >= 14
                    or (cur[-1][2][-1:] in ".?!" and len(cur) >= 3)):
            out.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur)))
            cur = []
        cur.append(w)
    if cur:
        out.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur)))
    return out


def _level(sig: E.Signals, p: str, tr: timeline.Track, r0: float, r1: float) -> float:
    db = sig.db[p]
    idx = [int(tr.cam_t(r0 + k / E.GRID) * E.GRID) for k in range(max(1, int((r1 - r0) * E.GRID)))]
    v = [db[i] for i in idx if 0 <= i < len(db)]
    return float(np.mean(v)) if v else float("-inf")


def _peaks(sig: E.Signals, p: str, tr: timeline.Track, r0: float, r1: float, k: int = 4) -> list[tuple[float, float]]:
    """Up to k loudest moments (r, dB above the team's median), at least 2s apart."""
    db = sig.db[p]
    med = float(np.median(db))
    pts = []
    for i in range(int(r0 * E.GRID), int(r1 * E.GRID)):
        j = int(tr.cam_t(i / E.GRID) * E.GRID)
        if 2 <= j < len(db) - 2:
            pts.append((float(np.mean(db[j - 2:j + 3])) - med, i / E.GRID))
    out: list[tuple[float, float]] = []
    for lv, r in sorted(pts, reverse=True):
        if all(abs(r - x) >= 2.0 for x, _ in out):
            out.append((r, lv))
        if len(out) == k:
            break
    return sorted(out)


def _cam_frame(src: Path, t: float, w: int = 320) -> np.ndarray:
    p = subprocess.run([ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}",
                        "-i", str(src), "-frames:v", "1", "-vf", f"scale={w}:-2", "-f", "image2pipe",
                        "-vcodec", "png", "-"], capture_output=True)
    img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR) if p.stdout else None
    return img if img is not None else np.zeros((180, w, 3), np.uint8)


def _screen_frame(sess: Session, p: str, t: float, w: int = 320) -> np.ndarray:
    f = sess.work / f"{p}_screen_ocr" / f"{max(0, int(round(t * OCR_FPS))):05d}.jpg"
    img = cv2.imread(str(f)) if f.exists() else None
    if img is None:
        return np.zeros((180, w, 3), np.uint8)
    return cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)


def _tile(rows: list[list[tuple[np.ndarray, str]]]) -> np.ndarray:
    h = max(im.shape[0] for row in rows for im, _ in row)
    out = []
    for row in rows:
        ims = []
        for im, label in row:
            im = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 4, cv2.BORDER_CONSTANT)
            cv2.putText(im, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            ims.append(im)
        out.append(np.hstack(ims))
    w = max(r.shape[1] for r in out)
    return np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in out])


def write_brief(sess: Session, sync: dict, cfg: dict, frames: bool = True, log=print) -> Path:
    cfg = {**cfg, "groups": E.groups(sync, cfg)}
    all_segs = timeline.build(sync, cfg)
    segs = [s for s in all_segs if s.n > 0]
    sig = E.Signals(sess, cfg)
    prompts = {q["n"]: q.get("prompt") for q in sync["questions"]}
    qinfo = {q["n"]: q for q in sync["questions"]}
    hl = cfg["layout"]["highlights"]
    fdir = sess.out / "brief_frames"
    if frames:
        fdir.mkdir(parents=True, exist_ok=True)
    tgt = cfg["output"].get("target_seconds")
    md = [INSTRUCTIONS.format(name=sess.name, end=hl["end_before_submit"], nocut=hl["no_cut_before_submit"],
                              fast=cfg["speed"]["boring"], filler_max=cfg["speed"].get("filler_max"),
                              target=f"{int(tgt) // 60}:{int(tgt) % 60:02d}" if tgt else "no limit",
                              intro=cfg["timeline"].get("intro"))]
    intro = next((s for s in all_segs if s.n == -1), None)
    if intro:
        p = next(iter(intro.tracks))
        tr = intro.tracks[p]
        loud = float(np.percentile(sig.db[p], 85))
        md.append(f"## Intro ({p}, r = cam time, {intro.length:.1f}s until the first prompt)\n")
        md.append("```")
        for r0, r1, text in _lines(sig.words[p], tr.start + tr.offset, 0.0, intro.length):
            mark = "!" if _level(sig, p, tr, r0, max(r1, r0 + 0.2)) > loud else " "
            md.append(f"{mark} {r0:6.1f}-{r1:6.1f}  {text}")
        md.append("```\n")
    default: dict = {}
    for c in E.plan_clips(segs, sig, {**cfg, "layout": {**cfg["layout"], "answering": "highlights"}}, prompts):
        default.setdefault(c["question"], []).append(c)
    start = 0
    for seg in segs:
        rev = {}
        for p in PLAYERS:
            tr = seg.tracks[p]
            rev[p] = (tr.reveal - tr.start) if tr.reveal is not None else tr.length - cfg["layout"]["reveal_after"]
        shown, flashed, fwhy = E.split_teams(seg, sig, cfg)
        a = E.anchors(seg, sig, cfg, rev, start, prompts.get(seg.n))
        start = 1 - start
        q = qinfo[seg.n]
        md.append(f"## Q{seg.n}: {prompts.get(seg.n) or ''}\n")
        if flashed:
            md.append(f"* **Shown: {' vs '.join(shown)}** (" + "; ".join(f"{p}: {w}" for p, w in fwhy.items())
                      + f"). Quick look only: {', '.join(flashed)}. Override with \"featured\".")
        for p in PLAYERS:
            tr, e = seg.tracks[p], q[p]
            read, aloud = a["read_end"][p]
            md.append(f"* **{p}**: {e.get('tier')} \"{e.get('answer') or '-'}\". Reads until r {read:.1f} "
                      f"({'aloud' if aloud else 'not read aloud'}). Submits at r {a['submit'][p]:.1f}, card at "
                      f"r {rev[p]:.1f}. **End phase from r {a['end_start'][p]:.1f}** (automatic). "
                      f"Free middle: r {min(read + hl['think_after_read'], a['end_start'][p]):.1f} to "
                      f"{a['end_start'][p]:.1f}.")
        md.append(f"* Reading aloud first: **{a['first']}** (with only the shown teams, the reading team is the "
                  f"first of them that reads aloud). Reveal order (lower tier first): {', '.join(a['order'])}.")
        md.append("* Automatic highlights cut, for comparison: " + "; ".join(
            f"{c['team']} {c['r0']:.1f}-{c['r1']:.1f}" for c in default.get(seg.n, [])))
        md.append("")
        rows = []
        for p in PLAYERS:
            tr = seg.tracks[p]
            base = tr.start + tr.offset
            med = float(np.median(sig.db[p]))
            loud = float(np.percentile(sig.db[p], 85))
            md.append(f"**{p} transcript** (r, `!` = loud):")
            md.append("```")
            for r0, r1, text in _lines(sig.words[p], base, 0.0, a["reveal_end"][p] + 0.5):
                mark = "!" if _level(sig, p, tr, r0, max(r1, r0 + 0.2)) > loud else " "
                md.append(f"{mark} {r0:6.1f}-{r1:6.1f}  {text}")
            md.append("```")
            pk = _peaks(sig, p, tr, 0.0, a["end_start"][p])
            md.append(f"{p} loudest before the end phase: " +
                      (", ".join(f"r {r:.1f} (+{lv:.0f} dB)" for r, lv in pk) or "none") + f" (median {med:.0f} dB)\n")
            if frames:
                times = [r for r, _ in pk[:3]] or [a["read_end"][p][0] + 2.0]
                row = [(_cam_frame(sess.files[f"{p}_cam"], tr.cam_t(r)), f"{p} r{r:.1f}") for r in times]
                mid = times[0]
                row.append((_screen_frame(sess, p, tr.screen_t(mid, cfg["output"]["fps"])), f"{p} game r{mid:.1f}"))
                rows.append(row)
        if frames and rows:
            f = fdir / f"q{seg.n}.jpg"
            cv2.imwrite(str(f), _tile(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
            md.append(f"Frames at the loudest moments: `{f.as_posix()}`\n")
        log(f"[brief] Q{seg.n} done")
    out = sess.out / "brief.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out


def picks_path(sess: Session) -> Path:
    return sess.out / "picks.json"


def picks_summary(sess: Session) -> str:
    f = picks_path(sess)
    if not f.exists():
        return "no picks.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    qs = d.get("questions") or {}
    return f"picks.json: {len(qs)} questions, {sum(len(v.get('clips', [])) for v in qs.values())} clips"
