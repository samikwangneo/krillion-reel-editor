"""krillion-reel CLI.

  krillion-reel run [raw/day1]      sync -> cut -> preview -> render, stopping at checkpoints
  krillion-reel sync [session]      Step 1 only (writes out/<session>/sync.json)
  krillion-reel cut [session]       Step 2 only (writes out/<session>/edl.json)
  krillion-reel review [session]    print sync + EDL summary
  krillion-reel stills [session]    a still of each team at its first reveal (check framing/zoom)
  krillion-reel preview [session]   10s low-res preview around the first reveal + a still frame
  krillion-reel render [session]    final out/<session>/reel.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import edl as edl_mod
from . import render as R
from . import sync as sync_mod
from .config import load_config, resolve_session

PLAYERS = ("p1", "p2")


def _fmt(t):
    return "  -  " if t is None else f"{t:6.1f}"


def print_sync(sync: dict, cfg: dict):
    print("\n== SYNC ==")
    for p in PLAYERS:
        o = sync["cam_offset"][p]
        print(f"  {p} cam_offset {o['seconds']:+.3f}s  method={o['method']}  confidence={o.get('confidence')}  "
              f"confirmed={o.get('confirmed')}")
    print(f"  {'Q':>2}  {'p1 clock':>8} {'reveal':>6} {'tier':>17}   {'p2 clock':>8} {'reveal':>6} {'tier':>17}  prompt")
    for q in sync["questions"]:
        a, b = q.get("p1") or {}, q.get("p2") or {}
        print(f"  {q['n']:>2}  {_fmt(a.get('clock_start')):>8} {_fmt(a.get('reveal'))} {str(a.get('tier')):>17}   "
              f"{_fmt(b.get('clock_start')):>8} {_fmt(b.get('reveal'))} {str(b.get('tier')):>17}  "
              f"{(q.get('prompt') or '')[:48]}")
    print(f"  results screen: p1 {sync['results'].get('p1')}  p2 {sync['results'].get('p2')}")


def print_edl(edl: dict):
    print("\n== CLIPS ==  (r = seconds into the question, 0 = prompt card appears)")
    for c in edl["clips"]:
        print(f"  Q{c['question'] or 'end'}  {c['team']}  r {c['r0']:6.2f}-{c['r1']:6.2f} ({c['r1'] - c['r0']:4.1f}s)  {c['why']}")
    print("\n== CUTS ==")
    for c in edl["cuts"]:
        print(f"  {c['team']}  r {c['r0']:6.2f}-{c['r1']:6.2f} ({c['r1'] - c['r0']:4.1f}s)  {c['reason']}")
    print(f"\n  per question (s): {edl['per_question_seconds']}")
    print(f"  total {edl['clips_seconds']:.1f}s -> {edl['total_seconds']:.1f}s after {len(edl['cuts'])} cuts")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="krillion-reel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "sync", "cut", "review", "stills", "preview", "render"])
    ap.add_argument("session", nargs="?", help="session folder, e.g. raw/day1 (default: newest folder in raw/)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--redetect", action="store_true", help="ignore sync.json and re-run detection")
    ap.add_argument("--force", action="store_true", help="render even if checks fail (unconfirmed sync, >max length)")
    a = ap.parse_args(argv)

    cfg = load_config(Path(a.config))
    sess = resolve_session(cfg, a.session)
    print(f"[krillion-reel] session '{sess.name}': {', '.join(f'{k}={v.name}' for k, v in sess.files.items())}")

    sync = sync_mod.load_or_detect(sess, cfg, redetect=a.redetect)
    issues = sync_mod.problems(sync, cfg)
    if a.command == "sync":
        print_sync(sync, cfg)
        _report(issues)
        return 0

    edl = edl_mod.load_or_build(sess, sync, cfg, rebuild=True)
    if a.command in ("cut", "review"):
        print_sync(sync, cfg) if a.command == "review" else None
        print_edl(edl)
        _report(issues)
        _length_check(sync, edl, cfg)
        return 0

    if a.command == "stills":
        rf = R.reveal_frames(sync, edl, cfg)
        for p in PLAYERS:
            if rf[p]:
                out = sess.out / f"still_{p}.png"
                R.still(sess, sync, edl, cfg, rf[p][0] + int(0.45 * cfg["output"]["fps"]), out)
                print(f"[stills] {out}  ({p}'s first reveal, at peak zoom)")
        return 0

    if a.command in ("run", "preview"):
        fps = cfg["output"]["fps"]
        rf = R.reveal_frames(sync, edl, cfg)
        first = min(f for p in PLAYERS for f in rf[p][:1]) if any(rf.values()) else 0
        f0 = max(0, first - 5 * fps)
        out = sess.out / "preview.mp4"
        R.render(sess, sync, edl, cfg, out, frames=(f0, f0 + 10 * fps), scale=0.5)
        png = sess.out / "preview.png"
        R.still(sess, sync, edl, cfg, first + int(0.45 * fps), png)
        print(f"[preview] {out}")
        print(f"[preview] {png}  (first reveal, at peak zoom)")
        if a.command == "preview":
            return 0

    # final render gates
    blockers = list(issues)
    if not _length_check(sync, edl, cfg):
        blockers.append("reel too long")
    if blockers and not a.force:
        _report(issues)
        print("\nNot rendering the final reel until the above is resolved (or pass --force).")
        return 2
    out = sess.out / "reel.mp4"
    R.render(sess, sync, edl, cfg, out)
    print(f"[render] {out}  ({edl['total_seconds']:.1f}s)")
    return 0


def _report(issues: list[str]):
    if issues:
        print("\n== NEEDS YOUR INPUT ==")
        for i in issues:
            print(f"  - {i}")


def _length_check(sync: dict, edl: dict, cfg: dict) -> bool:
    mx = cfg["output"]["max_seconds"]
    if edl["total_seconds"] <= mx:
        return True
    print(f"\n== TOO LONG: {edl['total_seconds']:.1f}s > {mx}s ==")
    print("  Suggested questions to drop (least exciting reveals first). Add them to drop_questions in config.yaml:")
    for r in edl_mod.suggest_drops(sync, edl, cfg):
        print(f"   Q{r['q']}: {r['seconds']:5.1f}s  tiers {r['tiers']}  answers {r['answers']}  - {(r['prompt'] or '')[:50]}")
    return False


if __name__ == "__main__":
    sys.exit(main())
