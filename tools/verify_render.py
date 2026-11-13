"""Check a rendered reel: for ~17 windows, find the source lag (s) that best explains the cam and game
frames shown. Every reading should be ~0.00s; a growing lag means the renderer drifted from the plan.

  .venv/Scripts/python tools/verify_render.py raw/dayN [out/dayN/reel.mp4]
"""
import json, subprocess, sys
from pathlib import Path
import cv2, numpy as np
from krillion_reel import config, render, ff
from krillion_reel.sync import PLAYERS
cfg = config.load_config(Path("config.yaml"))
sess = config.resolve_session(cfg, sys.argv[1] if len(sys.argv) > 1 else "raw/day2")
reel = sys.argv[2] if len(sys.argv) > 2 else str(sess.out / "reel.mp4")
sync = json.loads(sess.sync_json.read_text(encoding="utf-8"))
edl = json.loads(sess.edl_json.read_text(encoding="utf-8"))
plan = render.make_plan(sync, edl, cfg)
comp = render.Compositor(sess, sync, cfg)
FF = ff.tool("ffmpeg")
SW, WIN, SRCH = 64, 90, 4.0

def grab(src, vf, t0, dur, w, h, extra=""):
    sh = int(round(h * SW / w)) // 2 * 2
    p = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0,t0):.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                        "-an", "-vf", vf + f",scale={SW}:{sh},format=gray", "-f", "rawvideo", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, sh, SW).astype(np.float32), max(0, t0)

def z(a):
    a = a - a.mean(axis=(1, 2), keepdims=True)
    return a / (np.sqrt((a ** 2).sum(axis=(1, 2), keepdims=True)) + 1e-6)

# test windows: middle of pieces whose team stays on screen for >= WIN frames
runs, i = [], 0
while i < plan.n:
    j = i
    while j < plan.n and plan.shown[j] == plan.shown[i]:
        j += 1
    if j - i >= WIN + 10:
        runs.append((i, j))
    i = j
for a, b in runs[:: max(1, len(runs) // 14)]:
    m = (a + b) // 2 - WIN // 2
    p = PLAYERS[plan.shown[m]]
    ch = comp.cam_h[p]
    gh = comp.games[p].h
    reel_fr = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-ss", f"{m/30:.4f}", "-i", reel, "-frames:v", str(WIN),
                              "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    R = np.frombuffer(reel_fr, np.uint8).reshape(-1, 1920, 1080).astype(np.float32)
    res = []
    for kind, rd, tt, rows in (("cam", comp.cams[p].reader, plan.cam_t, (0, ch)), ("scr", comp.games[p], plan.screen_t, (ch, ch + gh))):
        ts = tt[m:m + WIN]
        src, t0 = grab(rd.src, rd.vf, ts.min() - SRCH - 0.2, ts.max() - ts.min() + 2 * SRCH + 0.4, rd.w, rd.h)
        hh = src.shape[1]
        if kind == "cam":  # cam tile is zoomed in around focus; compare a central crop on both
            src = src[:, hh // 6: hh - hh // 6, SW // 6: SW - SW // 6]
        band = np.stack([cv2.resize(f[rows[0]:rows[1]], (SW, hh), interpolation=cv2.INTER_AREA) for f in R])
        if kind == "cam":
            band = band[:, hh // 6: hh - hh // 6, SW // 6: SW - SW // 6]
        # use temporal differences so static content doesn't dominate
        dB = z(np.diff(band, axis=0))
        best = None
        for L in np.arange(-SRCH, SRCH + 1e-9, 1 / 30):
            idx = np.round((ts + L - t0) * 30).astype(int)
            if idx.min() < 0 or idx.max() + 1 >= len(src):
                continue
            dS = z(src[idx[1:]] - src[idx[:-1]])
            s = float((dS * dB).sum(axis=(1, 2)).mean())
            if best is None or s > best[1]:
                best = (L, s)
        motion = float(np.abs(np.diff(band, axis=0)).mean())
        res.append(f"{kind} lag {best[0]:+.2f}s (score {best[1]:.2f}, motion {motion:.1f})" if best else f"{kind} n/a")
    print(f"out {m/30:6.1f}s {p} cam_t {plan.cam_t[m]:7.2f} scr_t {plan.screen_t[m]:7.2f} | " + " | ".join(res), flush=True)
