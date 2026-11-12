"""Steps 3 + 5 - layout, compositing, audio mix and encode.

Layout: ONE team on screen at a time - their cam on top, their game below - cutting fully between
teams according to edl.json "shots". Each team's cam band is sized so their game fills the full
1080 width with no bars.

Video: one FFmpeg reader per source (seek + fps + crop + scale -> raw frames), composited in
Python (zoom via warpAffine, labels), piped into an FFmpeg libx264 encoder.
Audio: cam mics (+ game sound where recorded) sliced to the same keep pieces in numpy, following the
on-screen team (or mixing both), then two-pass loudnorm to the target LUFS.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import faces, ff, media, timeline, zoom
from .config import Session
from .sync import PLAYERS, cam_track

HEADROOM = 1.35          # cam content decoded this much larger than the tile, so zooms stay sharp
AUDIO_RATE = 48000
FADE = 0.01              # seconds of fade at every audio cut
COLORS = {"p1": (232, 209, 79), "p2": (123, 79, 255)}  # BGR: Krillion cyan / pink
DIVIDER = (26, 15, 10)


# ---------------------------------------------------------------- plan: output frame -> sources

@dataclass
class Piece:
    out_frame: int
    frames: int
    team: str
    track: timeline.Track
    r0: float  # segment-relative start


@dataclass
class Plan:
    fps: int
    n: int
    shown: np.ndarray                   # team index on screen per output frame (0=p1, 1=p2)
    screen_t: np.ndarray                # source time of the shown team's screen per output frame
    cam_t: np.ndarray                   # source time of the shown team's cam per output frame
    zoom: dict[str, np.ndarray]
    reveals_out: dict[str, list] = field(default_factory=dict)
    pieces: list[Piece] = field(default_factory=list)


def make_plan(sync: dict, edl: dict, cfg: dict) -> Plan:
    """edl.json "keep" pieces (one team each, in order) -> per-output-frame source times."""
    fps = int(cfg["output"]["fps"])
    segs = {s.n: s for s in timeline.build(sync, cfg)}
    durs = {k: float(v["duration"]) for k, v in sync["sources"].items()}
    pieces: list[Piece] = []
    f = 0
    for k in edl["keep"]:
        nk = int(round((k["r1"] - k["r0"]) * fps))
        if nk > 0:
            pieces.append(Piece(f, nk, k["team"], segs[k["question"]].tracks[k["team"]], k["r0"]))
            f += nk
    n = f
    shown = np.zeros(n, int)
    screen_t, cam_t = np.zeros(n), np.zeros(n)
    reveals_out: dict[str, list] = {p: [] for p in PLAYERS}
    for pc in pieces:
        p, tr = pc.team, pc.track
        r = pc.r0 + np.arange(pc.frames) / fps
        sl = slice(pc.out_frame, pc.out_frame + pc.frames)
        shown[sl] = PLAYERS.index(p)
        screen_t[sl] = np.clip([tr.screen_t(x, fps) for x in r], 0, durs[f"{p}_screen"] - 1.0 / fps)
        cam_t[sl] = np.clip(tr.start + r + tr.offset, 0, durs[f"{p}_cam"] - 1.0 / fps)
        if tr.reveal is not None:
            rr = tr.reveal - tr.start
            if pc.r0 <= rr < pc.r0 + pc.frames / fps:
                reveals_out[p].append((pc.out_frame / fps + rr - pc.r0, tr.tier))
    zooms = {p: zoom.curve(n, fps, reveals_out[p], cfg) for p in PLAYERS}
    return Plan(fps, n, shown, screen_t, cam_t, zooms, reveals_out, pieces)


# ---------------------------------------------------------------- frame readers

class Reader:
    """Sequential frame reader that re-seeks only when the requested time jumps."""

    def __init__(self, src: Path, vf: str, w: int, h: int, fps: int):
        self.src, self.vf, self.w, self.h, self.fps = src, vf, w, h, fps
        self.proc = None
        self.next_t = None
        self.last = None
        self.last_t = None

    def _start(self, t: float):
        self.close()
        args = [ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", str(self.src),
                "-an", "-sn", "-vf", self.vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     bufsize=self.w * self.h * 3 * 2)
        self.next_t = t

    def get(self, t: float) -> np.ndarray:
        if self.last is not None and abs(t - self.last_t) < 0.25 / self.fps:
            return self.last  # hold
        if self.proc is None or abs(t - self.next_t) > 0.5 / self.fps:
            self._start(t)
        buf = self.proc.stdout.read(self.w * self.h * 3)
        if len(buf) < self.w * self.h * 3:
            frame = self.last if self.last is not None else np.zeros((self.h, self.w, 3), np.uint8)
        else:
            frame = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.next_t = t + 1.0 / self.fps
        self.last, self.last_t = frame, t
        return frame

    def close(self):
        if self.proc is not None:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait()
            self.proc = None


# ---------------------------------------------------------------- tiles

@dataclass
class CamTile:
    reader: Reader
    cw: int
    ch: int
    s1: float          # content px -> tile px at zoom 1 ("contain")
    focus: tuple[float, float]
    tw: int
    th: int

    def render(self, frame: np.ndarray, z: float) -> np.ndarray:
        s = self.s1 * z
        fx, fy = self.focus
        Fx = fx * self.s1 + (self.tw - self.cw * self.s1) / 2
        Fy = fy * self.s1 + (self.th - self.ch * self.s1) / 2
        tx, ty = Fx - s * fx, Fy - s * fy
        sw, sh = self.cw * s, self.ch * s
        tx = min(max(tx, self.tw - sw), 0) if sw >= self.tw else (self.tw - sw) / 2
        ty = min(max(ty, self.th - sh), 0) if sh >= self.th else (self.th - sh) / 2
        if sw < self.tw - 0.5 or sh < self.th - 0.5:
            out = _blur_cover(frame, self.tw, self.th)
        else:
            out = np.zeros((self.th, self.tw, 3), np.uint8)
        M = np.float32([[s, 0, tx], [0, s, ty]])
        cv2.warpAffine(frame, M, (self.tw, self.th), dst=out, flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_TRANSPARENT)
        return out


def _blur_cover(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = frame.shape[:2]
    sc = max(tw / w, th / h)
    small = cv2.resize(frame, (max(8, int(w * sc / 10)), max(8, int(h * sc / 10))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    big = cv2.resize(small, (int(w * sc) + 2, int(h * sc) + 2), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (big.shape[0] - th) // 2, (big.shape[1] - tw) // 2
    return (big[y0:y0 + th, x0:x0 + tw] * 0.55).astype(np.uint8)


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def _label(text: str, color_bgr, font_path: str, size: int) -> np.ndarray:
    """Pill label as a BGRA numpy array."""
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        font = ImageFont.load_default()
    pad_x, pad_y = int(size * 0.6), int(size * 0.3)
    l, t, r, b = font.getbbox(text)
    w, h = r - l + 2 * pad_x, b - t + 2 * pad_y
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=(8, 14, 26, 210), outline=rgb + (255,), width=3)
    d.text((pad_x - l, pad_y - t), text, font=font, fill=rgb + (255,))
    return np.array(img)[..., [2, 1, 0, 3]]


def _blit(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, alpha: float = 1.0):
    h, w = rgba.shape[:2]
    roi = canvas[y:y + h, x:x + w]
    a = (rgba[..., 3:4].astype(np.float32) / 255.0) * alpha
    roi[:] = (roi * (1 - a) + rgba[..., :3] * a).astype(np.uint8)


class Compositor:
    def __init__(self, sess: Session, sync: dict, cfg: dict):
        o, lay = cfg["output"], cfg["layout"]
        self.W, self.H, self.fps = int(o["width"]), int(o["height"]), int(o["fps"])
        self.cams: dict[str, CamTile] = {}
        self.games: dict[str, Reader] = {}
        self.cam_h: dict[str, int] = {}
        for p in PLAYERS:
            # game fills the width; cam gets the rest (never less than min_cam_height)
            gx, gy, gw_, gh_ = sync["crop"][p]["rect"]
            gh = _even(self.W * gh_ / gw_)
            gw = self.W
            if gh > self.H - lay["min_cam_height"]:
                gh = _even(self.H - lay["min_cam_height"])
                gw = _even(gh * gw_ / gh_)
            cam_h = self.H - gh
            self.cam_h[p] = cam_h
            vf = f"fps={self.fps},crop={gw_}:{gh_}:{gx}:{gy},scale={gw}:{gh}:flags=lanczos"
            self.games[p] = Reader(sess.files[f"{p}_screen"], vf, gw, gh, self.fps)
            # cam: crop around everyone at the band's aspect
            v = sync["cam_view"][p]
            W0, H0 = v["frame"]
            rx, ry, rw, rh = faces.fit_aspect(v["box"], W0, H0, self.W / cam_h)
            c = min(self.W / rw, cam_h / rh)
            cw, ch = _even(min(rw, rw * c * HEADROOM)), _even(min(rh, rh * c * HEADROOM))
            vf = f"fps={self.fps},crop={rw}:{rh}:{rx}:{ry},scale={cw}:{ch}:flags=bicubic"
            focus = ((v["focus"][0] - rx) * cw / rw, (v["focus"][1] - ry) * ch / rh)
            self.cams[p] = CamTile(Reader(sess.files[f"{p}_cam"], vf, cw, ch, self.fps), cw, ch, rw * c / cw, focus,
                                   self.W, cam_h)
        names = cfg["players"]
        self.labels = {p: _label(str(names[p]), COLORS[p], lay["label_font"], 40) for p in PLAYERS}

    def frame(self, plan: Plan, i: int) -> np.ndarray:
        p = PLAYERS[int(plan.shown[i])]
        cv = np.empty((self.H, self.W, 3), np.uint8)
        cv[:] = DIVIDER
        ch = self.cam_h[p]
        raw = self.cams[p].reader.get(float(plan.cam_t[i]))
        cv[0:ch] = self.cams[p].render(raw, float(plan.zoom[p][i]))
        g = self.games[p].get(float(plan.screen_t[i]))
        gh, gw = g.shape[:2]
        x = (self.W - gw) // 2
        cv[ch:ch + gh, x:x + gw] = g
        cv[ch - 2:ch + 2] = DIVIDER
        _blit(cv, self.labels[p], 24, 24)
        return cv

    def close(self):
        for r in [c.reader for c in self.cams.values()] + list(self.games.values()):
            r.close()


# ---------------------------------------------------------------- audio

def build_audio(sess: Session, sync: dict, cfg: dict, plan: Plan, out_wav: Path,
                frames: tuple[int, int] | None = None, log=print) -> Path:
    """Each piece plays its own team's mic (+ their game sound if recorded), 10ms fades at every edge."""
    rate = AUDIO_RATE
    n_total = int(round(plan.n / plan.fps * rate))
    mix = np.zeros(n_total, np.float32)
    g_gain = 10 ** (cfg["audio"]["game_audio_db"] / 20)
    cams = {p: media.audio_track(sess.files[f"{p}_cam"], cam_track(sess, cfg, p),
                                 sess.work / f"{p}_cam_a{cam_track(sess, cfg, p)}_48k.wav", rate) for p in PLAYERS}
    scr = {p: media.audio_track(sess.files[f"{p}_screen"], 0, sess.work / f"{p}_screen_a0_48k.wav", rate)
           for p in PLAYERS if sync["sources"][f"{p}_screen"]["audio_tracks"] > 0}
    fade_n = int(FADE * rate)
    ramp = np.linspace(0, 1, fade_n, dtype=np.float32)

    def take(x: np.ndarray, start_s: float, n: int, valid_n: int | None = None) -> np.ndarray:
        out = np.zeros(n, np.float32)
        s0 = int(round(start_s * rate))
        a, b = max(0, s0), min(len(x), s0 + n)
        if b > a:
            out[a - s0:b - s0] = x[a:b]
        if valid_n is not None:
            out[max(0, valid_n):] = 0
        if n > 2 * fade_n:
            out[:fade_n] *= ramp
            out[n - fade_n:] *= ramp[::-1]
        return out

    for pc in plan.pieces:
        o0 = int(round(pc.out_frame / plan.fps * rate))
        n = min(int(round(pc.frames / plan.fps * rate)), n_total - o0)
        p, tr = pc.team, pc.track
        y = take(cams[p], tr.cam_t(pc.r0), n)
        if p in scr:
            valid = int(round(max(0.0, tr.length - pc.r0) * rate))
            y += g_gain * take(scr[p], tr.start + pc.r0, n, valid)
        mix[o0:o0 + n] = y
    st = np.stack([mix, mix], axis=1)
    if frames is not None:
        a, b = (int(round(f / plan.fps * rate)) for f in frames)
        st = st[a:b]
    raw = out_wav.with_name(out_wav.stem + "_raw.wav")
    media.write_wav(raw, st * 0.5, rate)
    _loudnorm(raw, out_wav, cfg["audio"]["lufs"], log)
    return out_wav


def _loudnorm(src: Path, dst: Path, lufs: float, log=print):
    base = f"loudnorm=I={lufs}:TP=-1.0:LRA=11"
    p = subprocess.run([ff.tool("ffmpeg"), "-hide_banner", "-i", str(src), "-af", base + ":print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr)
    if not m:
        raise RuntimeError("loudnorm pass 1 failed:\n" + p.stderr[-2000:])
    j = json.loads(m.group(0))
    second = (f"{base}:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}"
              f":measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    ff.run(["-i", str(src), "-af", second, "-ar", str(AUDIO_RATE), "-c:a", "pcm_s16le", str(dst)])
    log(f"  loudness: measured {j['input_i']} LUFS -> target {lufs} LUFS")


# ---------------------------------------------------------------- encode

def render(sess: Session, sync: dict, edl: dict, cfg: dict, out: Path,
           frames: tuple[int, int] | None = None, scale: float = 1.0, log=print) -> Path:
    plan = make_plan(sync, edl, cfg)
    f0, f1 = frames if frames else (0, plan.n)
    f0, f1 = max(0, f0), min(f1, plan.n)
    wav = build_audio(sess, sync, cfg, plan, sess.work / f"{out.stem}_audio.wav", (f0, f1), log=log)
    comp = Compositor(sess, sync, cfg)
    W, H = _even(comp.W * scale), _even(comp.H * scale)
    oc = cfg["output"]
    args = [ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(comp.fps), "-i", "-",
            "-i", str(wav), "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", oc["preset"], "-crf", str(oc["crf"]), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    enc = subprocess.Popen(args, stdin=subprocess.PIPE)
    try:
        for i in range(f0, f1):
            fr = comp.frame(plan, i)
            if scale != 1.0:
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            enc.stdin.write(fr.tobytes())
            if (i - f0) % (comp.fps * 10) == 0:
                log(f"  frame {i - f0}/{f1 - f0} ({(i - f0) / comp.fps:.0f}s)")
    finally:
        enc.stdin.close()
        enc.wait()
        comp.close()
    if enc.returncode != 0:
        raise RuntimeError("encoder failed")
    return out


def still(sess: Session, sync: dict, edl: dict, cfg: dict, frame: int, out_png: Path) -> Path:
    plan = make_plan(sync, edl, cfg)
    comp = Compositor(sess, sync, cfg)
    try:
        img = comp.frame(plan, min(max(frame, 0), plan.n - 1))
    finally:
        comp.close()
    cv2.imwrite(str(out_png), img)
    return out_png


def reveal_frames(sync: dict, edl: dict, cfg: dict) -> dict[str, list[int]]:
    plan = make_plan(sync, edl, cfg)
    return {p: [int(t * plan.fps) for t, _ in plan.reveals_out[p]] for p in PLAYERS}
