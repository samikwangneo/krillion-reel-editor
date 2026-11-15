"""Steps 3 + 5 - layout, compositing, audio mix and encode.

Layout: ONE team on screen at a time - their cam on top, their game below - cutting fully between
teams according to edl.json "shots". Each team's cam band is sized so their game fills the full
1080 width with no bars.

Video: one FFmpeg reader per source (seek + fps + crop + scale -> raw frames), composited in
Python (zoom via warpAffine), piped into an FFmpeg libx264 encoder.
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

from . import faces, ff, media, timeline, zoom
from .config import Session
from .sync import PLAYERS, cam_track

HEADROOM = 1.35          # cam content decoded this much larger than the tile, so zooms stay sharp
AUDIO_RATE = 48000
FADE = 0.01              # seconds of fade at every audio cut
DIVIDER = (26, 15, 10)


# ---------------------------------------------------------------- plan: output frame -> sources

@dataclass
class Piece:
    out_frame: int
    frames: int
    team: str
    track: timeline.Track
    r0: float  # segment-relative start
    speed: float = 1.0  # source seconds per output second


@dataclass
class Plan:
    fps: int
    n: int
    shown: np.ndarray                   # team index on screen per output frame (index into PLAYERS)
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
        speed = float(k.get("speed", 1.0))
        nk = int(round((k["r1"] - k["r0"]) / speed * fps))
        if nk > 0:
            pieces.append(Piece(f, nk, k["team"], segs[k["question"]].tracks[k["team"]], k["r0"], speed))
            f += nk
    n = f
    shown = np.zeros(n, int)
    screen_t, cam_t = np.zeros(n), np.zeros(n)
    reveals_out: dict[str, list] = {p: [] for p in PLAYERS}
    for pc in pieces:
        p, tr = pc.team, pc.track
        r = pc.r0 + np.arange(pc.frames) * pc.speed / fps
        sl = slice(pc.out_frame, pc.out_frame + pc.frames)
        shown[sl] = PLAYERS.index(p)
        screen_t[sl] = np.clip([tr.screen_t(x, fps) for x in r], 0, durs[f"{p}_screen"] - 1.0 / fps)
        cam_t[sl] = np.clip(tr.start + r + tr.offset, 0, durs[f"{p}_cam"] - 1.0 / fps)
        if tr.reveal is not None:
            rr = tr.reveal - tr.start
            if pc.r0 <= rr < pc.r0 + pc.frames * pc.speed / fps:
                reveals_out[p].append((pc.out_frame / fps + (rr - pc.r0) / pc.speed, tr.tier))
    zooms = {p: zoom.curve(n, fps, reveals_out[p], cfg) for p in PLAYERS}
    return Plan(fps, n, shown, screen_t, cam_t, zooms, reveals_out, pieces)


# ---------------------------------------------------------------- frame readers

class Reader:
    """Sequential frame reader: holds, reads forward (dropping frames for sped-up parts), and re-seeks
    only when the requested time jumps backwards or far ahead."""

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
        """Frame at source time t. next_t is the time of the next frame waiting in the pipe and only
        ever advances by one frame per frame actually read - it must never be reset to t, or sped-up
        parts (t stepping 1.25 frames per call) would silently play at 1x and drift behind the audio."""
        if self.last is not None and abs(t - self.last_t) < 0.25 / self.fps:
            return self.last  # hold
        size = self.w * self.h * 3
        if self.proc is None or t < self.next_t - 0.5 / self.fps or t - self.next_t > 1.0:
            self._start(t)
        while t - self.next_t > 0.5 / self.fps:  # skip ahead without re-seeking
            if len(self.proc.stdout.read(size)) < size:
                break
            self.next_t += 1.0 / self.fps
        buf = self.proc.stdout.read(size)
        if len(buf) < size:
            frame = self.last if self.last is not None else np.zeros((self.h, self.w, 3), np.uint8)
        else:
            frame = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.next_t += 1.0 / self.fps
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
    """Dimmed, blurred version of frame covering tw x th - all the work happens at 1/10 size."""
    h, w = frame.shape[:2]
    S = max(tw / w, th / h)                       # "cover" scale of the full frame
    sw, sh = max(8, int(w * S / 10)), max(8, int(h * S / 10))
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.convertScaleAbs(cv2.GaussianBlur(small, (0, 0), 3), alpha=0.55)
    cw, ch = tw / S * sw / w, th / S * sh / h     # visible part, in small-image pixels
    x0, y0 = (sw - cw) / 2, (sh - ch) / 2
    M = np.float32([[tw / cw, 0, -x0 * tw / cw], [0, th / ch, -y0 * th / ch]])
    return cv2.warpAffine(small, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    """float32 alpha (h, w, 1): 1 inside a rounded rectangle, anti-aliased edge."""
    m = np.zeros((h * 4, w * 4), np.uint8)
    R4 = r * 4
    cv2.rectangle(m, (R4, 0), (w * 4 - R4, h * 4), 255, -1)
    cv2.rectangle(m, (0, R4), (w * 4, h * 4 - R4), 255, -1)
    for cx, cy in ((R4, R4), (w * 4 - R4, R4), (R4, h * 4 - R4), (w * 4 - R4, h * 4 - R4)):
        cv2.circle(m, (cx, cy), R4, 255, -1)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return (m.astype(np.float32) / 255.0)[:, :, None]


@dataclass
class Bubble:
    """A teammate on a video call, cut out of that player's screen recording (sync.json "facetime":
    {"p2": {"rect": [x, y, w, h], "corner": "top-left", "width": 0.34}}), drawn as a rounded inset in
    the player's cam band (corner: top-left / top-center / top-right / bottom-...). It runs on the cam's clock, so it keeps moving while the game
    screen holds a tier card."""
    reader: Reader
    offset: float       # cam_offset of that player (screen_t = cam_t - offset)
    duration: float     # screen recording length
    x: int
    y: int
    alpha: np.ndarray   # rounded-rect mask incl. border
    inner: np.ndarray   # rounded-rect mask of the video itself

    def draw(self, cv: np.ndarray, cam_t: float):
        t = min(max(cam_t - self.offset, 0.0), self.duration - 0.05)
        img = self.reader.get(t)
        h, w = self.alpha.shape[:2]
        b = (h - img.shape[0]) // 2
        tile = np.full((h, w, 3), 245, np.float32)              # light border
        inner = tile[b:b + img.shape[0], b:b + img.shape[1]]
        m = self.inner
        inner[:] = inner * (1 - m) + img.astype(np.float32) * m
        roi = cv[self.y:self.y + h, self.x:self.x + w].astype(np.float32)
        cv[self.y:self.y + h, self.x:self.x + w] = (roi * (1 - self.alpha) + tile * self.alpha).astype(np.uint8)


class Compositor:
    def __init__(self, sess: Session, sync: dict, cfg: dict):
        o, lay = cfg["output"], cfg["layout"]
        self.W, self.H, self.fps = int(o["width"]), int(o["height"]), int(o["fps"])
        self.cams: dict[str, CamTile] = {}
        self.games: dict[str, Reader] = {}
        self.cam_h: dict[str, int] = {}
        # keep the game (and its answer bar) clear of Instagram's account name / caption overlay
        self.safe = _even(lay["bottom_safe"]) if lay.get("bottom_safe") else 0
        usable = self.H - self.safe
        for p in PLAYERS:
            # game fills the width; cam gets the rest (never less than min_cam_height)
            gx, gy, gw_, gh_ = sync["crop"][p]["rect"]
            gh = _even(self.W * gh_ / gw_)
            gw = self.W
            if gh > usable - lay["min_cam_height"]:
                gh = _even(usable - lay["min_cam_height"])
                gw = _even(gh * gw_ / gh_)
            cam_h = usable - gh
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
        self.bubbles: dict[str, Bubble] = {}
        for p, ft in (sync.get("facetime") or {}).items():
            if p not in PLAYERS or not ft.get("rect"):
                continue
            x, y, w, h = (int(v) for v in ft["rect"])
            border, margin = 5, 28
            bw = _even(self.W * float(ft.get("width", 0.34)))
            bh = _even(bw * h / w)
            bh = min(bh, _even(self.cam_h[p] - 2 * margin - 2 * border))
            bw = min(bw, _even(bh * w / h))
            vf = f"fps={self.fps},crop={w}:{h}:{x}:{y},scale={bw}:{bh}:flags=bicubic"
            ow, oh = bw + 2 * border, bh + 2 * border
            corner = ft.get("corner", "top-right")
            ox = (margin if "left" in corner else self.W - margin - ow if "right" in corner
                  else (self.W - ow) // 2)  # "top-center" / "bottom-center"
            oy = margin if "top" in corner else self.cam_h[p] - margin - oh
            self.bubbles[p] = Bubble(Reader(sess.files[f"{p}_screen"], vf, bw, bh, self.fps),
                                     float(sync["cam_offset"][p]["seconds"]),
                                     float(sync["sources"][f"{p}_screen"]["duration"]), ox, oy,
                                     _rounded_mask(ow, oh, 34), _rounded_mask(bw, bh, 30))

    def frame(self, plan: Plan, i: int) -> np.ndarray:
        p = PLAYERS[int(plan.shown[i])]
        cv = np.empty((self.H, self.W, 3), np.uint8)
        ch = self.cam_h[p]
        raw = self.cams[p].reader.get(float(plan.cam_t[i]))
        cv[0:ch] = self.cams[p].render(raw, float(plan.zoom[p][i]))
        if p in self.bubbles:
            self.bubbles[p].draw(cv, float(plan.cam_t[i]))
        g = self.games[p].get(float(plan.screen_t[i]))
        gh, gw = g.shape[:2]
        if gw < self.W:  # side bars: dimmed blur of the game behind everything below the cam
            cv[ch:] = _blur_cover(g, self.W, self.H - ch)
        elif self.safe:  # bottom safe band only
            band = _blur_cover(g, self.W, self.H - ch)
            cv[ch + gh:] = band[gh:]
        x = (self.W - gw) // 2
        cv[ch:ch + gh, x:x + gw] = g
        cv[ch - 2:ch + 2] = DIVIDER
        return cv

    def close(self):
        for r in ([c.reader for c in self.cams.values()] + list(self.games.values())
                  + [b.reader for b in self.bubbles.values()]):
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
        return out

    for pc in plan.pieces:
        o0 = int(round(pc.out_frame / plan.fps * rate))
        n = min(int(round(pc.frames / plan.fps * rate)), n_total - o0)
        if n <= 0:
            continue
        p, tr = pc.team, pc.track
        n_src = int(round(n * pc.speed))
        y = take(cams[p], tr.cam_t(pc.r0), n_src)
        if p in scr:
            valid = int(round(max(0.0, tr.length - pc.r0) * rate))
            y += g_gain * take(scr[p], tr.start + pc.r0, n_src, valid)
        if pc.speed != 1.0:
            y = _atempo(y, pc.speed, rate, n)
        if n > 2 * fade_n:
            y[:fade_n] *= ramp
            y[n - fade_n:] *= ramp[::-1]
        mix[o0:o0 + n] = y
    st = np.stack([mix, mix], axis=1)
    if frames is not None:
        a, b = (int(round(f / plan.fps * rate)) for f in frames)
        st = st[a:b]
    raw = out_wav.with_name(out_wav.stem + "_raw.wav")
    media.write_wav(raw, st * 0.5, rate)
    _loudnorm(raw, out_wav, cfg["audio"]["lufs"], log)
    return out_wav


def _atempo(y: np.ndarray, speed: float, rate: int, n_out: int) -> np.ndarray:
    """Speed audio up/down without changing pitch (FFmpeg atempo), fitted to exactly n_out samples."""
    p = subprocess.run([ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error",
                        "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
                        "-af", f"atempo={speed:.4f}", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-"],
                       input=np.ascontiguousarray(y, np.float32).tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("atempo failed: " + p.stderr.decode(errors="replace")[-500:])
    z = np.frombuffer(p.stdout, np.float32)
    out = np.zeros(n_out, np.float32)
    out[:min(n_out, len(z))] = z[:n_out]
    return out


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
    """Audio once, then the video in `output.workers` chunks rendered in parallel processes
    (each with its own decoders + encoder), joined without re-encoding."""
    plan = make_plan(sync, edl, cfg)
    f0, f1 = frames if frames else (0, plan.n)
    f0, f1 = max(0, f0), min(f1, plan.n)
    wav = build_audio(sess, sync, cfg, plan, sess.work / f"{out.stem}_audio.wav", (f0, f1), log=log)
    k = max(1, int(cfg["output"].get("workers", 1)))
    fps = int(cfg["output"]["fps"])
    if k == 1 or f1 - f0 < 20 * fps * k:
        _video(sess, sync, edl, cfg, out, f0, f1, scale, wav, "", log)
        return out
    from concurrent.futures import ProcessPoolExecutor
    bounds = np.linspace(f0, f1, k + 1).round().astype(int)
    parts = [sess.work / f"{out.stem}_part{i}.mp4" for i in range(k)]
    log(f"  rendering {f1 - f0} frames in {k} parallel chunks")
    with ProcessPoolExecutor(k) as ex:
        futs = [ex.submit(_video, sess, sync, edl, cfg, parts[i], int(bounds[i]), int(bounds[i + 1]), scale,
                          None, f"[chunk {i + 1}/{k}] ") for i in range(k)]
        for f in futs:
            f.result()
    lst = sess.work / f"{out.stem}_parts.txt"
    lst.write_text("".join(f"file '{q.resolve().as_posix()}'\n" for q in parts), encoding="utf-8")
    ff.run(["-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(wav), "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)])
    for q in parts:
        q.unlink(missing_ok=True)
    return out


def _video(sess: Session, sync: dict, edl: dict, cfg: dict, out: Path, f0: int, f1: int, scale: float,
           wav: Path | None, tag: str = "", log=print) -> Path:
    plan = make_plan(sync, edl, cfg)
    comp = Compositor(sess, sync, cfg)
    W, H = _even(comp.W * scale), _even(comp.H * scale)
    oc = cfg["output"]
    args = [ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(comp.fps), "-i", "-"]
    if wav is not None:
        args += ["-i", str(wav), "-map", "0:v", "-map", "1:a"]
    args += ["-c:v", "libx264", "-preset", oc["preset"], "-crf", str(oc["crf"]), "-pix_fmt", "yuv420p"]
    if wav is not None:
        args += ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart"]
    enc = subprocess.Popen(args + [str(out)], stdin=subprocess.PIPE)
    try:
        for i in range(f0, f1):
            fr = comp.frame(plan, i)
            if scale != 1.0:
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            enc.stdin.write(fr.tobytes())
            if (i - f0) % (comp.fps * 10) == 0:
                print(f"  {tag}frame {i - f0}/{f1 - f0} ({(i - f0) / comp.fps:.0f}s)", flush=True)
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
