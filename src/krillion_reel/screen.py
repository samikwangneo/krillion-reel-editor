"""Krillion screen-recording analysis: game-area crop, per-question events, reveal tiers.

Krillion's per-question flow (observed in footage):
  prompt card appears + "descending · the clock starts in 3..1"
  -> answer box + pink DIVE button appear (clock running)         = clock_start
  -> player submits, answer box disappears, answer sinks (~3-4s)  = submit
  -> tier card: big tier word, "+NN PTS · sink NNNm", DESCEND btn = reveal
  -> player clicks DESCEND                                        = tier_end
After prompt 7 a results screen ("KRILLION" + total score) appears.
"""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

from . import ff, media
from .ocr import norm, ocr

PROXY_FPS = 10
OCR_FPS = 4
OCR_WIDTH = 800


# ---------------------------------------------------------------- crop detection

@dataclass
class GameCrop:
    x: int
    y: int
    w: int
    h: int
    dive: tuple[float, float, float, float]  # DIVE button box, display coords
    answer_band: tuple[float, float]          # y0, y1 of the answer row, display coords

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def detect_crop(src: Path, debug_png: Path | None = None, log=print) -> GameCrop:
    """OCR a handful of full frames and find the HUD anchors (DEPTH, SCORE, DIVE).

    Using game text as anchors (rather than motion) ignores browser chrome, the macOS dock and
    animated ads around the game, which all move too.
    """
    info = ff.probe(src)
    W, H = info.width, info.height
    anchors: dict[str, list] = {"depth": [], "score": [], "dive": [], "answer": []}
    times = np.linspace(info.duration * 0.08, info.duration * 0.7, 12)
    last_img = None
    for t in times:
        img = media.grab_frame(src, float(t))
        last_img = img if anchors["dive"] == [] else last_img
        for box, text, _ in ocr(img, scale=min(1.0, 1600 / W)):
            n = norm(text)
            if n == "depth":
                anchors["depth"].append(box)
            elif n in ("score", "6core", "scor", "core") or (len(n) <= 6 and fuzz.ratio(n, "score") >= 80):
                anchors["score"].append(box)
            elif n == "dive":
                anchors["dive"].append(box)
                last_img = img
            elif "oneanswer" in n:
                anchors["answer"].append(box)
        have = {k: len(v) for k, v in anchors.items()}
        log(f"  crop scan t={t:6.1f}s anchors={have}")
        if have["depth"] >= 2 and have["score"] >= 2 and have["dive"] >= 2:
            break
    med = lambda boxes, i: float(np.median([b[i] for b in boxes]))  # noqa: E731
    if not anchors["dive"]:
        raise RuntimeError(f"{src.name}: could not find the DIVE button - is this a Krillion recording?")
    dive = tuple(med(anchors["dive"], i) for i in range(4))
    bottom = dive[3]
    if anchors["depth"] and anchors["score"]:
        left, right = med(anchors["depth"], 0), med(anchors["score"], 2)
        top = min(med(anchors["depth"], 1), med(anchors["score"], 1))
        gw, gh = right - left, bottom - top
        x0 = max(0, left - 0.06 * gw)
        x1 = min(W, right + 0.06 * gw)
        y0 = max(0, top - 0.025 * gh)
        y1 = min(H, bottom + 0.045 * gh)
    else:
        # DEPTH/SCORE bar scrolled out of view (short browser window): the game column has a fixed
        # layout, so size it from the DIVE button (calibrated on full-HUD recordings: column width ~
        # 14.9x the DIVE label width, DIVE's right edge 0.381 widths right of centre) and take
        # everything visible above it
        w = 14.9 * (dive[2] - dive[0])
        cx = dive[2] - 0.381 * w
        x0, x1 = max(0, cx - w / 2), min(W, cx + w / 2)
        y0, y1 = 0.0, min(H, bottom + 0.04 * w)
        log(f"  {src.name}: no DEPTH/SCORE bar in view - game area estimated from the DIVE button")
    ev = lambda v: int(round(v / 2)) * 2  # noqa: E731
    crop = GameCrop(ev(x0), ev(y0), ev(x1 - x0), ev(y1 - y0), dive, (dive[1], dive[3]))
    if debug_png is not None and last_img is not None:
        dbg = last_img.copy()
        cv2.rectangle(dbg, (crop.x, crop.y), (crop.x + crop.w, crop.y + crop.h), (0, 255, 0), 6)
        cv2.rectangle(dbg, (int(dive[0]), int(dive[1])), (int(dive[2]), int(dive[3])), (255, 0, 255), 4)
        cv2.imwrite(str(debug_png), cv2.resize(dbg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    return crop


# ---------------------------------------------------------------- per-frame signals

def ocr_frames(src: Path, crop: GameCrop, outdir: Path) -> list[Path]:
    """Game-crop JPGs at OCR_FPS (frame i is at t = i / OCR_FPS)."""
    if not outdir.exists() or not any(outdir.iterdir()):
        outdir.mkdir(parents=True, exist_ok=True)
        x, y, w, h = crop.rect
        ff.run(["-threads", "0", "-i", str(src), "-vf",
                f"crop={w}:{h}:{x}:{y},fps={OCR_FPS},scale={OCR_WIDTH}:-2:flags=area",
                "-q:v", "3", "-start_number", "0", str(outdir / "%05d.jpg")])
    return sorted(outdir.glob("*.jpg"))


SIG_WIDTH = 400  # game-crop width used for the per-frame signals


def frame_signals(src: Path, crop: GameCrop, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Stream the game crop at PROXY_FPS and compute two cheap per-frame signals (cached as .npz):

    dive:   fraction of DIVE-pink pixels in the DIVE button box (clock running <=> button visible)
    motion: fraction of game-area pixels that changed vs the previous frame. Ambient animation
            (the roaming krill, particles) stays well under 1%; prompt changes, typing, the sinking
            answer and the tier card all spike above it.
    """
    if cache.exists():
        z = np.load(cache)
        return z["dive"], z["motion"]
    x, y, w, h = crop.rect
    s = SIG_WIDTH / w
    H = int(round(h * s / 2)) * 2
    dx0, dy0, dx1, dy1 = crop.dive
    px, py = (dx1 - dx0) * 0.15, (dy1 - dy0) * 0.15
    X0, Y0 = int((dx0 - px - x) * s), int((dy0 - py - y) * s)
    X1, Y1 = int(np.ceil((dx1 + px - x) * s)), int(np.ceil((dy1 + py - y) * s))
    args = [ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-threads", "0", "-i", str(src),
            "-vf", f"crop={w}:{h}:{x}:{y},fps={PROXY_FPS},scale={SIG_WIDTH}:{H}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    size = SIG_WIDTH * H * 3
    dive, motion, prev = [], [], None
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, SIG_WIDTH, 3)
        hsv = cv2.cvtColor(np.ascontiguousarray(f[Y0:Y1, X0:X1]), cv2.COLOR_BGR2HSV)
        hh, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        dive.append((((hh >= 160) | (hh <= 6)) & (sat >= 70) & (val >= 70)).mean())
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, max(2, int(160 * H / SIG_WIDTH))),
                       interpolation=cv2.INTER_AREA)
        motion.append(0.0 if prev is None else float((cv2.absdiff(g, prev) > 18).mean()))
        prev = g
    p.wait()
    d, m = np.asarray(dive, np.float32), np.asarray(motion, np.float32)
    np.savez(cache, dive=d, motion=m)
    return d, m


def runs(mask: np.ndarray, fps: float, min_len: float, max_gap: float) -> list[tuple[float, float]]:
    """Contiguous True runs as (start_s, end_s), bridging gaps <= max_gap, dropping runs < min_len."""
    out: list[list[float]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            s, e = i / fps, j / fps
            if out and s - out[-1][1] <= max_gap:
                out[-1][1] = e
            else:
                out.append([s, e])
            i = j
        else:
            i += 1
    return [(s, e) for s, e in out if e - s >= min_len]


# ---------------------------------------------------------------- events

@dataclass
class Question:
    n: int
    clock_start: float
    submit: float
    reveal: float | None = None
    tier_end: float | None = None
    tier: str | None = None
    answer: str | None = None
    points: int | None = None
    prompt: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ScreenEvents:
    crop: GameCrop
    questions: list[Question]
    results_start: float | None
    duration: float

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def _match_tier(texts: list[str], tiers: list[str]) -> tuple[str | None, float]:
    best, score = None, 0.0
    joined = [norm(t) for t in texts]
    for tier in tiers:
        tn = norm(tier)
        for t in joined:
            if not t:
                continue
            sc = 100.0 if tn == t else fuzz.ratio(tn, t)
            if sc > score:
                best, score = tier, sc
    return (best, score) if score >= 80 else (None, score)


CARD_BUTTONS = ("descend", "surface")  # tier-card button; prompt 7 uses SURFACE instead of DESCEND


def _bottom_texts(frame: np.ndarray) -> list[str]:
    h = frame.shape[0]
    return [t for _, t, _ in ocr(frame[int(h * 0.82):, :])]


def _has_card_button(texts: list[str]) -> bool:
    """True if the tier-card button is visible. Must not match 'descending · the clock starts in 3'."""
    for t in texts:
        n = norm(t)
        # <= 8 chars: OCR sometimes splits "descending" off on its own, which must not match
        if 5 <= len(n) <= 8 and any(fuzz.ratio(n, b) >= 75 for b in CARD_BUTTONS):
            return True
    return False


def _center_texts(frame: np.ndarray) -> list[tuple]:
    h = frame.shape[0]
    return [(b, t, s) for b, t, s in ocr(frame[int(h * 0.15):int(h * 0.80), :])]


def detect_events(src: Path, crop: GameCrop, work: Path, key: str, tiers: list[str], expect: int,
                  log=print) -> tuple[ScreenEvents, dict]:
    info = ff.probe(src)
    dive, motion = frame_signals(src, crop, work / f"{key}_signals.npz")
    frames = ocr_frames(src, crop, work / f"{key}_ocr")
    load = lambda t: cv2.imread(str(frames[min(len(frames) - 1, max(0, int(round(t * OCR_FPS))))]))  # noqa: E731

    # clock-running runs: DIVE button visible. Threshold halfway between the typical "on" level and 0.
    on_level = float(np.percentile(dive, 90))
    thr = max(0.04, on_level * 0.35)
    clock_runs = runs(dive > thr, PROXY_FPS, min_len=1.5, max_gap=0.6)
    log(f"  {key}: DIVE on-level={on_level:.3f} thr={thr:.3f} -> {len(clock_runs)} clock runs")

    questions: list[Question] = []
    for i, (s, e) in enumerate(clock_runs):
        q = Question(n=i + 1, clock_start=round(s, 2), submit=round(e, 2))
        # prompt text: OCR the card shortly after the clock starts
        texts = [t for _, t, _ in _center_texts(load(s + 0.5))]
        prompt_lines = [t for t in texts if len(t) > 3 and fuzz.partial_ratio(norm(t)[:6], "prompt") < 70
                        and "rarer" not in norm(t)]
        q.prompt = " ".join(prompt_lines)[:200] or None
        # reveal: first frame after submit whose bottom band says DESCEND
        limit = clock_runs[i + 1][0] if i + 1 < len(clock_runs) else info.duration
        t, found = e + 0.5, None
        while t < min(limit, e + 25):
            if _has_card_button(_bottom_texts(load(t))):
                found = t
                break
            t += 1 / OCR_FPS
        if found is None:
            q.notes.append("no DESCEND/SURFACE button found after submit (timeout or missed?)")
            questions.append(q)
            continue
        # refine reveal: the card fading in is the biggest motion spike around the first button hit
        # (the button is often OCR-readable a frame before the card finishes fading in)
        lo, hi = int((found - 1.0) * PROXY_FPS), int((found + 0.5) * PROXY_FPS) + 1
        seg = motion[max(0, lo):hi]
        q.reveal = round((max(0, lo) + int(np.argmax(seg))) / PROXY_FPS, 2) if len(seg) else round(found, 2)
        # tier word + answer + points: scan every OCR frame from just after the card lands. Some
        # players click DESCEND within half a second, so a single fixed sample can miss the card.
        card: list[str] = []
        tt = q.reveal + 0.25
        while tt < q.reveal + 3.0 and q.tier is None:
            texts = [t for _, t, _ in _center_texts(load(tt))]
            q.tier, _ = _match_tier(texts, tiers)
            if q.tier is not None or not card:
                card = texts
            tt += 1 / OCR_FPS
        if q.tier is None:
            q.notes.append(f"tier word not recognised; card text: {card}")
        for t_ in card:
            if t_.strip().startswith(("\"", "“", "'")) and q.answer is None:
                q.answer = t_.strip().strip("\"“”' ")
            n = norm(t_)
            if "pts" in n and q.points is None:
                digits = "".join(ch for ch in n.split("pts")[0] if ch.isdigit())
                q.points = int(digits) if digits else None
        # tier_end: button gone (player clicked). Scan at 2 fps; require 2 consecutive misses because
        # the mouse cursor hovering the button breaks OCR on single frames.
        t, misses, first_miss = found + 0.5, 0, None
        while t < limit:
            if _has_card_button(_bottom_texts(load(t))):
                misses, first_miss = 0, None
            else:
                misses += 1
                first_miss = first_miss if first_miss is not None else t
                if misses >= 2:
                    break
            t += 0.5
        q.tier_end = round(min(first_miss if first_miss is not None else t, limit), 2)
        questions.append(q)
        log(f"  {key} Q{q.n}: clock {q.clock_start:6.1f}  submit {q.submit:6.1f}  reveal {q.reveal}  "
            f"end {q.tier_end}  tier={q.tier} answer={q.answer!r} pts={q.points}")

    # results screen: "KRILLION" title + big total score, after the last tier card
    results = None
    if questions and questions[-1].tier_end:
        t = questions[-1].tier_end
        while t < min(info.duration, questions[-1].tier_end + 20):
            frame = load(t)
            texts = [norm(x) for _, x, _ in ocr(frame[: int(frame.shape[0] * 0.3)])]
            if any(fuzz.partial_ratio("krillion", x) >= 85 for x in texts if len(x) >= 6):
                results = round(t, 2)
                break
            t += 0.25
    signals = {"fps": PROXY_FPS, "dive": dive.round(4).tolist(), "motion": motion.round(4).tolist()}
    ev = ScreenEvents(crop=crop, questions=questions, results_start=results, duration=info.duration)
    if len(questions) != expect:
        log(f"  WARNING {key}: found {len(questions)} questions, expected {expect}")
    return ev, signals
