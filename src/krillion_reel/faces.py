"""Cam framing: find everyone's face across the clip and pick one static crop that keeps them all.

A static crop (instead of tracking) never jitters; zooms are layered on top at render time.
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import ff

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
SAMPLE_W = 960


def _model(work: Path) -> Path:
    m = work.parent / "_models" / "face_detection_yunet_2023mar.onnx"
    if not m.exists():
        m.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, m)
    return m


def _sample_frames(src: Path, every_s: float = 2.0):
    info = ff.probe(src)
    h = int(round(SAMPLE_W * info.height / info.width / 2)) * 2
    args = [ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(src),
            "-vf", f"fps=1/{every_s},scale={SAMPLE_W}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    size = SAMPLE_W * h * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, SAMPLE_W, 3).copy()
    p.wait()


def frame_people(src: Path, work: Path, debug_png: Path | None, log=print) -> dict:
    """Returns {"box": people box [x0,y0,x1,y1] incl. margins (display coords), "focus", "frame": [W,H]}.

    The crop at the tile's aspect is derived at render time (fit_aspect), so the layout can change
    without re-running face detection.
    """
    info = ff.probe(src)
    W, H = info.width, info.height
    s = W / SAMPLE_W
    det = None
    boxes, last = [], None
    for img in _sample_frames(src):
        if det is None:
            det = cv2.FaceDetectorYN.create(str(_model(work)), "", (img.shape[1], img.shape[0]), 0.75, 0.3, 5000)
        _, faces = det.detect(img)
        if faces is not None:
            for f in faces:
                x, y, w, h = (float(v) * s for v in f[:4])
                if w * h > (W * H) * 0.002:  # ignore tiny false positives (posters etc.)
                    boxes.append((x, y, w, h))
        last = img
    if not boxes:
        log(f"  WARNING: no faces found in {src.name}; using centre of frame")
        return {"box": [W * 0.2, H * 0.1, W * 0.8, H * 0.9], "focus": [W / 2, H / 2], "frame": [W, H], "faces": 0}
    b = np.array(boxes)
    x0 = np.percentile(b[:, 0], 3)
    x1 = np.percentile(b[:, 0] + b[:, 2], 97)
    y0 = np.percentile(b[:, 1], 3)
    y1 = np.percentile(b[:, 1] + b[:, 3], 97)
    fs = float(np.median(np.maximum(b[:, 2], b[:, 3])))
    box = [max(0.0, x0 - 0.4 * fs), max(0.0, y0 - 0.5 * fs), min(W, x1 + 0.4 * fs), min(H, y1 + 1.0 * fs)]
    focus = [float(np.median(b[:, 0] + b[:, 2] / 2)), float(np.median(b[:, 1] + b[:, 3] / 2))]
    log(f"  {len(boxes)} face detections; people box {[round(v) for v in box]} focus {[round(v) for v in focus]}")
    if debug_png is not None and last is not None:
        dbg = last.copy()
        for (x, y, w, h) in boxes[-40:]:
            cv2.rectangle(dbg, (int(x / s), int(y / s)), (int((x + w) / s), int((y + h) / s)), (0, 200, 255), 1)
        cv2.rectangle(dbg, (int(box[0] / s), int(box[1] / s)), (int(box[2] / s), int(box[3] / s)), (0, 255, 0), 3)
        cv2.imwrite(str(debug_png), dbg)
    return {"box": [round(v, 1) for v in box], "focus": [round(v, 1) for v in focus], "frame": [W, H],
            "faces": len(boxes)}


def fit_aspect(box, W, H, aspect) -> list[int]:
    """Crop rect [x, y, w, h] at `aspect` (w/h) around the people box, clamped inside the frame."""
    return _fit_aspect(*box, W, H, aspect)


def _fit_aspect(x0, y0, x1, y1, W, H, aspect) -> list[int]:
    """Grow the box to `aspect` (w/h) where possible and clamp it inside the frame.

    If the people are spread too wide to fit at the tile aspect, the box stays wider than the
    tile and the renderer letterboxes it over a blurred copy.
    """
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if h > H:
        h = H
    if w > W:
        w = W
    x = min(max(cx - w / 2, 0), W - w)
    y = min(max(cy - h / 2, 0), H - h)
    ev = lambda v: int(round(v / 2)) * 2  # noqa: E731
    return [ev(x), ev(y), ev(w), ev(h)]
