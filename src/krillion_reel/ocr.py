"""Lazy RapidOCR wrapper.

Single-threaded on purpose: on this class of laptop CPU, onnxruntime gets *slower* with more
threads (measured 0.55s @1 thread vs 2.5s @8 threads for the same ROI).
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logging.getLogger("RapidOCR").setLevel(logging.ERROR)
logging.getLogger("rapidocr_onnxruntime").setLevel(logging.ERROR)


@lru_cache
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1,
                    det_limit_type="max", det_limit_side_len=1600)


def ocr(img: np.ndarray, scale: float = 1.0) -> list[tuple[tuple[float, float, float, float], str, float]]:
    """Return [(x0, y0, x1, y1), text, score] in the coordinates of `img` (before `scale`)."""
    import cv2

    src = img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    res, _ = _engine()(src)
    out = []
    for box, text, score in res or []:
        xs = [p[0] / scale for p in box]
        ys = [p[1] / scale for p in box]
        out.append(((min(xs), min(ys), max(xs), max(ys)), str(text), float(score)))
    return out


def norm(text: str) -> str:
    """Lowercase, letters/digits only (OCR often drops or inserts spaces)."""
    return "".join(ch for ch in text.lower() if ch.isalnum())
