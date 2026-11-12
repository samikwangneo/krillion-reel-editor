"""Frame and audio extraction via FFmpeg pipes, with on-disk caching of analysis proxies."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import ff


def grab_frame(src: Path, t: float, crop: tuple[int, int, int, int] | None = None, width: int | None = None) -> np.ndarray:
    """One BGR frame at time t (seconds), optionally cropped (x, y, w, h in display coords) and scaled."""
    import cv2

    vf = []
    if crop:
        x, y, w, h = crop
        vf.append(f"crop={w}:{h}:{x}:{y}")
    if width:
        vf.append(f"scale={width}:-2")
    args = [ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(src),
            "-frames:v", "1"]
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-f", "image2pipe", "-vcodec", "png", "-"]
    p = subprocess.run(args, capture_output=True, check=True)
    img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not grab frame at {t:.2f}s from {src}")
    return img


def video_proxy(src: Path, cache: Path, fps: float = 10, width: int = 640,
                crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Decode the whole video to a small BGR uint8 array [n, h, w, 3] at `fps`. Cached as .npy."""
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    info = ff.probe(src)
    vf = []
    if crop:
        x, y, w, h = crop
        vf.append(f"crop={w}:{h}:{x}:{y}")
        aspect = h / w
    else:
        aspect = info.height / info.width
    height = int(round(width * aspect / 2)) * 2
    vf += [f"fps={fps}", f"scale={width}:{height}:flags=area"]
    args = [ff.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-threads", "0", "-i", str(src),
            "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.run(args, capture_output=True, check=True)
    arr = np.frombuffer(p.stdout, np.uint8).reshape(-1, height, width, 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def audio_track(src: Path, track: int, out: Path, rate: int = 16000) -> np.ndarray:
    """Extract one audio stream as mono float32 at `rate`. Cached as .wav; returns samples."""
    import wave

    if not out.exists():
        ff.run(["-i", str(src), "-map", f"0:a:{track}", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(out)])
    with wave.open(str(out), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return data.astype(np.float32) / 32768.0


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    import wave

    s = np.clip(samples, -1, 1)
    if s.ndim == 1:
        s = s[:, None]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(s.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((s * 32767).astype(np.int16).tobytes())
