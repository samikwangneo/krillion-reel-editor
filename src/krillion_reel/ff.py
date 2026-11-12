"""Thin FFmpeg/ffprobe subprocess helpers."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_WINGET_LINKS = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"


@lru_cache
def tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    cand = _WINGET_LINKS / f"{name}.exe"
    if cand.exists():
        return str(cand)
    raise SystemExit(f"{name} not found. Install with: winget install Gyan.FFmpeg")


def run(args: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    cmd = [tool("ffmpeg"), "-hide_banner", "-y"] + (["-loglevel", "error"] if quiet else []) + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({p.returncode}):\n{' '.join(cmd)}\n{p.stderr[-3000:]}")
    return p


@dataclass
class MediaInfo:
    path: Path
    duration: float
    width: int  # display width (after rotation)
    height: int
    rotation: int
    fps: float
    audio_streams: list[dict]


@lru_cache
def probe(path: Path) -> MediaInfo:
    p = subprocess.run(
        [tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    d = json.loads(p.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if not rot and "rotate" in v.get("tags", {}):
        rot = -int(v["tags"]["rotate"])
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = v.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    audio = [s for s in d["streams"] if s["codec_type"] == "audio"]
    return MediaInfo(Path(path), float(d["format"]["duration"]), w, h, rot, fps, audio)
