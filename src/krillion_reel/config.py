"""Config loading: config.yaml merged over defaults, plus per-session path resolution."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "raw_root": "raw",
    "out_root": "out",
    "work_root": "work",
    "files": {
        "p1_screen": "p1_screen.*",
        "p2_screen": "p2_screen.*",
        "p1_cam": "p1_cam.*",
        "p2_cam": "p2_cam.*",
    },
    # Which audio track of each cam is the mic (index among audio streams). null = auto (loudest speech band).
    "cam_audio_track": {"p1": None, "p2": None},
    "questions": 7,
    "drop_questions": [],
    "tiers": ["nothing landed", "plankton", "schooler", "rare", "deep cut", "one in a krillion"],  # nothing landed = timed out / no score
    # lead: seconds of prompt card shown before each clock start; ending: hold on the results screen
    "timeline": {"lead": 3.5, "include_results": True, "ending_hold": 3.5},
    # cut a gap only if the ON-SCREEN team is silent and their game screen is static, and never while
    # they're still thinking (prompt card -> answer submitted) when protect_thinking is on
    "cut": {"min_gap": 1.5, "padding": 0.4, "protect_before": 0.5, "protect_after": 2.0,
            "protect_thinking": True, "static_motion": 0.004, "speech_db_above_floor": 12},
    # boring stretches play at `boring` speed (video + pitch-preserved audio); reveals, the moment the
    # answer is submitted, the results and loud reactions (each team's loudest `reaction_percentile`
    # of mic level, padded by reaction_pad [before, after]) stay at normal speed
    # everything at `boring` speed except the reveal + reaction of a normal_tiers card
    "speed": {"enabled": True, "boring": 1.25, "normal_tiers": ["deep cut", "one in a krillion"],
              "around_reveal": [0.3, 3.0], "reaction_max": 6.0, "reaction_percentile": 93, "reaction_pad": [0.5, 1.5],
              "slow_submit": False, "around_submit": [1.5, 0.5], "slow_reactions": False, "slow_results": False,
              "min_fast": 2.0},
    # One team on screen at a time (their cam on top, their game below), cutting between teams.
    "layout": {
        "answering": "model",    # model | highlights | interleave | sequential | alternate (see edl.py)
        "highlights": {          # answering: highlights - the shorter cut
            "think_after_read": 5.0,     # each team's first seconds of thinking after reading the prompt
            "middle": 4.0,               # liveliest bit of each team's discussion in between
            "end_before_submit": 10.0,   # always keep each team's last seconds before submitting
            "no_cut_before_submit": 3.0, # final shot runs uncut from here through the card + reaction
            "read_max": 8.0,             # look this long (after the segment start) for them reading aloud
            "read_fallback": 1.5,        # nobody read it aloud: show the prompt until clock start + this
        },
        "pause_search": 4.0,     # interleave: look this far for the end of a line before cutting anyway
        "reveal_before": 1.0,    # always show a team from this long before its reveal...
        "reveal_after": 2.5,     # ...until this long after
        "big_reveal_after": 4.0, # deep cut / krillion: at least this long, longer while they're still loud
        "min_shot": 3.0,         # otherwise follow whoever is talking, holding each shot at least this long
        "max_shot": 8.0,         # and force a switch after this long so it keeps alternating
        "min_cam_height": 620,   # cam band never shorter than this (game is scaled to the rest)
        "bottom_safe": 150,      # px kept clear at the bottom for the IG account name / caption
    },
    "zoom": {
        "enabled": True,
        "strength": 1.0,
        "ease": 0.2,
        "hold": 0.6,
        "ease_out": 0.4,
        "idle_after": 4.0,
        "idle_strength": 1.04,
        "tiers": {
            "nothing landed": 1.06,
            "plankton": 1.08,
            "schooler": 1.10,
            "rare": 1.15,
            "deep cut": 1.20,
            "one in a krillion": 1.30,
        },
    },
    "whisper": {"model": "small.en", "device": "cpu", "compute_type": "int8", "language": "en"},
    # audio follows the on-screen team: their mic + their game sound (if the screen recording has it)
    "audio": {"lufs": -14, "game_audio_db": -12},
    "output": {"width": 1080, "height": 1920, "fps": 30, "max_seconds": None, "preset": "veryfast", "crf": 20,
               "workers": 1},   # workers: render this many chunks in parallel
}

SOURCES = ("p1_screen", "p2_screen", "p1_cam", "p2_cam")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    if not path.exists():
        path.write_text(yaml.safe_dump(DEFAULTS, sort_keys=False), encoding="utf-8")
        print(f"[config] wrote default config to {path}")
    user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _merge(DEFAULTS, user)


@dataclass
class Session:
    name: str
    raw: Path
    out: Path
    work: Path
    files: dict[str, Path]

    @property
    def sync_json(self) -> Path:
        return self.out / "sync.json"

    @property
    def edl_json(self) -> Path:
        return self.out / "edl.json"


def resolve_session(cfg: dict, session_dir: str | None) -> Session:
    raw_root = Path(cfg["raw_root"])
    if session_dir:
        raw = Path(session_dir)
    else:
        subs = [p for p in raw_root.iterdir() if p.is_dir()] if raw_root.exists() else []
        raw = max(subs, key=lambda p: p.stat().st_mtime) if subs else raw_root
    if not raw.exists():
        raise SystemExit(f"session folder not found: {raw}")
    name = raw.name if raw != raw_root else "default"
    files: dict[str, Path] = {}
    entries = list(raw.iterdir())
    for key in SOURCES:
        pat = cfg["files"][key].lower()
        stem = pat.split(".")[0]
        hits = [p for p in entries if p.is_file() and p.stem.lower() == stem
                and (pat.endswith(".*") or p.name.lower() == pat)]
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one file matching {cfg['files'][key]!r} in {raw}, found {[h.name for h in hits]}")
        files[key] = hits[0]
    out = Path(cfg["out_root"]) / name
    work = Path(cfg["work_root"]) / name
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return Session(name=name, raw=raw, out=out, work=work, files=files)
