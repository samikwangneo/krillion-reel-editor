# krillion-reel

Turns raw Krillion 1v1 footage into a vertical 1080x1920 Instagram Reel.

## Setup (once)

```bash
winget install Gyan.FFmpeg                       # ffmpeg + ffprobe
uv venv --python 3.12 .venv                      # WhisperX doesn't support 3.13+ yet
uv pip install --python .venv/Scripts/python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e . pytest
```

## Each new session

Drop the four recordings in a new folder (any extension, case-insensitive):

```
raw/day2/p1_screen.mov   raw/day2/p2_screen.mov   raw/day2/p1_cam.mov   raw/day2/p2_cam.mov
```

then

```bash
.venv/Scripts/krillion-reel run raw/day2       # no argument = newest folder in raw/
```

`run` goes sync -> cut -> preview -> render and stops at checkpoints that need you:

| checkpoint | what to do |
|---|---|
| cam sync came from a fallback (screen recording had no audio) | watch `out/<session>/preview.mp4`; if OK set `cam_offset.pX.confirmed: true` in `sync.json`, otherwise edit `seconds` |
| not exactly 7 prompts / a tier not recognised | fix the entry in `sync.json` (frames are in `work/<session>/pX_screen_ocr/`, 4 per second) |
| reel longer than `max_seconds` (180) | it prints which questions to drop; put them in `drop_questions` |

Individual steps: `sync`, `cut`, `review`, `stills`, `preview`, `render`. `--redetect` ignores the
cached `sync.json` and detection caches; `--force` renders despite open checkpoints.

## Files you can hand-edit

* `out/<session>/sync.json` - cam offsets (`cam_t = screen_t + seconds`), game crops, cam framing
  (`cam_view`), and per-question `clock_start` / `reveal` / `tier_end` / `tier`. Re-running uses it as-is.
* `out/<session>/edl.json` - rebuilt from sync.json + config on every run; read the `cuts[].reason`
  to review what was removed.
* `config.yaml` - player names, padding / min gap / protected reveal window, zoom strength
  (`zoom.enabled: false` turns zooms off), layout, loudness, output limits.

## How it works

* **Sync.** Screen events come from the game UI. The pink DIVE button marks when the clock is
  running. OCR of the tier card ("DEEP CUT", "RARE", ... and the DESCEND/SURFACE button) marks the
  reveal. The prompt text pairs questions across players. Each cam is matched to its screen by
  cross-correlating audio, which works when the cam mic hears the game's sounds. If the screen
  recording has no audio, the offset comes from when the players read each prompt aloud (WhisperX),
  refined by their loud reactions just after each reveal.
* **Timeline.** Each question segment starts 3.5s before the clock starts, so both prompts appear
  together. It ends when the slower player clicks DESCEND. The faster player's screen holds on its
  tier card, while their cam keeps rolling.
* **Layout.** One team is on screen at a time: their cam on top and their game below, filling the
  width. The video cuts fully between teams. Each question is a sequence of clips (`edl.json` -> `clips`):
  1. the prompt and the answering, cutting between teams by who is talking (each shot at least 3s,
     switching after 8s at most);
  2. the first team's reveal, from 1s before the card to 2.5s after;
  3. the second team's reveal, rewound a few seconds if both cards landed at the same moment.
* **Cuts.** A stretch is cut only when the on-screen team is silent (no words and quiet mic) and
  their game screen is static. Nothing within 0.5s before or 2s after a reveal is cut. 0.25s of padding is
  kept on each side of a cut.
* **Render.** FFmpeg decodes, crops and scales each source. Python composites the frame, applies
  zoom punches and labels, and pipes it to libx264 (veryfast). The audio follows the on-screen team: their mic, plus their game sound at -12 dB if it was
  recorded It is loudness-normalized in two passes to -14 LUFS.
