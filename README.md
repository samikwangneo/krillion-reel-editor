# krillion-reel-editor

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

Individual steps: `sync`, `cut`, `review`, `stills`, `preview`, `render`. `--redetect` ignores the
cached `sync.json` and detection caches; `--force` renders despite open checkpoints.

## Files you can hand-edit

* `out/<session>/sync.json` - cam offsets (`cam_t = screen_t + seconds`), game crops, cam framing
  (`cam_view`), and per-question `clock_start` / `reveal` / `tier_end` / `tier`. Re-running uses it as-is.
* `out/<session>/edl.json` - rebuilt from sync.json + config on every run; read the `cuts[].reason`
  to review what was removed.
* `config.yaml` - padding / min gap / protected windows, speed-ups (`speed.enabled: false` or
  `speed.boring: 1.0` turns them off), zoom strength (`zoom.enabled: false` turns zooms off),
  layout, loudness. No length limit by default; set `output.max_seconds` to get drop suggestions,
  or list questions in `drop_questions`.

## How it works

* **Sync.** Screen events come from the game UI. The game area is found from the DEPTH / SCORE bar
  and DIVE button (or from DIVE alone when a short browser window scrolls the bar out of view). The pink DIVE button marks when the clock is
  running. OCR of the tier card ("DEEP CUT", "RARE", ... and the DESCEND/SURFACE button) marks the
  reveal. The prompt text pairs questions across players. Each cam is matched to its screen by
  cross-correlating audio, which works when the cam mic hears the game's sounds. If the screen
  recording has no audio, the offset comes from when the players read each prompt aloud (WhisperX),
  refined by their loud reactions just after each reveal.
* **Timeline.** Each question segment starts 3.5s before the clock starts, so both prompts appear
  together. It ends when the slower player clicks DESCEND. The faster player's screen holds on its
  tier card, while their cam keeps rolling.
* **Layout.** One team is on screen at a time: their cam on top and their game below, filling the
  width and sitting 150px above the bottom (`layout.bottom_safe`) so the answer bar stays clear of
  the Instagram caption. That band is a dimmed blur of the game.
  The video cuts fully between teams. Each question is a sequence of clips (`edl.json` -> `clips`).
  With `layout.answering: model` (the default), Claude picks the clips:
  1. `krillion-reel brief raw/dayN` (or `run`, which stops here) writes `out/dayN/brief.md`. It has
     each question's timings, both teams' timestamped transcripts with loud lines marked, the
     loudest moments, the automatic cut for comparison, and frame sheets in
     `out/dayN/brief_frames/`.
  2. In Claude Code, ask Claude to **"edit dayN"**. It reads the brief and writes
     `out/dayN/picks.json`: per question, the opening and middle clips it thinks are worth keeping,
     each with a reason and an optional `"speed": 1.0`. This runs on your Claude subscription;
     no API key is needed.
  3. `krillion-reel cut raw/dayN` to review, then `render`. The picks are checked against the rules
     below:
     * Anything the picks leave out of the reading or first-5s windows is added back.
     * Nothing replays, and a team never jumps backwards.
     * The last 10s and the uncut submit -> reaction shots are always added automatically.

     Each clip in `edl.json` says `model: ...` or `rule: ...`. A question missing from `picks.json`
     gets the automatic highlights cut. Without a `picks.json`, `run`/`render` stop at a checkpoint;
     `--force` renders the automatic cut instead.

  With `highlights` (the automatic shorter cut), each question is:
  1. one team reading the prompt aloud, from when it appears until they finish;
  2. 5s of that team thinking, then 5s of the other team right after *they* finish reading;
  3. the liveliest 4s (loudest and most talking) of each team's middle discussion, with the rest of
     the middle cut;
  4. both teams' last 10s before submitting, interleaved. Each team's final shot runs uncut from 3s
     before they submit, through the card dropping and the whole reaction.

  Lengths are in `layout.highlights`. Two clips of the same team less than 1.5s apart play straight
  through, so there are no micro-jumps.

  With `interleave`, it cuts back and forth while the teams think,
  but each team keeps its own place, so going back to a team picks up exactly where it was left and
  no thinking is lost:
  * A shot lasts at least 3s and until the team has caught up with where the other team was left.
  * It then cuts right after the team on screen finishes a line: a sentence end or breath from the
    WhisperX words, otherwise a gap between words, looking up to `pause_search` (4s) ahead.
  * Once both teams have submitted, both reveals play. The lower tier goes first so the better card
    lands last. On a tie, the team that goes first alternates from question to question.

  Other modes:
  * `sequential`: one team's whole think, then the other's.
  * `alternate`: shorter, but about half of each team's thinking is lost. Both teams share one clock.
* **Cuts.** A stretch of at least 1.5s is cut only when the on-screen team is silent (no words and
  quiet mic) and their game screen is static. The thinking time (prompt card up to the answer being
  submitted) and 0.5s before to 2s after a reveal are never cut. 0.4s of padding is kept on each
  side of a cut.
* **Speed.** Everything plays at 1.25x (video and pitch-preserved audio), except a DEEP CUT or ONE IN
  A KRILLION reveal (`speed.normal_tiers`). Those run at normal speed from 0.3s before the card to
  3s after, and stay at normal speed while the team is still loud (their loudest 7% of mic level),
  up to 6s after.
  * The reveal clip stays on for at least 4s for those tiers (`layout.big_reveal_after`), even after
    they click DESCEND. The tier card holds while the cam keeps rolling.
  * `slow_submit`, `slow_reactions` and `slow_results` turn normal speed back on for those parts.
  * Fast bits shorter than 2s stay at normal speed.
* **Render.** FFmpeg decodes, crops and scales each source. Python composites the frame, applies
  zoom punches, and pipes it to libx264 (veryfast). The audio follows the on-screen team: their mic, plus their game sound at -12 dB if it was
  recorded. It is loudness-normalized in two passes to -14 LUFS.
