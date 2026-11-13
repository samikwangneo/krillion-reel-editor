# krillion-reel: how to "edit dayN"

This repo turns raw Krillion 1v1 footage into a vertical Instagram Reel. The README covers how it
works. When the user drops footage into `raw/dayN/` and says **"edit dayN"** (or "edit this day"
= the newest folder in `raw/`), do all of the steps below without stopping, unless a step says to ask.

Use Git Bash for every command (never PowerShell). The CLI is `.venv/Scripts/krillion-reel`. FFmpeg
lives in `C:/Users/samik/AppData/Local/Microsoft/WinGet/Links` (not on PATH; the code finds it).
The user's rule: **ask before guessing anything you can't detect reliably.**

## 1. Sync (about 15-20 min, run in the background)
`krillion-reel sync raw/dayN`. It finds the game crop, the 7 questions (clock / submit / reveal /
tier / answer), the transcripts (WhisperX) and each cam's offset (`cam_t = screen_t + offset`).

## 2. Check what sync found
* **Crop.** Open `out/dayN/debug_crop_p1.png` and `debug_crop_p2.png`. The top bar (DEPTH,
  progress dots, SCORE) must be inside the green box for both. If the recording itself doesn't
  show it (a short browser window), **tell the user before rendering**.
* **Questions.** Every question needs a tier for both teams. If `tier` is null, look at the frames
  in `work/dayN/pX_screen_ocr/` (4 per second, file = round(t*4)). A timeout shows a
  NOTHING LANDED card. Fill in `reveal` and `tier` in `out/dayN/sync.json` by hand and add a note.
* **Cam offsets.** This is the step that went wrong before, so check it carefully.
  * `audio_xcorr` with confidence over 3 and agreeing windows: the cam heard the game, so trust it.
  * A fallback (`prompt_read...`): verify it. OCR when each prompt first appears on that team's
    screen (top half of the frames, two prompt words matched). Then check against their WhisperX
    words:
    * They can't read a prompt aloud before it appears. People start reading 0.0-0.3s after it
      appears; calibrate on the other team if that team's offset is exact.
    * An answer said aloud before submitting gives a lower bound.
    * Deep cut / "let's go" reactions should land just after the card.
  * Set `seconds`, `method: "manual"`, `confirmed: true` and a `note` with the evidence.
  * Don't trust "it looks 0.5s off" reports without checking the render first: day 2's "sync
    problems" were a renderer bug.

## 3. Pick the clips (Claude's job)
Run `krillion-reel brief raw/dayN`, read `out/dayN/brief.md` and the `brief_frames/qN.jpg`
sheets, then write `out/dayN/picks.json` following the instructions at the top of the brief. Keep
it shorter than the full thing, keeping jokes, roasts, near-misses and good arguments, and cutting
filler. The rules (prompt reading, 5s of thinking each, both teams' last 10s, the uncut
submit -> reaction shot) are enforced automatically afterwards.

## 4. Review, render, verify
* `krillion-reel cut raw/dayN`. Read the clip list: `model:` = your picks, `rule:` = added by the
  rules.
* `krillion-reel stills raw/dayN` and look at the stills (framing, top bar visible).
* `krillion-reel render raw/dayN` in the background. It takes about 3-4x the reel length (about
  20 min for a 5-min reel). Tell the user the expected finish time.
* Verify before saying it's done:
  * ffprobe shows 1080x1920, 30fps, h264 + aac.
  * The `ebur128` integrated loudness is about -14 LUFS.
  * A contact sheet (`fps=1/12,scale=180:320,tile=10x3`) looks right.
  * `.venv/Scripts/python tools/verify_render.py raw/dayN` shows lags of about 0.00s.
* Report the length, the path, and which offsets are estimates.

## User preferences (settled; don't re-ask)
* Everything plays at 1.25x except DEEP CUT / ONE IN A KRILLION reveals and reactions (1x).
* One team on screen at a time, no P1/P2 labels. The game sits 150px above the bottom (IG caption).
* No length limit, but no dead space.
* **Never re-render or overwrite a reel the user has already posted** unless they ask.
* Ask before committing.
