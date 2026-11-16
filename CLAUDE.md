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
  The frame each card holds on is `tier_end - 0.25`: it must still show the card (day 5 froze on a
  "checking your result" frame because tier_end was 0.5s late; sync now refines it at 4 fps).
* **Cam offsets.** This is the step that went wrong before, so check it carefully.
  * `audio_xcorr` with confidence over 3 and agreeing windows: the cam heard the game, so trust it.
  * `sfx_match`: the game's own sound effects found waveform-exact in the cam mic. 5+ events agreeing
    within 0.03s is exact (confirmed automatically); fewer: check `sfx_evidence`. Days 4 and 5 both
    synced this way once it existed (day 4's hand estimates were 2.3s / 1.0s off).
  * A fallback (`prompt_read...`): it was 1-3s off on days 4 and 5, so never trust it as is. OCR when each prompt first appears on that team's
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
jokes, roasts, near-misses and good arguments, and cut filler. The rules (prompt reading once, 3s of
thinking after it, each team's last 5s before submitting, the uncut submit -> reaction shot) are
enforced automatically afterwards.
* **Length: exactly 2:59** (`output.target_seconds: 179`). Your picks and the answer sinking are
  filler, sped between 1.25x and 1.55x to land on 179s. `cut` prints the fit: add or drop picks until it
  says "lands exactly on the target".
* **Intro.** Keep most of p1's intro as `"intro"` in picks.json (only cut mic checks like "testing"
  and setup chatter like checking who's in frame).
* **Ending.** Post-game banter (roasts after the last card) can be kept: set
  `"results_window": {"p1": [scr_t0, scr_t1], ...}` in sync.json and pick from it with
  `"results": {"clips": [{"team", "r0", "r1"}]}` (r = seconds into that window, played in list order).
  End on p1's own sign-off and watch p1's cam to the end of the recording (day 5: they hold up the
  keyboard after the last line; the user wanted that in). Keep the ending short (~15s, day 6), and
  **show both final scores** so viewers know who won: p1's score for 1s+ (results clips take
  `"speed": 1.0`; p1 often scrolls away from the score fast), then cut back to p2 on theirs.
* **Too long?** Before dropping questions, play deep cuts at 1.25x for that session with
  `"speed": {"normal_tiers": []}` in picks.json (day 6: user preferred all questions over 1x reveals).
  Drop at most one question.
* **Fit fast.** Start picks lean (one short clip per team per middle). The fixed parts (deep cuts at
  1x, ~11s each) eat most of the budget; drop the weakest question early if over by 25s+.
* **3+ players** (day 4 had a guest): set `"groups": [["p1", "p2"]]` in sync.json (roommates vs the
  guest). Each question shows the better roommate (override with `"featured"`) vs the guest; the other
  roommate gets a 2s look at their card.
* **FaceTime teammate** on someone's screen recording: set `"facetime": {"p2": {"rect": [x, y, w, h],
  "corner": "top-center"}}` in sync.json (screen pixels; top-left / top-center / top-right). It's drawn as a bubble on that team's cam. Make
  sure the game crop excludes the FaceTime pane and any ads.

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
* Reels must be **2:59 or shorter, ideally exactly 2:59** (IG doesn't push longer reels). No dead space.
* Every reel opens on p1's intro (p1 cam + idle p1 screen). Keep most of it (user, day 5).
* **Never re-render or overwrite a reel the user has already posted** unless they ask.
* Ask before committing.
