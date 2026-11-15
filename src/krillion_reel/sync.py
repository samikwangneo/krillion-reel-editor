"""Step 1 - SYNC: produce sync.json (cam offsets, game crops, per-question events, cam framing).

Everything here is cached: if sync.json exists it is loaded as-is (so hand edits stick) unless
--redetect is passed.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from . import audiosync, faces, ff, media, screen
from .config import PLAYERS, Session
from .transcribe import transcribe

CONF_OK = 3.0  # xcorr peak / next-best peak
SFX_VOTES_OK = 5  # independent sound effects that must agree (+-0.03s) to trust sfx_match without review


def cam_wav(sess: Session, cfg: dict, p: str, rate: int = 16000) -> Path:
    src = sess.files[f"{p}_cam"]
    track = cam_track(sess, cfg, p)
    out = sess.work / f"{p}_cam_a{track}{'' if rate == 16000 else f'_{rate}'}.wav"
    media.audio_track(src, track, out, rate)
    return out


def cam_track(sess: Session, cfg: dict, p: str) -> int:
    t = cfg.get("cam_audio_track", {}).get(p)
    if t is not None:
        return int(t)
    info = ff.probe(sess.files[f"{p}_cam"])
    if len(info.audio_streams) <= 1:
        return 0
    # several mic tracks (iPhone): take the loudest in the speech band. Skip tracks FFmpeg can't
    # decode (e.g. iPhone spatial audio, codec apple_apac).
    best, best_rms = 0, -1.0
    for i in range(len(info.audio_streams)):
        try:
            x = media.audio_track(sess.files[f"{p}_cam"], i, sess.work / f"{p}_cam_a{i}.wav")
        except RuntimeError:
            continue
        rms = float(np.sqrt((x ** 2).mean()))
        if rms > best_rms:
            best, best_rms = i, rms
    return best


STOP = {"name", "with", "that", "whose", "from", "contains", "naturally", "listed", "your", "this", "what",
        "which", "they", "their", "there", "letter"}
COUNTDOWN = 3.0     # prompt card appears ~3s before the clock starts
READ_DELAY = 0.8    # people start reading a prompt aloud ~0.5-1s after it appears


def _keywords(prompt: str) -> set[str]:
    toks = "".join(c.lower() if c.isalnum() else " " for c in prompt or "").split()
    return {t for t in toks if len(t) >= 4 and t not in STOP}


def prompt_read_offset(words: list[dict], questions: list[dict], log=print) -> tuple[float | None, list[dict]]:
    """Fallback anchor: players read each prompt aloud right after it appears.

    For every candidate offset, count the prompts whose keywords are spoken 0-4s after the card
    appears (in cam time); pick the offset that explains the most prompts, then take the median
    of their first-keyword times. Returns (offset, per-question evidence).
    """
    cands: list[tuple[int, float, int]] = []  # (question, appear-relative offset candidate, n keywords hit)
    norm_words = [("".join(c for c in w["word"].lower() if c.isalnum()), w["start"]) for w in words]
    for q in questions:
        kws = _keywords(q.get("prompt") or "")
        if not kws:
            continue
        appear = q["clock_start"] - COUNTDOWN
        for i, (wn, t) in enumerate(norm_words):
            if len(wn) >= 4 and any(fuzz.ratio(wn, k) >= 85 for k in kws):
                # how many distinct prompt keywords are said within 4s from here?
                hit = {k for wn2, t2 in norm_words[i:i + 25] if t2 - t <= 4.0
                       for k in kws if len(wn2) >= 4 and fuzz.ratio(wn2, k) >= 85}
                cands.append((q["n"], t - appear, len(hit)))
    if not cands:
        return None, []
    best, best_score = None, -1.0
    for off in np.arange(-300, 300, 0.1):
        explained: dict[int, float] = {}
        for n, c, k in cands:
            if off <= c <= off + 4.0:
                explained[n] = max(explained.get(n, 0), min(k, 3))
        score = sum(explained.values())
        if score > best_score:
            best, best_score = off, score
    ev = []
    firsts: dict[int, tuple[float, int]] = {}
    for n, c, k in cands:
        if best <= c <= best + 4.0 and (n not in firsts or c < firsts[n][0]):
            firsts[n] = (c, k)
    for n, (c, k) in sorted(firsts.items()):
        ev.append({"q": n, "delay_vs_offset": round(c - best, 2), "keywords_heard": k})
    if len(firsts) < 3:
        return None, ev
    off = float(np.median([c for c, _ in firsts.values()])) - READ_DELAY
    return round(off, 2), ev


def _answer_word_offsets(words: list[dict], questions: list[dict]) -> list[dict]:
    """Evidence for the fallback: when did the player *say* the answer they typed?"""
    out = []
    for q in questions:
        ans = (q.get("answer") or "").lower()
        toks = [t for t in "".join(c if c.isalnum() else " " for c in ans).split() if len(t) >= 4]
        if not toks or q.get("submit") is None:
            continue
        best = None
        for w in words:
            wn = "".join(c for c in w["word"].lower() if c.isalnum())
            if any(fuzz.ratio(wn, t) >= 85 for t in toks):
                off = w["start"] - q["submit"]
                if best is None or abs(off) < abs(best[1]):
                    best = (w["word"], off, w["start"])
        if best:
            out.append({"q": q["n"], "answer": q["answer"], "said": best[0], "cam_t": best[2],
                        "offset_if_said_at_submit": round(best[1], 2)})
    return out


def _analyze_screen(src: Path, debug_png: Path, work: Path, key: str, tiers: list[str], expect: int,
                    redetect: bool):
    """Crop + events for one screen. Cached in work/<session>/<key>_events.json."""
    cache = work / f"{key}_events.json"
    lines: list[str] = []
    if cache.exists() and not redetect:
        d = json.loads(cache.read_text(encoding="utf-8"))
        crop = screen.GameCrop(**{**d["crop"], "dive": tuple(d["crop"]["dive"]),
                                  "answer_band": tuple(d["crop"]["answer_band"])})
        ev = screen.ScreenEvents(crop, [screen.Question(**q) for q in d["questions"]], d["results_start"], d["duration"])
        return crop, ev, [f"  {key}: using cached {cache.name}"]
    crop = screen.detect_crop(src, debug_png, log=lines.append)
    ev, _ = screen.detect_events(src, crop, work, key, tiers, expect, log=lines.append)
    cache.write_text(json.dumps(ev.to_json(), indent=1), encoding="utf-8")
    return crop, ev, lines


def detect(sess: Session, cfg: dict, redetect: bool = False, log=print) -> dict:
    data: dict = {"session": sess.name, "sources": {}, "crop": {}, "cam_offset": {}, "events": {},
                  "cam_view": {}, "questions": [], "results": {}}
    for k, path in sess.files.items():
        i = ff.probe(path)
        data["sources"][k] = {"path": str(path), "duration": round(i.duration, 3), "width": i.width,
                              "height": i.height, "rotation": i.rotation, "audio_tracks": len(i.audio_streams)}

    # ---- game crop + events per screen (OCR is single-threaded, so run both screens in parallel)
    log(f"[sync] detecting game area, prompts and reveals on {len(PLAYERS)} screens (OCR at {screen.OCR_FPS} fps)...")
    with ProcessPoolExecutor(max_workers=len(PLAYERS)) as pool:
        futs = {p: pool.submit(_analyze_screen, sess.files[f"{p}_screen"], sess.out / f"debug_crop_{p}.png",
                               sess.work, f"{p}_screen", cfg["tiers"], cfg["questions"], redetect) for p in PLAYERS}
        for p in PLAYERS:
            crop, ev, lines = futs[p].result()
            for line in lines:
                log(line)
            data["crop"][p] = {"rect": list(crop.rect), "dive": [round(v, 1) for v in crop.dive]}
            data["events"][p] = {"questions": [asdict(q) for q in ev.questions], "results_start": ev.results_start}

    # ---- transcripts (needed for the sync fallback and for cutting)
    words = {p: transcribe(cam_wav(sess, cfg, p), sess.work / f"{p}_words.json", cfg, log=log) for p in PLAYERS}

    # ---- cam <-> screen offsets
    for p in PLAYERS:
        cam = media.audio_track(sess.files[f"{p}_cam"], cam_track(sess, cfg, p), cam_wav(sess, cfg, p))
        info = ff.probe(sess.files[f"{p}_screen"])
        entry: dict = {"confirmed": False}
        if info.audio_streams:
            scr = media.audio_track(sess.files[f"{p}_screen"], 0, sess.work / f"{p}_screen_a0.wav")
            off, conf = audiosync.xcorr_offset(cam, scr)
            checks = audiosync.windowed_check(cam, scr, off)
            consistent = all(abs(c - off) < 0.1 for c in checks)
            entry.update(seconds=round(off, 3), method="audio_xcorr", confidence=round(conf, 2), window_checks=checks)
            if conf >= CONF_OK and consistent:
                entry["confirmed"] = True
            log(f"[sync] {p}: cam_offset {off:+.3f}s via audio xcorr (confidence {conf:.1f}, windows {checks})")
            if not entry["confirmed"]:
                # the game's sound effects, found waveform-exact in the cam mic (day 4/5: the
                # envelope xcorr failed and the prompt-reading fallback was 1-3s off)
                s_off, s_ev = audiosync.sfx_offset(cam, scr)
                votes = [e for e in s_ev if e["agrees"]]
                entry["sfx_evidence"] = s_ev[:12]
                if s_off is not None:
                    entry.update(seconds=round(s_off, 3), method="sfx_match", confidence=len(votes),
                                 confirmed=len(votes) >= SFX_VOTES_OK)
                    log(f"[sync] {p}: cam_offset {s_off:+.3f}s via game sound effects heard by the cam "
                        f"({len(votes)} events agree: {[e['offset'] for e in votes][:8]})")
                else:
                    log(f"[sync] {p}: game sound effects not found in the cam audio")
        else:
            log(f"[sync] {p}: screen recording has no audio -> fallback")
        if not entry.get("confirmed") and entry.get("method") != "sfx_match":
            qs = data["events"][p]["questions"]
            reveals = [q["reveal"] for q in qs if q.get("reveal") is not None]
            pr_off, pr_ev = prompt_read_offset(words[p], qs)
            entry["prompt_read_evidence"] = pr_ev
            if pr_off is not None:
                # coarse: prompts read aloud; fine: loud reactions just after the reveals (+-1.5s)
                off, conf = pr_off, float(len(pr_ev))
                method = "prompt_read_fallback"
                if len(reveals) >= 3:
                    r_off, r_conf = audiosync.reaction_offset(cam, reveals, search=(pr_off - 1.5, pr_off + 1.5))
                    entry["reaction_refined"] = {"seconds": round(r_off, 3), "confidence": round(r_conf, 2)}
                    off, method = r_off, "prompt_read+reaction_fallback"
                entry.update(seconds=round(off, 3), method=method, confidence=round(conf, 2),
                             prompt_read_seconds=pr_off)
            elif len(reveals) >= 3:
                off, conf = audiosync.reaction_offset(cam, reveals)
                entry.update(seconds=round(off, 3), method="reaction_fallback", confidence=round(conf, 2))
            else:
                # last resort: first game event (first clock start) <-> first speech in the cam
                first_evt = qs[0]["clock_start"] if qs else 0.0
                first_word = words[p][0]["start"] if words[p] else 0.0
                entry.update(seconds=round(first_word - first_evt, 3), method="first_event_fallback", confidence=0)
            entry["answer_evidence"] = _answer_word_offsets(words[p], qs)
            log(f"[sync] {p}: cam_offset {entry['seconds']:+.3f}s via {entry['method']} "
                f"(confidence {entry['confidence']}) - NEEDS CONFIRMATION")
            for ev_ in entry.get("prompt_read_evidence", []):
                log(f"         Q{ev_['q']} prompt read aloud {ev_['delay_vs_offset']:+.1f}s after it appeared "
                    f"({ev_['keywords_heard']} keywords)")
            for ev_ in entry["answer_evidence"]:
                log(f"         said {ev_['said']!r} at cam {ev_['cam_t']:.1f}s for Q{ev_['q']} "
                    f"(=> offset {ev_['offset_if_said_at_submit']:+.2f}s if said exactly at submit)")
        data["cam_offset"][p] = entry

    # ---- pair questions across players
    qs = {p: data["events"][p]["questions"] for p in PLAYERS}
    for i in range(max(len(v) for v in qs.values())):
        es = {p: (qs[p][i] if i < len(qs[p]) else None) for p in PLAYERS}
        a = next((e for e in es.values() if e), {})
        sims = [fuzz.token_set_ratio(a.get("prompt") or "", e.get("prompt") or "") for e in es.values() if e]
        data["questions"].append({"n": i + 1, "prompt": a.get("prompt"),
                                  "prompt_match": round(min(sims)) if len(sims) == len(PLAYERS) else 0, **es})
    data["results"] = {p: data["events"][p]["results_start"] for p in PLAYERS}

    # ---- cam framing (faces)
    for p in PLAYERS:
        cache = sess.work / f"{p}_faces.json"
        if cache.exists() and not redetect:
            data["cam_view"][p] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        log(f"[sync] {p}: finding faces in {sess.files[f'{p}_cam'].name}")
        data["cam_view"][p] = faces.frame_people(sess.files[f"{p}_cam"], sess.work, sess.out / f"debug_faces_{p}.png",
                                                 log=log)
        cache.write_text(json.dumps(data["cam_view"][p]), encoding="utf-8")
    return data


def problems(data: dict, cfg: dict) -> list[str]:
    """Things a human must look at before rendering (the tool never guesses these)."""
    out = []
    for p in PLAYERS:
        o = data["cam_offset"][p]
        if not o.get("confirmed"):
            out.append(f"{p} cam offset {o['seconds']:+.3f}s came from {o['method']} - check the preview, then set "
                       f"cam_offset.{p}.confirmed: true in sync.json (or edit seconds)")
        n = len(data["events"][p]["questions"])
        if n != cfg["questions"]:
            out.append(f"{p}: found {n} prompts, expected {cfg['questions']} - see work/<session>/{p}_screen_ocr/")
    for q in data["questions"]:
        for p in PLAYERS:
            e = q.get(p)
            if e is None:
                out.append(f"Q{q['n']}: missing for {p}")
                continue
            if e.get("reveal") is None:
                out.append(f"Q{q['n']} {p}: no reveal found ({'; '.join(e.get('notes') or [])})")
            elif e.get("tier") is None:
                out.append(f"Q{q['n']} {p}: tier not recognised - set questions[{q['n'] - 1}].{p}.tier in sync.json")
        if all(q.get(p) for p in PLAYERS) and q["prompt_match"] < 60:
            out.append(f"Q{q['n']}: prompts differ between players ("
                       + " vs ".join(repr(q[p].get('prompt')) for p in PLAYERS) + ")")
    return out


def load_or_detect(sess: Session, cfg: dict, redetect: bool = False, log=print) -> dict:
    if sess.sync_json.exists() and not redetect:
        log(f"[sync] using {sess.sync_json} (pass --redetect to re-run detection)")
        return json.loads(sess.sync_json.read_text(encoding="utf-8"))
    data = detect(sess, cfg, redetect=redetect, log=log)
    sess.sync_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log(f"[sync] wrote {sess.sync_json}")
    return data
