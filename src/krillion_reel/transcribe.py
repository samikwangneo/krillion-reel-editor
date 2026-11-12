"""WhisperX word-level transcription of the cam mics (cached JSON)."""
from __future__ import annotations

import json
from pathlib import Path


def transcribe(wav16k: Path, cache: Path, cfg: dict, log=print) -> list[dict]:
    """Return [{word, start, end, score}] in cam time (seconds)."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import whisperx

    wc = cfg["whisper"]
    log(f"  whisperx {wc['model']} ({wc['device']}/{wc['compute_type']}) on {wav16k.name} ...")
    audio = whisperx.load_audio(str(wav16k))
    model = whisperx.load_model(wc["model"], wc["device"], compute_type=wc["compute_type"],
                                language=wc.get("language", "en"))
    result = model.transcribe(audio, batch_size=8, language=wc.get("language", "en"))
    align_model, meta = whisperx.load_align_model(language_code=result.get("language", "en"), device=wc["device"])
    aligned = whisperx.align(result["segments"], align_model, meta, audio, wc["device"],
                             return_char_alignments=False)
    words = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({"word": w["word"], "start": round(float(w["start"]), 3),
                              "end": round(float(w["end"]), 3), "score": round(float(w.get("score", 0)), 3)})
    cache.write_text(json.dumps(words, indent=1), encoding="utf-8")
    log(f"  {len(words)} words")
    return words
