"""
audio_gen.py — Audio generation with word-level timestamps.

PRIMARY PATH: Edge TTS stream() captures WordBoundary events → exact timestamps.
FALLBACK 2:   Kokoro local TTS (no timestamps → estimated from duration).
"""

import os
import asyncio
import re
from datetime import datetime
from config import OUTPUT_DIR

def _apply_stable_ts(audio_path, text):
    try:
        import stable_whisper
        import warnings
        warnings.filterwarnings("ignore")
        print("Running stable-ts to extract REAL word timestamps...")
        model = stable_whisper.load_model('base')
        result = model.align(audio_path, text, language='en')
        word_timestamps = []
        for segment in result.segments:
            for word in segment.words:
                clean_word = word.word.strip()
                if clean_word:
                    word_timestamps.append({
                        "word": clean_word,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3)
                    })
        if word_timestamps:
            print(f"stable-ts extracted {len(word_timestamps)} real word timestamps.")
            return word_timestamps
    except Exception as e:
        print(f"stable-ts unavailable or failed: {e}")
    return None
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
KOKORO_MODEL = os.path.join(ASSETS_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(ASSETS_DIR, "voices-v1.0.bin")

# Edge TTS offset is in 100-nanosecond units → divide by 10_000_000 for seconds
_NS100_PER_SEC = 10_000_000


def _estimate_timestamps(text, duration):
    """Evenly distribute timestamps when real ones are unavailable."""
    words = text.split()
    if not words:
        return []
    interval = duration / len(words)
    return [
        {"word": w, "start": round(i * interval, 3), "end": round((i + 1) * interval, 3)}
        for i, w in enumerate(words)
    ]


def get_audio_duration(path):
    try:
        from mutagen.mp3 import MP3
        return MP3(path).info.length
    except Exception:
        try:
            import soundfile as sf
            data, sr = sf.read(path)
            return len(data) / sr
        except Exception:
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY: Edge TTS — stream() gives us word boundary events (exact timestamps)
# ─────────────────────────────────────────────────────────────────────────────
async def _edge_tts_stream(text, voice, output_path):
    import edge_tts
    # Decreasing the rate to match the slow and deliberate pace
    communicate = edge_tts.Communicate(text, voice, rate="-10%")
    sentence_events = []
    audio_data = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
        elif chunk["type"] in ("SentenceBoundary", "WordBoundary"):
            sentence_events.append({
                "text":  chunk.get("text", ""),
                "start": chunk["offset"] / _NS100_PER_SEC,
                "dur":   chunk["duration"] / _NS100_PER_SEC,
            })

    with open(output_path, "wb") as f:
        f.write(bytes(audio_data))

    if not sentence_events:
        return []  # fallback handles this

    # Distribute word timestamps within each sentence proportionally by char count
    word_timestamps = []
    for evt in sentence_events:
        sent_start = evt["start"]
        sent_dur   = evt["dur"]
        sent_text  = evt["text"].strip()
        words      = sent_text.split()
        if not words:
            continue

        total_chars = sum(len(w) for w in words)
        cursor = sent_start
        for w in words:
            fraction = len(w) / max(total_chars, 1)
            w_dur    = fraction * sent_dur
            word_timestamps.append({
                "word":  w,
                "start": round(cursor, 3),
                "end":   round(cursor + w_dur, 3),
            })
            cursor += w_dur

    return word_timestamps


LOCKED_VOICE = "en-US-AndrewNeural"

def _generate_edge_tts(text, voice, output_path):
    voice = LOCKED_VOICE  # Always use AndrewNeural — warm, confident, authentic
    print(f"Edge TTS → {voice} (with word timestamps)...")
    word_timestamps = asyncio.run(_edge_tts_stream(text, voice, output_path))
    duration = get_audio_duration(output_path)
    print(f"Edge TTS done: {duration:.2f}s | {len(word_timestamps)} word timestamps")
    return output_path, duration, word_timestamps


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK 1: Kokoro local TTS (no real timestamps)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_kokoro(text, output_path):
    import soundfile as sf
    from kokoro_onnx import Kokoro

    print("Kokoro TTS → am_echo...")
    kokoro  = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    samples, sr = kokoro.create(text, voice="am_echo", speed=1.1, lang="en-us")
    wav_path = output_path.replace(".mp3", ".wav")
    sf.write(wav_path, samples, sr)
    duration = len(samples) / sr
    word_timestamps = _estimate_timestamps(text, duration)
    print(f"Kokoro done: {duration:.2f}s (estimated timestamps)")
    return wav_path, duration, word_timestamps


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def generate_voiceover(text, voice_request="en-US-GuyNeural", emotion="excited"):
    """
    Returns: (audio_path, duration, word_timestamps)
    word_timestamps: [{"word": str, "start": float, "end": float}, ...]
    """
    today     = datetime.now().strftime("%Y-%m-%d")
    mp3_path  = os.path.join(OUTPUT_DIR, f"audio_{today}.mp3")
    # Edge TTS — FALLBACK 1
    try:
        path, dur, word_timestamps = _generate_edge_tts(text, LOCKED_VOICE, mp3_path)
        real_ts = _apply_stable_ts(path, text)
        if real_ts:
            word_timestamps = real_ts
        print("⚠️ Falling back to Edge TTS (American accent)")
        return path, dur, word_timestamps
    except Exception as e:
        print(f"Edge TTS failed: {e}")

    # Kokoro — FALLBACK 2
    if os.path.exists(KOKORO_MODEL):
        try:
            path, dur, word_timestamps = _generate_kokoro(text, mp3_path)
            real_ts = _apply_stable_ts(path, text)
            if real_ts:
                word_timestamps = real_ts
            return path, dur, word_timestamps
        except Exception as e:
            print(f"Kokoro failed: {e}")

    print("All TTS engines failed.")
    return None, 0, []
