"""
audio_gen.py — Audio generation with word-level timestamps.

PRIMARY PATH: Edge TTS stream() captures WordBoundary events → exact timestamps.
FALLBACK 2:   Kokoro local TTS (no timestamps → estimated from duration).
"""

import os
import asyncio
import re
import random
from datetime import datetime
from config import OUTPUT_DIR, BASE_DIR
import imageio_ffmpeg
from pydub import AudioSegment
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# F5-TTS Imports (Local Voice Cloning)
import torch
import soundfile as sf
from f5_tts.api import F5TTS

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

def trim_audio_silence(path, word_timestamps):
    """
    Trims silence from the start and end of the audio file 
    and shifts all word timestamps so that the first word starts at 0.0s.
    """
    from pydub import AudioSegment
    from pydub.silence import detect_leading_silence

    audio = AudioSegment.from_file(path)
    
    # Detect start silence (using a conservative -50dBFS threshold)
    start_trim = detect_leading_silence(audio, silence_threshold=-50.0)
    # Detect end silence
    reversed_audio = audio.reverse()
    end_trim = detect_leading_silence(reversed_audio, silence_threshold=-50.0)

    duration = len(audio)
    trimmed_audio = audio[start_trim:duration-end_trim]
    
    # Boost volume by 4 decibels for better clarity
    trimmed_audio = trimmed_audio + 4
    
    trimmed_audio.export(path, format="wav" if path.endswith(".wav") else "mp3")
    
    # Recalibrate timestamps (shift by start_trim in seconds)
    shift_sec = start_trim / 1000.0
    new_ts = []
    for ws in word_timestamps:
        new_ts.append({
            "word": ws["word"],
            "start": max(0.0, round(ws["start"] - shift_sec, 3)),
            "end": max(0.0, round(ws["end"] - shift_sec, 3))
        })
    
    new_dur = len(trimmed_audio) / 1000.0
    print(f"Audio trimmed: -{shift_sec:.2f}s from start. New duration: {new_dur:.2f}s")
    return new_dur, new_ts
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
KOKORO_MODEL = os.path.join(ASSETS_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(ASSETS_DIR, "voices-v1.0.bin")

# ── F5-TTS Config ─────────────────────────────────────────────────────────────
VJ_REF_WAV = os.path.join(ASSETS_DIR, "vj_voice.wav")
VJ_REF_TEXT = (
    "US President Donald Trump has demanded Iran's unconditional surrender as the American and Israeli military continued to launch strikes."
)

_f5_instance = None

def _get_f5_model():
    global _f5_instance
    if _f5_instance is None:
        print("Initialising F5-TTS (Local Voice Cloning)...")
        _f5_instance = F5TTS()
    return _f5_instance

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
# PRIMARY: F5-TTS (0 Rs / Local Voice Cloning)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_f5_clone(text, output_path):
    print(f"F5-TTS → Cloning VJ's Voice (0 Rs)...")
    
    # Run inference via the high-level API
    wav_path = output_path.replace(".mp3", ".wav")
    
    # Pre-splitting into smaller chunks to avoid the "Audio over 12s clipping" warning
    # We split by common sentence delimiters but keep chunks under ~200 chars
    import re
    sentences = re.split(r'([.?!]+)', text)
    chunks = []
    current = ""
    for i in range(0, len(sentences), 2):
        s = sentences[i]
        p = sentences[i+1] if i+1 < len(sentences) else ""
        if len(current + s + p) < 180:
            current += s + p
        else:
            if current: chunks.append(current.strip())
            current = s + p
    if current: chunks.append(current.strip())
    
    # Generate
    f5 = _get_f5_model()
    segment_paths = []
    for i, chunk in enumerate(chunks):
        seg_path = wav_path.replace(".wav", f"_seg_{i}.wav")
        f5.infer(
            ref_file=VJ_REF_WAV,
            ref_text=VJ_REF_TEXT,
            gen_text=chunk,
            file_wave=seg_path,
            speed=0.95
        )
        segment_paths.append(seg_path)
        
    # Combine segments
    combined = AudioSegment.empty()
    for sp in segment_paths:
        combined += AudioSegment.from_wav(sp)
        try: os.remove(sp) # Clean up partial segments
        except: pass
        
    combined.export(wav_path, format="wav")
    duration = get_audio_duration(wav_path)
    
    # Word timestamps via stable-ts
    word_timestamps = _apply_stable_ts(wav_path, text)
    if not word_timestamps:
        word_timestamps = _estimate_timestamps(text, duration)
        
    print(f"F5-TTS done: {duration:.2f}s | {len(word_timestamps)} word timestamps")
    return wav_path, duration, word_timestamps


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY: Kokoro (Local Open Source / 100% Free)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_kokoro(text, output_path):
    import soundfile as sf
    from kokoro_onnx import Kokoro

    if not os.path.exists(KOKORO_MODEL) or not os.path.exists(KOKORO_VOICES):
        raise FileNotFoundError("Kokoro ONNX model or voice bin missing from assets folder.")

    print("Kokoro TTS → am_echo (Local CPU)...")
    kokoro  = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    
    # 2026 Monetization Strategy: Unique Pitching/Speed
    import random
    unique_speed = 1.1 + random.uniform(-0.05, 0.05)
    
    samples, sr = kokoro.create(text, voice="am_echo", speed=unique_speed, lang="en-us")
    wav_path = output_path.replace(".mp3", ".wav")
    sf.write(wav_path, samples, sr)
    duration = len(samples) / sr
    
    # Run stable-ts to get actual timestamps from the generated audio
    word_timestamps = _apply_stable_ts(wav_path, text)
    if not word_timestamps:
        word_timestamps = _estimate_timestamps(text, duration)
        
    print(f"Kokoro done: {duration:.2f}s | {len(word_timestamps)} word timestamps")
    return wav_path, duration, word_timestamps


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def clean_tts_text(text):
    """
    Strips out AI meta-instructions like [pause], (silence), [intense music] etc.
    while keeping the punctuation like ... and -- which creates the actual pause.
    """
    if not text: return ""
    # Remove anything inside brackets [bra-ckets] or (paren-theses) if they contain 'pause', 'silence', 'music', 'sound'
    # This specifically target instructions but keeps normal parentheses if they don't look like instructions.
    cleaned = re.sub(r'\[[^\]]*(pause|silence|music|sound|breath)[^\]]*\]', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\([^)]*(pause|silence|music|sound|breath)[^)]*\)', '', cleaned, flags=re.IGNORECASE)
    
    # Also remove literal instances of " [pause] " or " (pause) " if for some reason they exist outside brackets
    cleaned = re.sub(r'\s*\[pause\]\s*', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(pause\)\s*', ' ', cleaned, flags=re.IGNORECASE)
    
    # Prevent TTS from saying 'strike', 'asterisk', or 'dash'
    cleaned = cleaned.replace("—", "...") # Em-dash -> pause
    cleaned = cleaned.replace("–", "...") # En-dash -> pause
    cleaned = cleaned.replace("--", "...") # Double hyphen -> pause
    cleaned = cleaned.replace("*", "")    # Remove asterisks
    cleaned = cleaned.replace("~", "")    # Remove tildes
    
    # Clean up double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def generate_voiceover(text, voice_request="en-US-GuyNeural", emotion="excited"):
    """
    Returns: (audio_path, duration, word_timestamps)
    word_timestamps: [{"word": str, "start": float, "end": float}, ...]
    """
    text = clean_tts_text(text)
    today     = datetime.now().strftime("%Y-%m-%d")
    mp3_path  = os.path.join(OUTPUT_DIR, f"audio_{today}.mp3")
    
    path, dur, word_timestamps = None, 0, []
    
    # 1. PRIMARY: F5-TTS Local Voice Cloning (Your Voice)
    try:
        path, dur, word_timestamps = _generate_f5_clone(text, mp3_path)
    except Exception as e:
        print(f"⚠️ F5-TTS Cloning failed: {e}")
        # 2. SECONDARY: Kokoro TTS
        try:
            path, dur, word_timestamps = _generate_kokoro(text, mp3_path)
        except Exception as e2:
            print(f"⚠️ Kokoro failed: {e2}")
            # FALLBACK: Edge TTS
            try:
                path, dur, word_timestamps = _generate_edge_tts(text, LOCKED_VOICE, mp3_path)
                real_ts = _apply_stable_ts(path, text)
                if real_ts: word_timestamps = real_ts
            except Exception as e3:
                print(f"Edge TTS fallback failed: {e3}")

    # Post-process: Trim dead air at start/end to ensure 0-gap hook
    if path and word_timestamps:
        dur, word_timestamps = trim_audio_silence(path, word_timestamps)

    return path, dur, word_timestamps
