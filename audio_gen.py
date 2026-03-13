"""
PRIMARY PATH: F5-TTS Local Voice Cloning (Your Voice).
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

# F5-TTS Imports are delayed to _get_f5_model to prevent CI crashes
import soundfile as sf


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
            # Explicitly cleanup model to free VRAM
            del model
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
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
    
    # Boost volume by 8 decibels for better clarity (increased for presence)
    trimmed_audio = trimmed_audio + 8
    
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
# F5-TTS Paths
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ── F5-TTS Config ─────────────────────────────────────────────────────────────
VJ_REF_WAV = os.path.join(ASSETS_DIR, "vj_voice_new.wav")
VJ_REF_TEXT = (
    "A student of Punjab University fell off the fourth floor of a paying-guest accommodation in 2013 and suffered a serious injuries. He was put on a life support since then he has been confined to a bed."
)

_f5_instance = None

def _get_f5_model():
    global _f5_instance
    if _f5_instance is None:
        import torch
        from f5_tts.api import F5TTS

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Initialising F5-TTS (Local Voice Cloning) on {device}...")
        _f5_instance = F5TTS(device=device)
    return _f5_instance

def unload_f5_model():
    """Explicitly unload F5-TTS model from GPU to free up memory."""
    global _f5_instance
    if _f5_instance is not None:
        import torch
        import gc
        print("🧹 Unloading F5-TTS model and clearing CUDA cache...")
        _f5_instance = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# ── Global Phonetic Dictionary (Source of Truth for Audio & Subtitles) ────────
# Comprehensive dictionary in separate module (350+ words + g2p_en auto-detection)
from phonetic_dict import PHONETIC_DICT, auto_detect_hard_words


# Nanosecond conversion removed.


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
# PRIMARY: F5-TTS (0 Rs / Local Voice Cloning) — High-Quality Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _smart_split_sentences(text, max_chars=120):
    """
    Split text into natural sentence-boundary chunks for F5-TTS.
    Keeps chunks under max_chars to prevent the 12s clipping issue.
    Splits at sentence boundaries (.?!) then at clause boundaries (,;:—) as fallback.
    """
    import re
    # First split into sentences
    parts = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for part in parts:
        if len(current + " " + part) < max_chars:
            current = (current + " " + part).strip()
        else:
            if current:
                chunks.append(current.strip())
            # If a single sentence is still too long, split at clause boundaries
            if len(part) > max_chars:
                clause_parts = re.split(r'(?<=[,;:\—])\s+', part)
                sub_current = ""
                for cp in clause_parts:
                    if len(sub_current + " " + cp) < max_chars:
                        sub_current = (sub_current + " " + cp).strip()
                    else:
                        if sub_current:
                            chunks.append(sub_current.strip())
                        sub_current = cp
                current = sub_current
            else:
                current = part
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _postprocess_voice_audio(wav_path):
    """
    Post-processing chain to polish F5-TTS output:
    1. Gentle high-pass filter at 80Hz to remove low-frequency rumble
    2. Normalize to -1dB for consistent loudness
    3. Add 2ms fade-in/out to prevent click artifacts
    """
    try:
        audio = AudioSegment.from_wav(wav_path)
        
        # 1. High-pass filter (remove rumble below 80Hz)
        # Simple implementation: apply a 2-pass moving average subtraction
        import numpy as np
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        sr = audio.frame_rate
        channels = audio.channels
        
        # Rolling average with window ~ 1/80Hz = 12.5ms
        window = max(1, int(sr * 0.0125))
        if len(samples) > window * 2:
            # Subtract the low-frequency component
            low_freq = np.convolve(samples, np.ones(window)/window, mode='same')
            samples = samples - low_freq * 0.7  # Subtract 70% of rumble (gentle)
        
        # 2. Normalize to -1dB peak
        peak = np.max(np.abs(samples))
        if peak > 0:
            target = 10 ** (-1.0 / 20) * 32767  # -1dB in 16-bit
            samples = samples * (target / peak)
        
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        
        # Reconstruct AudioSegment
        processed = AudioSegment(
            samples.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=channels
        )
        
        # 3. Fade in/out to prevent clicks
        processed = processed.fade_in(2).fade_out(2)
        
        processed.export(wav_path, format="wav")
        print(f"   🎙️ Audio post-processed: high-pass 80Hz, normalized to -1dB")
    except Exception as e:
        print(f"   ⚠ Audio post-processing skipped: {e}")


def _generate_f5_clone(text, output_path):
    print(f"F5-TTS → Cloning VJ's Voice (High-Quality Pipeline)...")
    
    wav_path = output_path.replace(".mp3", ".wav")
    
    # Smart sentence-boundary splitting (120 chars max per chunk)
    chunks = _smart_split_sentences(text, max_chars=120)
    print(f"   Split into {len(chunks)} voice segments")
    
    # Generate each segment with F5-TTS
    f5 = _get_f5_model()
    segment_paths = []
    for i, chunk in enumerate(chunks):
        seg_path = wav_path.replace(".wav", f"_seg_{i}.wav")
        f5.infer(
            ref_file=VJ_REF_WAV,
            ref_text=VJ_REF_TEXT,
            gen_text=chunk,
            file_wave=seg_path,
            speed=1.0  # Natural speed for best quality (enforcement handled by script length)
        )
        segment_paths.append(seg_path)
        
    # Combine segments with 30ms cross-fade for seamless joins
    CROSSFADE_MS = 30
    combined = AudioSegment.from_wav(segment_paths[0]) if segment_paths else AudioSegment.empty()
    for sp in segment_paths[1:]:
        seg = AudioSegment.from_wav(sp)
        combined = combined.append(seg, crossfade=CROSSFADE_MS)
    
    # Clean up segment files
    for sp in segment_paths:
        try: os.remove(sp)
        except: pass
        
    combined.export(wav_path, format="wav")
    
    # Post-process for professional voice quality
    _postprocess_voice_audio(wav_path)
    
    duration = get_audio_duration(wav_path)
    
    # Word timestamps via stable-ts
    word_timestamps = _apply_stable_ts(wav_path, text)
    if not word_timestamps:
        word_timestamps = _estimate_timestamps(text, duration)
        
    print(f"F5-TTS done: {duration:.2f}s | {len(word_timestamps)} word timestamps")
    return wav_path, duration, word_timestamps


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: Edge TTS (Cloud-based, very reliable)
# ─────────────────────────────────────────────────────────────────────────────
async def _async_generate_edge_tts(text, output_path):
    import edge_tts
    # High-quality voice for tech/research content
    VOICE = "en-US-AndrewNeural" 
    communicate = edge_tts.Communicate(text, VOICE, rate="+5%")
    await communicate.save(output_path)

def _generate_edge_tts(text, output_path):
    print(f"📡 Falling back to Edge TTS (Cloud)...")
    try:
        asyncio.run(_async_generate_edge_tts(text, output_path))
        duration = get_audio_duration(output_path)
        # For Edge TTS, estimate timestamps as stable-ts might be overkill/slow on cloud output
        # unless user strictly wants real timestamps. Let's try stable-ts first.
        word_timestamps = _apply_stable_ts(output_path, text)
        if not word_timestamps:
            word_timestamps = _estimate_timestamps(text, duration)
        return output_path, duration, word_timestamps
    except Exception as e:
        print(f"❌ Edge TTS also failed: {e}")
        return None, 0, []


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def clean_tts_text(text, phonetic=True, custom_phonetic_map=None):
    """
    Strips out AI meta-instructions and fixes pronunciation issues.
    If phonetic=True, it replaces difficult words with phonetic spellings.
    """
    if not text: return ""
    
    # 1. Broadly remove ALL bracketed and parenthesized meta-instructions/timestamps
    # This prevents the TTS from speaking things like [0.0], [14.0], (pause), [glitch], etc.
    cleaned = re.sub(r'\[[^\]]*\]', ' ', text)
    cleaned = re.sub(r'\([^)]*\)', ' ', cleaned)
    
    # 2. Fix pronunciation artifacts (The "Strike" issue)
    cleaned = cleaned.replace("—", "...") # Em-dash
    cleaned = cleaned.replace("–", "...") # En-dash
    cleaned = cleaned.replace("--", "...") # Double hyphen
    cleaned = cleaned.replace("*", " ")    # Asterisks
    cleaned = cleaned.replace("•", " ")    # Bullet point
    cleaned = cleaned.replace("·", " ")    # Middle dot
    cleaned = cleaned.replace("⁃", " ")    # Hyphen bullet
    cleaned = cleaned.replace("●", " ")    # Circle bullet
    cleaned = cleaned.replace("▪", " ")    # Square bullet
    cleaned = cleaned.replace("~", " ")    # Tilde
    
    # Standalone hyphens
    cleaned = re.sub(r'\n\s*-\s*', '\n ', cleaned)
    cleaned = re.sub(r'\s+-\s+', ' ... ', cleaned)
    
    # 3. Phonetic Cleanups for Clarity
    if phonetic:
        # Merge global dictionary and custom Gemini map
        full_map = PHONETIC_DICT.copy()
        if custom_phonetic_map:
            full_map.update(custom_phonetic_map)

        for word, replacement in full_map.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # 4. Auto-detect remaining hard words via g2p_en (neural G2P fallback)
        try:
            auto_corrections = auto_detect_hard_words(cleaned)
            for word, respelling in auto_corrections.items():
                pattern = r'\b' + re.escape(word) + r'\b'
                cleaned = re.sub(pattern, respelling, cleaned, flags=re.IGNORECASE)
        except Exception as e:
            print(f"g2p_en auto-detection skipped: {e}")
    
    # Clean up double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def restore_original_words(word_timestamps, original_text, custom_phonetic_map=None):
    """
    Matches the phonetically spoke words back to the original script words 
    to ensure subtitles look professional.
    """
    if not word_timestamps or not original_text:
        return word_timestamps
        
    original_words = original_text.split()
    
    # Subtitle Restoration Map (Key: Spoken-Normalized -> Value: Display-Original)
    # Automatically build from global and custom dictionaries
    restore_map = {}
    
    # helper to normalize a phonetic string for comparison
    def norm_key(s): return re.sub(r'[^\w]', '', s.upper())
    
    # 1. Add global constants
    for orig, phonetic in PHONETIC_DICT.items():
        restore_map[norm_key(phonetic)] = orig
            
    # 2. Add custom Gemini map
    if custom_phonetic_map:
        for orig, phonetic in custom_phonetic_map.items():
            restore_map[norm_key(phonetic)] = orig

    for i, wt in enumerate(word_timestamps):
        spoken_clean = norm_key(wt["word"])
        if spoken_clean in restore_map:
            wt["word"] = restore_map[spoken_clean]
        else:
            # Fallback: check if it matches the original word directly
            if i < len(original_words):
                orig_clean = norm_key(original_words[i])
                if spoken_clean == orig_clean:
                    wt["word"] = original_words[i]
                        
    return word_timestamps

def generate_voiceover(text, custom_phonetic_map=None):
    """
    Returns: (audio_path, duration, word_timestamps)
    """
    original_raw_text = text
    text_to_speak = clean_tts_text(text, phonetic=True, custom_phonetic_map=custom_phonetic_map)
    
    today     = datetime.now().strftime("%Y-%m-%d")
    mp3_path  = os.path.join(OUTPUT_DIR, f"audio_{today}.mp3")
    
    path, dur, word_timestamps = None, 0, []
    
    # ── ENGINE PRIORITY ──────────────────────────────────────────────────────
    # 1. PRIMARY: F5-TTS Local Voice Cloning (Your Voice) - ONLY ENGINE
    try:
        path, dur, word_timestamps = _generate_f5_clone(text_to_speak, mp3_path)
    except Exception as e:
        print(f"❌ F5-TTS Cloning failed (possibly NumPy/VRAM): {e}")
        # 2. FALLBACK: Edge TTS
        path, dur, word_timestamps = _generate_edge_tts(text_to_speak, mp3_path)

    # Post-process: Restore original word spellings for subtitles
    if word_timestamps:
        word_timestamps = restore_original_words(word_timestamps, original_raw_text, custom_phonetic_map=custom_phonetic_map)

    # Post-process: Trim dead air at start/end
    if path and word_timestamps:
        dur, word_timestamps = trim_audio_silence(path, word_timestamps)

    return path, dur, word_timestamps
