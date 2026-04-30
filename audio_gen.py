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
    
    # Detect start silence (using a very aggressive -60dBFS threshold for zero latency)
    start_trim = detect_leading_silence(audio, silence_threshold=-60.0)
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
VJ_REF_WAV = os.path.join(ASSETS_DIR, "vj.wav")
VJ_REF_TEXT = (
    "Welcome you are listening to your channel, we bring you the best insights, ideas and stories. Drafted just for you Stay tuned and let's get started."
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
    Professional post-processing chain to enhance clarity and presence:
    1. High-pass filter at 100Hz to remove low-end "mud" and rumble.
    2. Dynamic Range Compression to level out the voice and make it "pop".
    3. Final normalization to -1dB for consistent loudness.
    4. Subtle fade-in/out to prevent clicks.
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize, compress_dynamic_range
        
        audio = AudioSegment.from_wav(wav_path)
        
        # 1. High-pass filter (100Hz) - Removes low-frequency energy that masks speech clarity
        audio = audio.high_pass_filter(100)
        
        # 2. Dynamic Compression - Makes the voice sound more authoritative
        # Threshold -20dB, Ratio 3:1, Attack 5ms, Release 50ms
        audio = compress_dynamic_range(audio, threshold=-20.0, ratio=3.0, attack=5.0, release=50.0)
        
        # 3. Final Normalization
        audio = normalize(audio, headroom=1.0)
        
        # 4. Prevent click artifacts
        audio = audio.fade_in(5).fade_out(5)
        
        audio.export(wav_path, format="wav")
        print(f"   🎙️ Audio enhanced: 100Hz HPF, dynamic compression, normalized to -1dB")
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
            nfe_step=64,      # Increased from 32 for higher audio fidelity and clarity
            remove_silence=True, # Cleanup of chunk edges
            speed=1.0
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
    else:
        # 3b. Non-phonetic (Subtitle) Spell Correction
        # This handles cases where Gemini mispells words that should be caught
        corrections = {
            "parallesim": "parallelism",
            "paralllesim": "parallelism",
            "technolgies": "technologies",
            "breakthru": "breakthrough",
            "golden ore": "golden hour",
            "LLT mode dull": "multimodal",
            "Pew": "Popsa"
        }
        for wrong, right in corrections.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            cleaned = re.sub(pattern, right, cleaned, flags=re.IGNORECASE)
    
    # Clean up double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def restore_original_words(word_timestamps, original_text, custom_phonetic_map=None):
    """
    Matches the phonetically spoken words back to the original script words 
    to ensure subtitles look professional (reserving case and punctuation).
    """
    if not word_timestamps or not original_text:
        return word_timestamps
        
    # 1. Prepare display tokens (original words that are NOT meta-instructions)
    raw_tokens = original_text.split()
    display_tokens = []
    for t in raw_tokens:
        # Check if the token is JUST a meta-instruction like [1.0] or (pause)
        if not re.fullmatch(r'\[[^\]]*\]|\([^)]*\)', t):
            display_tokens.append(t)
    
    # 2. Build canonical map from dictionaries
    # (Mapping Spoken-Normalized-Phonetic -> Canonical-Original-Word)
    phonetic_to_canonical = {}
    def norm_key(s): return re.sub(r'[^\w]', '', s.upper()) if s else ""
    
    for orig, phonetic in PHONETIC_DICT.items():
        phonetic_to_canonical[norm_key(phonetic)] = orig
            
    if custom_phonetic_map:
        for orig, phonetic in custom_phonetic_map.items():
            phonetic_to_canonical[norm_key(phonetic)] = orig

    # 3. Synchronized Alignment
    orig_idx = 0
    new_timestamps = []
    
    for wt in word_timestamps:
        spoken_clean = norm_key(wt["word"])
        if not spoken_clean:
            new_timestamps.append(wt)
            continue
            
        found_match = False
        # Lookahead 10 words to stay robust against script/audio drift
        for j in range(orig_idx, min(orig_idx + 10, len(display_tokens))):
            target_token = display_tokens[j]
            target_clean = norm_key(target_token)
            
            # Case A: Spoken word is a phonetic respelling of this target
            # e.g. Spoken: "SYMULTAYNEEUSLEE" maps to "SIMULTANEOUSLY" (canonical)
            # which matches "Simultaneously," (target_clean = "SIMULTANEOUSLY")
            if spoken_clean in phonetic_to_canonical:
                canonical_word = phonetic_to_canonical[spoken_clean]
                if norm_key(canonical_word) == target_clean:
                    wt["word"] = target_token
                    orig_idx = j + 1
                    found_match = True
                    break
            
            # Case B: Direct match (no respelling or model "heard" through it)
            if spoken_clean == target_clean:
                wt["word"] = target_token
                orig_idx = j + 1
                found_match = True
                break
                
        new_timestamps.append(wt)
                        
    return new_timestamps

def _generate_elevenlabs(text, output_path):
    print(f"📡 Using ElevenLabs Turbo v2.5 (High Quality Fallback)...")
    from config import ELEVENLABS_API_KEY
    if not ELEVENLABS_API_KEY:
        print("   ✗ ElevenLabs API Key missing.")
        return None, 0, []
    
    try:
        import requests
        # Voice ID for VJ-like tech persona
        VOICE_ID = "EXAVITQu4vr4xnSDxMaL" # Bella or another high-quality Turbo voice
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            duration = get_audio_duration(output_path)
            word_timestamps = _apply_stable_ts(output_path, text)
            if not word_timestamps:
                word_timestamps = _estimate_timestamps(text, duration)
            return output_path, duration, word_timestamps
        else:
            print(f"   ✗ ElevenLabs API error: {response.text}")
            return None, 0, []
    except Exception as e:
        print(f"   ✗ ElevenLabs failed: {e}")
        return None, 0, []

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
    # 1. PRIMARY: F5-TTS Local Voice Cloning
    try:
        path, dur, word_timestamps = _generate_f5_clone(text_to_speak, mp3_path)
    except Exception as e:
        print(f"❌ F5-TTS failed: {e}")
        
    # 2. PRIORITY FALLBACK: ElevenLabs (Fast & High Quality)
    if not path:
        path, dur, word_timestamps = _generate_elevenlabs(text_to_speak, mp3_path)
        
    # 3. ABSOLUTE FALLBACK: Edge TTS
    if not path:
        path, dur, word_timestamps = _generate_edge_tts(text_to_speak, mp3_path)

    # Post-process: Restore original word spellings for subtitles
    if word_timestamps:
        word_timestamps = restore_original_words(word_timestamps, original_raw_text, custom_phonetic_map=custom_phonetic_map)

    # Post-process: Trim dead air at start/end
    if path and word_timestamps:
        dur, word_timestamps = trim_audio_silence(path, word_timestamps)

    return path, dur, word_timestamps
