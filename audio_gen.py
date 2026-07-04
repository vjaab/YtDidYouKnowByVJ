"""
PRIMARY PATH: F5-TTS Local Voice Cloning (Your Voice).
"""

import os
import json
import asyncio
import re
import random
import numpy as np
from datetime import datetime
from config import OUTPUT_DIR, BASE_DIR
import imageio_ffmpeg
from pydub import AudioSegment
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

from google import genai
from google.genai import types
import soundfile as sf


def _apply_stable_ts(audio_path, text):
    try:
        import stable_whisper
        import warnings
        import signal

        # Suppress ALL warnings including C-level scipy Cholesky warnings
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

        # Use 'tiny' model on CPU for speed (base hangs on GHA runners)
        model_size = 'tiny'
        print(f"Running stable-ts ({model_size}) to extract REAL word timestamps...")
        model = stable_whisper.load_model(model_size)

        # Timeout mechanism to prevent infinite Cholesky loops on CPU
        class AlignmentTimeout(Exception):
            pass

        def _timeout_handler(signum, frame):
            raise AlignmentTimeout("stable-ts alignment timed out after 120s")

        # Set 120s timeout (only works on Unix/Linux — GHA is Linux)
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(120)  # 120 second timeout
        except (AttributeError, ValueError):
            pass  # Windows or signal not available — skip timeout

        try:
            result = model.align(audio_path, text, language='en')
        finally:
            # Cancel the alarm
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass

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
        # Cleanup model on failure too
        try:
            del model
        except NameError:
            pass
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

def optimize_audio_gaps(audio_path, word_timestamps, max_gap_s=0.25, target_gap_s=0.08):
    """
    Detects silent gaps between words and shortens them if they exceed max_gap_s.
    Modifies both the audio file and the word_timestamps in place, and shifts
    all subsequent word timings to keep audio, subtitles, and lip-sync 100% in-sync.
    """
    from pydub import AudioSegment
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # We will build a new AudioSegment and a new list of word timestamps
        new_audio = AudioSegment.empty()
        new_ts = []
        
        # Sort word timestamps
        word_timestamps = sorted(word_timestamps, key=lambda x: x["start"])
        
        if not word_timestamps:
            return len(audio) / 1000.0, word_timestamps
            
        # Add the audio before the first word
        first_word_start_ms = int(word_timestamps[0]["start"] * 1000)
        if first_word_start_ms > 0:
            new_audio += audio[:first_word_start_ms]
            
        for idx, wt in enumerate(word_timestamps):
            w_start_ms = int(wt["start"] * 1000)
            w_end_ms = int(wt["end"] * 1000)
            
            # Start of the word in the new audio
            new_w_start_ms = len(new_audio)
            new_audio += audio[w_start_ms:w_end_ms]
            new_w_end_ms = len(new_audio)
            
            # Save adjusted timestamps
            new_ts.append({
                "word": wt["word"],
                "start": round(new_w_start_ms / 1000.0, 3),
                "end": round(new_w_end_ms / 1000.0, 3)
            })
            
            # Check gap to the next word
            if idx < len(word_timestamps) - 1:
                next_w_start_ms = int(word_timestamps[idx + 1]["start"] * 1000)
                gap_ms = next_w_start_ms - w_end_ms
                
                if gap_ms > max_gap_s * 1000:
                    # Compress the gap to target_gap_s
                    target_gap_ms = int(target_gap_s * 1000)
                    silence_segment = audio[w_end_ms:next_w_start_ms]
                    new_audio += silence_segment[:target_gap_ms]
                else:
                    # Keep original gap
                    if gap_ms > 0:
                        new_audio += audio[w_end_ms:next_w_start_ms]
                        
        # Append remaining audio after the last word
        last_word_end_ms = int(word_timestamps[-1]["end"] * 1000)
        if last_word_end_ms < len(audio):
            new_audio += audio[last_word_end_ms:]
            
        # Export the optimized audio back to the path
        new_audio.export(audio_path, format="wav" if audio_path.endswith(".wav") else "mp3")
        new_duration = len(new_audio) / 1000.0
        print(f"   🔊 Pacing Optimized: silent gaps trimmed. {len(audio)/1000.0:.2f}s -> {new_duration:.2f}s")
        return new_duration, new_ts
    except Exception as e:
        print(f"   ⚠️ Gap pacing optimization failed: {e}")
        return len(audio) / 1000.0, word_timestamps

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

def _smart_split_sentences(text, max_chars=80):
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


def _remove_vocal_artifacts(wav_path, word_timestamps, text):
    """
    Clean up AI voice generation artifacts:
    1. Detect and remove common vocal glitches ("tatas", "uh", "um", "eh")
    2. Fill gaps with cross-faded silence or interpolated audio
    3. Adjust word timestamps accordingly
    """
    from pydub import AudioSegment
    import numpy as np
    
    audio = AudioSegment.from_wav(wav_path)
    audio_arr = np.array(audio.get_array_of_samples())
    
    if audio.channels == 2:
        audio_arr = audio_arr.reshape((-1, 2))
    
    sample_rate = audio.frame_rate
    artifacts_removed = []
    
    # Common AI vocal artifacts to detect (in phonetic form)
    artifact_patterns = [
        ("tatas", 0.15, 0.4),   # "tatas" artifact ~150-400ms
        ("tat a", 0.15, 0.4),   # split "tatas"
        ("ta tas", 0.15, 0.4),  # split "tatas"
    ]
    
    # Use word timestamps to find suspicious short segments
    cleaned_timestamps = []
    accumulated_shift = 0.0
    
    unconditional_fillers = {
        "uh", "um", "eh", "ah", "er", "hmm", "uhh", "umm", "uhhh", "ummm"
    }
    
    for i, wt in enumerate(word_timestamps):
        word = wt["word"].lower().strip().strip(".,?!:;-\"'_")
        duration = wt["end"] - wt["start"]
        
        is_artifact = False
        
        # 1. Unconditional filler word check
        if word in unconditional_fillers:
            is_artifact = True
            artifacts_removed.append((wt["start"], wt["end"], wt["word"]))
        
        # 2. Check other artifact patterns with duration constraints (e.g. "tatas")
        if not is_artifact:
            for pattern, min_dur, max_dur in artifact_patterns:
                if pattern in word and min_dur <= duration <= max_dur:
                    # Check if this word is isolated (surrounded by longer words)
                    prev_dur = word_timestamps[i-1]["end"] - word_timestamps[i-1]["start"] if i > 0 else 0
                    next_dur = word_timestamps[i+1]["end"] - word_timestamps[i+1]["start"] if i < len(word_timestamps)-1 else 0
                    
                    # More aggressive detection: if surrounded by real words (>0.25s) and this is short (<0.4s)
                    if (prev_dur > 0.25 or next_dur > 0.25) and duration < 0.4:
                        is_artifact = True
                        artifacts_removed.append((wt["start"], wt["end"], wt["word"]))
                        break
        
        # 3. Also detect: any very short word (<0.15s) that's not a common short word
        common_short = {"a", "i", "in", "on", "at", "to", "of", "the", "and", "or", "but", "for", "as", "is", "it", "he", "she", "we", "you", "my", "me", "by", "do", "go", "no", "so", "up", "us"}
        if not is_artifact and duration < 0.15 and word not in common_short:
            # Check if surrounded by normal words
            prev_dur = word_timestamps[i-1]["end"] - word_timestamps[i-1]["start"] if i > 0 else 0
            next_dur = word_timestamps[i+1]["end"] - word_timestamps[i+1]["start"] if i < len(word_timestamps)-1 else 0
            if prev_dur > 0.2 and next_dur > 0.2:
                is_artifact = True
                artifacts_removed.append((wt["start"], wt["end"], wt["word"]))
        
        if is_artifact:
            # Shift all subsequent word timestamps by this removed segment's duration
            accumulated_shift += duration
        else:
            shifted_wt = wt.copy()
            shifted_wt["start"] = max(0.0, round(wt["start"] - accumulated_shift, 3))
            shifted_wt["end"] = max(0.0, round(wt["end"] - accumulated_shift, 3))
            cleaned_timestamps.append(shifted_wt)
    
    if artifacts_removed:
        print(f"   🧹 Removed {len(artifacts_removed)} vocal artifacts: {[a[2] for a in artifacts_removed]}")
        
        # Rebuild audio without artifact segments using crossfade.
        # CRITICAL: Process in REVERSE order (last-to-first) so that each
        # splice doesn't shift the byte positions of earlier artifacts.
        for start_s, end_s, word in reversed(artifacts_removed):
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            
            # Guard against positions that exceed current audio length
            if start_ms >= len(audio) or end_ms > len(audio):
                continue
            
            # Create a short crossfade bridge
            before = audio[:start_ms]
            after = audio[end_ms:]
            
            # Crossfade 50ms
            fade_ms = 50
            if len(before) > fade_ms and len(after) > fade_ms:
                before = before.fade_out(fade_ms)
                after = after.fade_in(fade_ms)
            audio = before + after
    
    audio.export(wav_path, format="wav")
    return cleaned_timestamps if artifacts_removed else word_timestamps


def _inject_human_phrasing(wav_path, word_timestamps, text):
    """
    Add natural human-like phrasing variations:
    1. Micro-pauses at clause boundaries (commas, semicolons)
    2. Breath-like pauses at sentence boundaries
    3. Subtle pitch/tempo variation (prosody simulation via time-stretch)
    4. Pre-emphasis on key technical terms
    4. Simulated breathing at natural intervals
    """
    from pydub import AudioSegment
    import random
    
    audio = AudioSegment.from_wav(wav_path)
    
    # Parse text for punctuation-based pause insertion
    import re
    sentences = re.split(r'([.!?]+)', text)
    clauses = re.split(r'([,;:]+)', text)
    
    # We'll apply subtle time-stretching to simulate prosody
    # Speed variation: ±3% per clause for natural flow
    segments = []
    current_pos = 0
    
    # Track position for breathing
    last_breath_pos = 0
    breath_interval_ms = random.randint(8000, 15000)  # Breath every 8-15 seconds
    
    for i, wt in enumerate(word_timestamps):
        word = wt["word"]
        start_ms = int(wt["start"] * 1000)
        end_ms = int(wt["end"] * 1000)
        
        # Add micro-pause after commas/semicolons
        if word.rstrip(',;:').endswith((',', ';', ':')):
            pause_ms = random.randint(80, 150)  # 80-150ms micro-pause
            silence = AudioSegment.silent(duration=pause_ms)
            segments.append(audio[current_pos:start_ms])
            segments.append(audio[start_ms:end_ms])
            segments.append(silence)
            current_pos = end_ms
        
        # Add breath pause after sentences
        elif word.rstrip('.!?').endswith(('.', '!', '?')):
            pause_ms = random.randint(200, 350)  # 200-350ms breath pause
            silence = AudioSegment.silent(duration=pause_ms)
            segments.append(audio[current_pos:start_ms])
            segments.append(audio[start_ms:end_ms])
            segments.append(silence)
            current_pos = end_ms
            
            # Simulate a breath every ~10 seconds
            if end_ms - last_breath_pos > breath_interval_ms:
                breath_pause = AudioSegment.silent(duration=random.randint(300, 500))
                segments.append(breath_pause)
                last_breath_pos = end_ms
                breath_interval_ms = random.randint(8000, 15000)
    
    if segments:
        # Rebuild with pauses
        new_audio = sum(segments, AudioSegment.empty())
        if current_pos < len(audio):
            new_audio += audio[current_pos:]
        
        # Apply subtle per-clause speed variation (±4%)
        # Split into ~4-6 second chunks and vary speed slightly
        chunk_ms = random.randint(4000, 6000)
        final_audio = AudioSegment.empty()
        
        for i in range(0, len(new_audio), chunk_ms):
            chunk = new_audio[i:i+chunk_ms]
            if len(chunk) > 1000:  # Only process chunks > 1s
                speed_factor = 1.0 + random.uniform(-0.04, 0.04)
                # Time stretch without pitch change (approximate via frame rate manipulation)
                new_len = int(len(chunk) / speed_factor)
                if new_len > 0:
                    chunk = chunk._spawn(chunk.raw_data, overrides={
                        "frame_rate": int(chunk.frame_rate * speed_factor)
                    }).set_frame_rate(chunk.frame_rate)
                    if len(chunk) > new_len:
                        chunk = chunk[:new_len]
                    elif len(chunk) < new_len:
                        chunk = chunk + AudioSegment.silent(duration=new_len - len(chunk))
            final_audio += chunk
        
        # Add very subtle volume variation (±1.5dB) to simulate natural emphasis
        final_audio = _add_natural_volume_variation(final_audio)
        
        final_audio.export(wav_path, format="wav")
        print(f"   🎭 Human phrasing injected: micro-pauses, breath pauses, ±4% tempo variation, natural volume dynamics")
    
    return word_timestamps  # Timestamps would need recalculation in production


def _add_natural_volume_variation(audio):
    """
    Apply subtle volume automation to simulate natural speech dynamics.
    Randomly boosts/reduces 200-500ms regions by ±1.5dB.
    """
    import random
    from pydub import AudioSegment
    
    samples = audio.get_array_of_samples()
    if audio.channels == 2:
        samples = np.array(samples).reshape((-1, 2))
    else:
        samples = np.array(samples)
    
    sample_rate = audio.frame_rate
    total_samples = len(samples)
    
    # Apply ~3-5 volume variations across the audio
    num_variations = random.randint(3, 5)
    for _ in range(num_variations):
        center = random.randint(int(0.1 * total_samples), int(0.9 * total_samples))
        width = random.randint(int(0.1 * sample_rate), int(0.5 * sample_rate))  # 100-500ms
        gain_db = random.uniform(-1.5, 1.5)
        gain_linear = 10 ** (gain_db / 20.0)
        
        start = max(0, center - width // 2)
        end = min(total_samples, center + width // 2)
        
        # Smooth gain envelope (cosine)
        for i in range(start, end):
            progress = (i - start) / (end - start)
            envelope = 0.5 * (1 - np.cos(2 * np.pi * progress))
            actual_gain = 1.0 + (gain_linear - 1.0) * envelope
            if audio.channels == 2:
                samples[i, 0] = np.clip(samples[i, 0] * actual_gain, -32768, 32767)
                samples[i, 1] = np.clip(samples[i, 1] * actual_gain, -32768, 32767)
            else:
                samples[i] = np.clip(samples[i] * actual_gain, -32768, 32767)
    
    if audio.channels == 2:
        new_samples = samples.flatten().astype(np.int16)
    else:
        new_samples = samples.astype(np.int16)
    
    return audio._spawn(new_samples.tobytes())


def _postprocess_voice_audio(wav_path, word_timestamps=None, original_text=None):
    """
    Professional post-processing chain to enhance clarity and presence:
    1. High-pass filter at 120Hz to remove low-end rumble and mud.
    2. Three-Band Presence EQ Crossover Network:
       - Lows (100Hz - 200Hz): warm chest boost (+1.5dB).
       - Mids (200Hz - 3kHz): vocal core.
       - Highs (3kHz - 15kHz): crisp sparkle presence boost (+3.5dB) and high-end air.
    3. Dynamic Range Compression to level out the voice and make it "pop".
    4. Final normalization to -1dB for consistent loudness.
    5. Subtle fade-in/out to prevent clicks.
    6. NEW: Remove vocal artifacts (glitches, filler sounds)
    7. NEW: Inject human-like phrasing variation
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize, compress_dynamic_range
        
        # Step 0: Clean artifacts if timestamps available
        if word_timestamps and original_text:
            word_timestamps = _remove_vocal_artifacts(wav_path, word_timestamps, original_text)
            word_timestamps = _inject_human_phrasing(wav_path, word_timestamps, original_text)
        
        audio = AudioSegment.from_wav(wav_path)
        
        # 1. High-pass filter (120Hz) - Removes low-frequency room rumble
        audio = audio.high_pass_filter(120)
        
        # 2. Three-Band Presence EQ Crossover
        lows = audio.low_pass_filter(200).high_pass_filter(100)
        mids = audio.high_pass_filter(200).low_pass_filter(3000)
        highs = audio.high_pass_filter(3000)
        
        # Apply premium boosting gains
        lows = lows + 1.5   # Warmth chest boost
        highs = highs + 3.5 # Sparkle & presence air boost
        
        # Recombine frequency crossover bands
        audio = lows.overlay(mids).overlay(highs)
        
        # 3. Dynamic Compression - Makes the voice sound authoritative and professional
        # Threshold -15dB, Ratio 3:1, Attack 5ms, Release 50ms
        audio = compress_dynamic_range(audio, threshold=-15.0, ratio=3.0, attack=5.0, release=50.0)
        
        # 4. Final Normalization
        audio = normalize(audio, headroom=1.0)
        
        # 5. Prevent click artifacts
        audio = audio.fade_in(5).fade_out(5)
        
        audio.export(wav_path, format="wav")
        print(f"   🎙️ Audio enhanced: 120Hz HPF, 3-band presence EQ, dynamic compression, normalized to -1dB, artifacts cleaned, human phrasing injected")
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
            nfe_step=64,
            remove_silence=True,
            speed=1.02
        )
        segment_paths.append(seg_path)
        
    # Combine segments with 80ms cross-fade for seamless joins (was 30ms — too short,
    # caused audible hard-cuts between sentence segments)
    CROSSFADE_MS = 80
    SEGMENT_FADE_MS = 15  # Per-segment fade to eliminate click artifacts at boundaries
    combined = AudioSegment.empty()
    for idx_sp, sp in enumerate(segment_paths):
        seg = AudioSegment.from_wav(sp)
        # Apply per-segment fade-in/out to prevent click artifacts before crossfading
        if len(seg) > SEGMENT_FADE_MS * 2:
            seg = seg.fade_in(SEGMENT_FADE_MS).fade_out(SEGMENT_FADE_MS)
        if idx_sp == 0:
            combined = seg
        else:
            combined = combined.append(seg, crossfade=CROSSFADE_MS)
    
    # Clean up segment files
    for sp in segment_paths:
        try: os.remove(sp)
        except: pass
        
    combined.export(wav_path, format="wav")
    
    duration = get_audio_duration(wav_path)
    
    # Word timestamps via stable-ts
    word_timestamps = _apply_stable_ts(wav_path, text)
    if not word_timestamps:
        word_timestamps = _estimate_timestamps(text, duration)
    
    # Post-process for professional voice quality
    _postprocess_voice_audio(wav_path, word_timestamps, text)
        
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
        
        # Post-process for professional voice quality (artifact removal, human phrasing)
        _postprocess_voice_audio(output_path, word_timestamps, text)
        
        return output_path, duration, word_timestamps
    except Exception as e:
        print(f"❌ Edge TTS also failed: {e}")
        return None, 0, []


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT SANITIZATION (Strip LLM artifacts before TTS)
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_script_for_tts(text):
    """
    Aggressively strips LLM generation artifacts that Gemini sometimes leaks
    into the 'script' field. Must run BEFORE clean_tts_text().
    
    Catches:
    - Alternate phrasing options: "word1/word2", "use X or contextually appropriate Y"
    - Stage directions: [pause], [beat], (dramatic tone), (softly)
    - Meta-instructions: "as one compound word", "pronounce as"
    - Formatting artifacts: numbered lists, bullets, markdown headers
    - Trailing loop artifacts: hook text repeated after CTA
    """
    if not text:
        return ""
    
    cleaned = text
    
    # 1. Remove alternate phrasing patterns: "word1/word2" → keep word1
    #    e.g., "use stricks/tricks" → "use stricks"
    #    But preserve legit slashes like "CI/CD", "24/7", "input/output"
    legit_slash_terms = {
        'CI/CD', 'I/O', 'input/output', 'read/write', 'TCP/IP',
        'client/server', 'OS/2', 'w/', '24/7', 'he/she', 'and/or'
    }
    # Protect legit slash terms by replacing with placeholders
    placeholders = {}
    for i, term in enumerate(legit_slash_terms):
        placeholder = f"__SLASH_PLACEHOLDER_{i}__"
        placeholders[placeholder] = term
        cleaned = cleaned.replace(term, placeholder)
    
    # Remove "word1/word2" patterns (keep the first word)
    cleaned = re.sub(r'\b(\w+)/(\w+)\b', r'\1', cleaned)
    
    # Restore legit slash terms
    for placeholder, term in placeholders.items():
        cleaned = cleaned.replace(placeholder, term)
    
    # 2. Remove meta-instruction phrases that LLMs sometimes leak
    meta_patterns = [
        r'(?i)\bas one compound word\b',
        r'(?i)\bor contextually appropriate\b[^.!?]*',
        r'(?i)\bpronounce\s+(as|it|this)\b[^.!?]*',
        r'(?i)\buse\s+\w+\s+or\s+if\s+contextually\b[^.!?]*',
        r'(?i)\bspoken\s+as\b[^.!?]*',
        r'(?i)\bsay\s+it\s+as\b[^.!?]*',
        r'(?i)\bformat(?:ted)?\s+as\b[^.!?]*',
        r'(?i)\bnote\s*:\s*[^.!?]*',
        r'(?i)\binstruction\s*:\s*[^.!?]*',
        r'(?i)\bdirector\'?s?\s+note\s*:\s*[^.!?]*',
    ]
    for pattern in meta_patterns:
        cleaned = re.sub(pattern, ' ', cleaned)
    
    # 3. Remove stage directions in brackets/parens: [pause], (beat), [dramatic tone]
    cleaned = re.sub(r'\[\s*(pause|beat|silence|breathe|dramatic|softly|loudly|whisper|slowly|quickly|emphasis|transition|cut)\s*[^\]]*\]', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\(\s*(pause|beat|silence|breathe|dramatic|softly|loudly|whisper|slowly|quickly|emphasis|transition)\s*[^)]*\)', ' ', cleaned, flags=re.IGNORECASE)
    
    # 4. Remove scene/section labels that shouldn't be spoken
    cleaned = re.sub(r'(?i)^\s*(HOOK|PROBLEM|SOLUTION|CTA|OUTRO|INTRO|SCENE|PART)\s*[:\-]\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'(?i)\b(HOOK|PROBLEM|SOLUTION|CTA|SECTION)\s*\d*\s*:', ' ', cleaned)
    
    # 5. Remove numbered list formatting ("1. ", "2. ", etc.)
    cleaned = re.sub(r'(?m)^\s*\d+\.\s+', '', cleaned)
    
    # 6. Remove markdown formatting artifacts
    cleaned = re.sub(r'#{1,6}\s+', '', cleaned)  # Headers
    cleaned = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', cleaned)  # Bold/italic
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)  # Inline code
    
    # 7. Detect and truncate trailing loop artifacts
    #    If the CTA appears and then the hook repeats, cut at the CTA
    #    Pattern: "...follow for more..." followed by text that re-states the hook
    cta_markers = [
        r'(?i)(follow\s+for\s+more[^.!?]*[.!?])',
        r'(?i)(subscribe\s+for\s+more[^.!?]*[.!?])',
        r'(?i)(save\s+this[^.!?]*[.!?])',
        r'(?i)(drop\s+a\s+comment[^.!?]*[.!?])',
        r'(?i)(comment\s+below[^.!?]*[.!?])',
    ]
    for cta_pattern in cta_markers:
        match = re.search(cta_pattern, cleaned)
        if match:
            cta_end = match.end()
            remaining = cleaned[cta_end:].strip()
            # If remaining text is short (<40 chars) and looks like a loop-back, remove it
            if remaining and len(remaining) < 80:
                cleaned = cleaned[:cta_end].strip()
                break
    
    # 8. Clean up whitespace artifacts
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def clean_tts_text(text, phonetic=True, custom_phonetic_map=None):
    """
    Strips out AI meta-instructions and fixes pronunciation issues.
    If phonetic=True, it replaces difficult words with phonetic spellings.
    Implements Phase 1: Text Script Pre-processing & Cleanup rules.
    """
    if not text: return ""
    
    # ── PHASE 0: SANITIZE LLM ARTIFACTS ────────────────────────────────────
    cleaned = sanitize_script_for_tts(text)
    
    # ── PHASE 0.5: TRANSITIONS & GRAMMAR CLEANUP ───────────────────────────
    # 1. Replace settings path chevrons/arrows with a natural spoken transition
    # Matches '->', '=>', '→', '>>', '›', '»', or single '>'
    cleaned = re.sub(r'\s*(?:->|=>|→|>\s*>|›|»)\s*', ' ... then ', cleaned)
    cleaned = re.sub(r'\s*>\s*', ' ... then ', cleaned)

    # 2. Fix common grammar/stutter patterns in generated script text
    cleaned = re.sub(r'\bprivacy vulnerable\b', 'privacy vulnerability', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bprivacy b now\b', 'privacy now', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bb now\b', 'now', cleaned, flags=re.IGNORECASE)
    
    # ── PHASE 1: TEXT SCRIPT PRE-PROCESSING & CLEANUP ──────────────────────
    
    # 1. Broadly remove ALL bracketed and parenthesized meta-instructions/timestamps
    cleaned = re.sub(r'\[[^\]]*\]', ' ', cleaned)
    cleaned = re.sub(r'\([^)]*\)', ' ', cleaned)
    
    # 2. Fix pronunciation artifacts (The "Strike" issue)
    cleaned = cleaned.replace("—", "...")  # Em-dash
    cleaned = cleaned.replace("–", "...")  # En-dash
    cleaned = cleaned.replace("--", "...")  # Double hyphen
    cleaned = cleaned.replace("*", " ")     # Asterisks
    cleaned = cleaned.replace("•", " ")     # Bullet point
    cleaned = cleaned.replace("·", " ")     # Middle dot
    cleaned = cleaned.replace("⁃", " ")     # Hyphen bullet
    cleaned = cleaned.replace("●", " ")     # Circle bullet
    cleaned = cleaned.replace("▪", " ")     # Square bullet
    cleaned = cleaned.replace("~", " ")     # Tilde
    
    # Standalone hyphens
    cleaned = re.sub(r'\n\s*-\s*', '\n ', cleaned)
    cleaned = re.sub(r'\s+-\s+', ' ... ', cleaned)
    
    # 3. ENFORCE PHRASING PAUSES & EMPHASIS (Phase 2 Rule 1) - Text-level markers
    # Must run BEFORE phonetic processing to catch urgency words in original form
    cleaned = _enforce_phrasing_pauses(cleaned)
    
    # 4. EXPAND FILE EXTENSIONS TO PHONETICS (Phase 1 Rule 1)
    cleaned = _expand_file_extensions(cleaned)
    
    # 5. CLEAN NUMERIC & VERSION RESOLUTION (Phase 1 Rule 2)
    cleaned = _expand_version_numbers(cleaned)
    
    # 6. RE-FORMAT REPO & TECH TERMINOLOGY (Phase 1 Rule 3)
    cleaned = _reformat_tech_terminology(cleaned)
    
    # 7. Phonetic Cleanups for Clarity (TECH TERMS ONLY)
    if phonetic:
        # Merge global dictionary and custom Gemini map
        full_map = PHONETIC_DICT.copy()
        if custom_phonetic_map:
            full_map.update(custom_phonetic_map)

        # NOTE: pronunciation_fixes for common English words ("automatically" → "aw-to-mat-ik-lee")
        # has been REMOVED. ElevenLabs/Edge-TTS already pronounce these correctly, and the
        # hyphenated forms were rendering literally in captions ("aw-to-mat-ik-lee uncovers").
        # Only tech-specific terms from PHONETIC_DICT are kept (NVIDIA, GGUF, vLLM, etc.).
        
        # Filter: Skip common English words that TTS engines already handle correctly.
        # Only apply phonetics to: brand names, acronyms, tech jargon, foreign terms,
        # and genuinely tricky words that TTS consistently mispronounces.
        _COMMON_ENGLISH_SKIP = {
            # -ly adverbs that TTS handles fine
            "locally", "virtually", "globally", "technically", "theoretically",
            "specifically", "artificially", "automatically", "dramatically",
            "systematically", "practically", "effectively", "relatively",
            "literally", "obviously", "certainly", "immediately", "simultaneously",
            "substantially", "significantly", "considerably", "remarkably",
            "fundamentally", "essentially", "explicitly", "implicitly",
            "intrinsically", "extrinsically", "synchronously", "asynchronously",
            "basically", "probably", "actually", "naturally", "generally",
            "especially", "definitely", "particularly", "temporarily",
            "extraordinarily", "consequently", "subsequently", "ultimately",
            "historically", "empirically", "statistically", "hypothetically",
            "figuratively", "potentially", "successfully", "inevitably",
            # Common nouns/adjectives TTS handles fine
            "vulnerability", "vulnerabilities", "comfortable", "vegetable",
            "chocolate", "interesting", "different", "temperature", "restaurant",
            "necessary", "laboratory", "category", "environment", "government",
            "development", "industry", "innovation", "analysis", "strategy",
            "comprehensive", "revolutionary", "transformative", "groundbreaking",
            "unprecedented", "sophisticated", "uncomplicated", "incredible",
            "incredibly", "accessible", "accessibility", "relevant",
            "significant", "remarkable", "available", "availability",
            "reliability", "scalability", "infrastructure", "implementation",
            "configuration", "optimization", "architecture", "architectural",
            "perspective", "perspectives", "resilience", "compliance",
            # Common words TTS handles fine 
            "fact", "facts", "way", "ways", "only", "says", "said",
            "iron", "often", "listen", "height", "month", "months",
            "island", "islands", "subtle", "subtly", "debt", "doubt",
            "receipt", "salmon", "almond", "Wednesday", "February",
            "writer", "officer", "market", "analyst", "analysts",
            "specific", "prioritize", "distributed", "parallel",
            "concurrency", "orchestration", "technical",
            # Heteronyms that could change meaning if replaced
            "read", "lead", "live", "content", "present", "record",
            "object", "produce", "project", "estimate", "conduct",
            # Common "-tion" words
            "standardization", "modernization", "digitalization",
            "pronunciation", "instantaneous", "instantaneously", "instantly",
        }
        
        filtered_map = {
            word: replacement 
            for word, replacement in full_map.items() 
            if word.lower() not in _COMMON_ENGLISH_SKIP
        }

        # Sort keys by word count and length descending to apply longer match first
        sorted_keys = sorted(filtered_map.keys(), key=lambda k: (len(k.split()), len(k)), reverse=True)
        for word in sorted_keys:
            pattern = r'\b' + re.escape(word) + r'\b'
            cleaned = re.sub(pattern, filtered_map[word], cleaned, flags=re.IGNORECASE)

        # 8. Auto-detect remaining hard words via g2p_en (neural G2P fallback)
        try:
            auto_corrections = auto_detect_hard_words(cleaned)
            for word, respelling in auto_corrections.items():
                pattern = r'\b' + re.escape(word) + r'\b'
                cleaned = re.sub(pattern, respelling, cleaned, flags=re.IGNORECASE)
        except Exception as e:
            print(f"g2p_en auto-detection skipped: {e}")
    else:
        # Non-phonetic (Subtitle) Spell Correction
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


def _expand_file_extensions(text):
    """
    Phase 1 Rule 1: Expand file extensions to phonetics.
    """
    # Order matters - longer extensions first to avoid partial matches
    extensions = [
        (r'\.ipynb\b', ' dot i p y notebook'),
        (r'\.txt\b', ' dot text'),
        (r'\.text\b', ' dot text'),
        (r'\.py\b', ' dot p y'),
        (r'\.csv\b', ' dot c s v'),
        (r'\.json\b', ' dot j son'),
        (r'\.js\b', ' dot j s'),
        (r'\.ts\b', ' dot t s'),
        (r'\.jsx\b', ' dot j s x'),
        (r'\.tsx\b', ' dot t s x'),
        (r'\.html\b', ' dot h t m l'),
        (r'\.css\b', ' dot c s s'),
        (r'\.md\b', ' dot m d'),
        (r'\.pdf\b', ' dot p d f'),
        (r'\.png\b', ' dot p n g'),
        (r'\.jpg\b', ' dot j p g'),
        (r'\.jpeg\b', ' dot j p e g'),
        (r'\.gif\b', ' dot g i f'),
        (r'\.mp4\b', ' dot m p four'),
        (r'\.mov\b', ' dot m o v'),
        (r'\.wav\b', ' dot w a v'),
        (r'\.mp3\b', ' dot m p three'),
        (r'\.zip\b', ' dot zip'),
        (r'\.tar\b', ' dot tar'),
        (r'\.gz\b', ' dot g z'),
        (r'\.yml\b', ' dot y m l'),
        (r'\.yaml\b', ' dot y a m l'),
        (r'\.toml\b', ' dot t o m l'),
        (r'\.ini\b', ' dot i n i'),
        (r'\.cfg\b', ' dot c f g'),
        (r'\.conf\b', ' dot conf'),
        (r'\.log\b', ' dot log'),
        (r'\.sql\b', ' dot s q l'),
        (r'\.db\b', ' dot d b'),
        (r'\.sqlite\b', ' dot s q lite'),
        (r'\.dockerfile\b', ' dot docker file'),
        (r'\.gitignore\b', ' dot git ignore'),
        (r'\.env\b', ' dot env'),
        (r'\.sh\b', ' dot s h'),
        (r'\.bash\b', ' dot bash'),
        (r'\.zsh\b', ' dot z s h'),
        (r'\.rs\b', ' dot r s'),
        (r'\.go\b', ' dot go'),
        (r'\.java\b', ' dot java'),
        (r'\.kt\b', ' dot k t'),
        (r'\.swift\b', ' dot swift'),
        (r'\.rb\b', ' dot r b'),
        (r'\.php\b', ' dot p h p'),
        (r'\.cs\b', ' dot c s'),
        (r'\.cpp\b', ' dot c p p'),
        (r'\.h\b', ' dot h'),
        (r'\.hpp\b', ' dot h p p'),
    ]
    
    for pattern, replacement in extensions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def _expand_version_numbers(text):
    """
    Phase 1 Rule 2: Clean numeric & version resolution.
    Convert dot-notation versions to spoken words.
    """
    # Match version patterns like 26.5.2, 1.0.0, 3.14.15, etc.
    # Also handles v26.5.2, version 26.5.2 patterns
    
    def version_to_words(match):
        version_str = match.group(0)
        # Remove leading 'v' or 'version' prefix if present
        version_str = re.sub(r'^[vV](?=\d)', '', version_str)
        version_str = re.sub(r'^version\s+', '', version_str, flags=re.IGNORECASE)
        
        parts = version_str.split('.')
        words = []
        for part in parts:
            if part.isdigit():
                words.append(_number_to_words(int(part)))
            else:
                words.append(part)
        return ' point '.join(words)
    
    # Pattern: version numbers with 2+ dot-separated parts, optionally prefixed with v/version
    # Matches: 1.0, 26.5.2, v1.2.3, version 2.0, etc.
    version_pattern = r'\b(?:v|version\s+)?\d+(?:\.\d+){1,3}\b'
    text = re.sub(version_pattern, version_to_words, text, flags=re.IGNORECASE)
    
    # Also handle standalone decimal numbers that look like versions (e.g., "Python 3.11")
    # But be careful not to match regular decimals like prices ($3.99) or measurements
    # We'll target version-like contexts
    return text


def _number_to_words(n):
    """Convert integer to words (supports up to billions)."""
    if n == 0:
        return "zero"
    
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    def convert_hundreds(num):
        if num == 0:
            return ""
        result = ""
        if num >= 100:
            result += ones[num // 100] + " hundred"
            num %= 100
            if num > 0:
                result += " "
        if num >= 20:
            result += tens[num // 10]
            num %= 10
            if num > 0:
                result += "-" + ones[num]
        elif num >= 10:
            result += teens[num - 10]
        elif num > 0:
            result += ones[num]
        return result
    
    if n >= 1_000_000_000:
        return convert_hundreds(n // 1_000_000_000) + " billion" + (" " + convert_hundreds((n % 1_000_000_000) // 1_000_000) + " million" if n % 1_000_000_000 >= 1_000_000 else "") + (" " + convert_hundreds((n % 1_000_000) // 1000) + " thousand" if n % 1_000_000 >= 1000 else "") + (" " + convert_hundreds(n % 1000) if n % 1000 > 0 else "")
    elif n >= 1_000_000:
        return convert_hundreds(n // 1_000_000) + " million" + (" " + convert_hundreds((n % 1_000_000) // 1000) + " thousand" if n % 1_000_000 >= 1000 else "") + (" " + convert_hundreds(n % 1000) if n % 1000 > 0 else "")
    elif n >= 1000:
        return convert_hundreds(n // 1000) + " thousand" + (" " + convert_hundreds(n % 1000) if n % 1000 > 0 else "")
    else:
        return convert_hundreds(n)


def _reformat_tech_terminology(text):
    """
    Phase 1 Rule 3: Re-format repo & tech terminology.
    Inject spaces/hyphens between compound technical phrases.
    Add syllable breaks for tongue-twister terms.
    """
    # Compound technical terms that need separation
    compounds = [
        (r'\binferencenotebook\b', 'inference notebook'),
        (r'\binference\s*notebook\b', 'inference notebook'),
        (r'\bparameterization\b', 'parameter-ization'),
        (r'\bcompatibility\b', 'com-pat-i-bil-i-ty'),
        (r'\binfrastructure\b', 'in-fra-struc-ture'),
        (r'\borchestration\b', 'or-ches-tra-tion'),
        (r'\bcontainerization\b', 'con-tain-er-ization'),
        (r'\bvirtualization\b', 'vir-tu-al-ization'),
        (r'\bobservability\b', 'ob-serv-a-bil-i-ty'),
        (r'\binteroperability\b', 'in-ter-op-er-a-bil-i-ty'),
        (r'\bscalability\b', 'scal-a-bil-i-ty'),
        (r'\bmaintainability\b', 'main-tain-a-bil-i-ty'),
        (r'\breliability\b', 're-li-a-bil-i-ty'),
        (r'\bavailability\b', 'avail-a-bil-i-ty'),
        (r'\bconfiguration\b', 'con-fig-u-ra-tion'),
        (r'\bimplementation\b', 'im-ple-men-ta-tion'),
        (r'\boptimization\b', 'op-ti-mi-za-tion'),
        (r'\bstandardization\b', 'stan-dar-diza-tion'),
        (r'\bmodernization\b', 'mod-ern-iza-tion'),
        (r'\bdigitalization\b', 'dig-i-tal-iza-tion'),
        (r'\bauthentication\b', 'au-then-ti-ca-tion'),
        (r'\bauthorization\b', 'au-thor-iza-tion'),
        (r'\bserialization\b', 'se-ri-al-iza-tion'),
        (r'\bdeserialization\b', 'de-se-ri-al-iza-tion'),
        (r'\binitialization\b', 'in-it-ial-iza-tion'),
        (r'\bdeinitialization\b', 'de-in-it-ial-iza-tion'),
        (r'\bvisualization\b', 'vis-u-al-iza-tion'),
        (r'\bconceptualization\b', 'con-cep-tu-al-iza-tion'),
        (r'\brevolutionary\b', 'rev-o-lu-tion-ary'),
        (r'\btransformative\b', 'trans-for-ma-tive'),
        (r'\bunprecedented\b', 'un-prece-dent-ed'),
        (r'\bsophisticated\b', 'so-phis-ti-cated'),
        (r'\bcomprehensive\b', 'com-pre-hen-sive'),
        (r'\bsignificantly\b', 'sig-nif-i-cant-ly'),
        (r'\bsubstantially\b', 'sub-stan-tial-ly'),
        (r'\bconsiderably\b', 'con-sid-er-a-bly'),
        (r'\bremarkably\b', 're-mark-a-bly'),
        (r'\bincredibly\b', 'in-cred-i-bly'),
        (r'\bfundamentally\b', 'fun-da-men-tal-ly'),
        (r'\btechnically\b', 'tech-ni-cal-ly'),
        (r'\btheoretically\b', 'the-o-ret-i-cal-ly'),
        (r'\bhypothetically\b', 'hy-po-thet-i-cal-ly'),
        (r'\bstatistically\b', 'sta-tis-ti-cal-ly'),
        (r'\bempirically\b', 'em-pir-i-cal-ly'),
        (r'\bhistorically\b', 'his-tor-i-cal-ly'),
        (r'\bglobally\b', 'glob-al-ly'),
        (r'\blocally\b', 'lo-cal-ly'),
        (r'\bvirtually\b', 'vir-tu-al-ly'),
        (r'\bfiguratively\b', 'fig-u-ra-tive-ly'),
        (r'\bessentially\b', 'es-sen-tial-ly'),
        (r'\bimplicitly\b', 'im-pli-cit-ly'),
        (r'\bexplicitly\b', 'ex-plic-it-ly'),
        (r'\bintrinsically\b', 'in-trin-sic-al-ly'),
        (r'\bextrinsically\b', 'ex-trin-sic-al-ly'),
        (r'\bsynchronously\b', 'syn-chro-nous-ly'),
        (r'\basynchronously\b', 'a-syn-chro-nous-ly'),
        (r'\bdistributed\b', 'dis-trib-u-ted'),
        (r'\bconfiguration\b', 'con-fig-u-ra-tion'),
        (r'\benvironment\b', 'en-vi-ron-ment'),
    ]
    
    for pattern, replacement in compounds:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def _enforce_phrasing_pauses(text):
    """
    Phase 2 Rule 1: Enforce phrasing pauses at sentence boundaries.
    Only adds pauses at sentence endings (., ?, !) — NOT after every
    comma, colon, or semicolon, which was causing stuttering/run-on feel.
    """
    # 1. Mandatory pause at sentence boundaries (., ?, !) only
    # We use ... (ellipsis) which TTS engines typically render as a pause.
    # Exclude dots that are part of an ellipsis '...' to prevent duplication.
    text = re.sub(r'(?<!\.)([!?]|\.(?!\.))\s+', r'\1 ... ', text)
    
    # 2. DON'T add pauses after commas, colons, semicolons — the TTS engine
    # handles these naturally. Over-adding "..." was causing stuttering.
    
    return text


def _sanitize_tts_symbols(text):
    """
    Pre-TTS cleanup: strips/replaces characters that trip up TTS engines
    and cause glitchy audio artifacts. Runs just before TTS generation.
    """
    if not text:
        return text
    
    cleaned = text
    
    # 1. Replace curly/smart quotes with straight quotes
    cleaned = cleaned.replace('\u201c', '"').replace('\u201d', '"')  # \u201c\u201d \u2192 ""
    cleaned = cleaned.replace('\u2018', "'").replace('\u2019', "'")  # \u2018\u2019 \u2192 ''
    
    # 2. Remove URLs (TTS reads them character by character)
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    cleaned = re.sub(r'www\.\S+', '', cleaned)
    
    # 3. Remove leftover markdown links [text](url) \u2192 text
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', cleaned)
    
    # 4. Replace excessive ellipsis (4+) with single pause marker
    cleaned = re.sub(r'\.{4,}', '...', cleaned)
    
    # 5. Remove inline em/en dashes that aren't already converted
    #    (these cause "dash" to be spoken)
    cleaned = re.sub(r'\s*[\u2013\u2014]\s*', ' ... ', cleaned)
    
    # 6. Remove stray special characters that cause glitches
    cleaned = re.sub(r'[\u2022\u2023\u25aa\u25cf\u2043]', ' ', cleaned)  # bullet chars
    cleaned = re.sub(r'[\u00ab\u00bb\u2039\u203a]', '', cleaned)  # guillemets
    cleaned = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', cleaned)  # zero-width chars
    
    # 7. Clean up resulting whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def _inject_context_pauses(audio_path, word_timestamps):
    """
    Post-generation silence injection at major context shifts.
    Inserts deterministic pauses AFTER TTS generation to give the audience
    time to digest information. Updates word timestamps to stay in sync.
    
    Pauses:
    - 150ms after sentence-ending words (. ! ?)
    - 250ms after major context shifts (But, However, Now, Here's, So, etc.)
    """
    from pydub import AudioSegment
    
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # Context shift markers - words that typically start a new major point
        CONTEXT_SHIFT_WORDS = {
            'but', 'however', 'now', "here's", 'so', 'meanwhile',
            'instead', 'yet', 'still', 'actually', 'seriously',
            'basically', 'look', 'think', 'imagine', 'remember',
            'worse', 'better', 'first', 'second', 'finally',
            'why', 'how', 'what', 'the',
        }
        
        SENTENCE_PAUSE_MS = 300
        CONTEXT_PAUSE_MS = 450
        
        # Build list of pause insertions: (position_ms, pause_duration_ms)
        insertions = []
        
        for i, wt in enumerate(word_timestamps):
            word_raw = wt['word'].strip()
            end_ms = int(wt['end'] * 1000)
            
            # Check if this word ends a sentence
            is_sentence_end = bool(re.search(r'[.!?]$', word_raw))
            
            if is_sentence_end and i < len(word_timestamps) - 1:
                # Check if the NEXT word is a context shift marker
                next_word = word_timestamps[i + 1]['word'].strip().lower().rstrip('.,!?;:')
                
                if next_word in CONTEXT_SHIFT_WORDS:
                    insertions.append((end_ms, CONTEXT_PAUSE_MS))
                else:
                    insertions.append((end_ms, SENTENCE_PAUSE_MS))
        
        if not insertions:
            return len(audio) / 1000.0, word_timestamps
        
        # Build new audio with pauses inserted (process front-to-back)
        new_audio = AudioSegment.empty()
        prev_end = 0
        cumulative_shift_ms = 0
        new_timestamps = [dict(wt) for wt in word_timestamps]  # deep copy
        
        # Track which insertion we need to process next
        insert_idx = 0
        
        for i, wt in enumerate(new_timestamps):
            orig_end_ms = int(word_timestamps[i]['end'] * 1000)
            
            # Shift this word's timestamps by cumulative pause added so far
            wt['start'] = round((word_timestamps[i]['start'] * 1000 + cumulative_shift_ms) / 1000.0, 3)
            wt['end'] = round((word_timestamps[i]['end'] * 1000 + cumulative_shift_ms) / 1000.0, 3)
            
            # Check if we need to insert a pause after this word
            if insert_idx < len(insertions):
                target_pos_ms = insertions[insert_idx][0]
                pause_ms = insertions[insert_idx][1]
                
                # Match by original end position
                if abs(orig_end_ms - target_pos_ms) < 50:
                    # Add audio up to this point
                    new_audio += audio[prev_end:orig_end_ms]
                    # Add the pause
                    new_audio += AudioSegment.silent(duration=pause_ms)
                    prev_end = orig_end_ms
                    cumulative_shift_ms += pause_ms
                    insert_idx += 1
        
        # Add remaining audio
        if prev_end < len(audio):
            new_audio += audio[prev_end:]
        
        new_audio.export(audio_path, format='wav' if audio_path.endswith('.wav') else 'mp3')
        new_duration = len(new_audio) / 1000.0
        
        print(f"   \u23f8\ufe0f Context pauses injected: {len(insertions)} pauses "
              f"({sum(p[1] for p in insertions)}ms total). "
              f"Duration: {len(audio)/1000.0:.2f}s \u2192 {new_duration:.2f}s")
        
        return new_duration, new_timestamps
        
    except Exception as e:
        print(f"   \u26a0\ufe0f Context pause injection failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            return len(AudioSegment.from_file(audio_path)) / 1000.0, word_timestamps
        except Exception:
            return 0, word_timestamps


def apply_audio_pacing(audio_path, word_timestamps):
    """
    Phase 2 Rule 2: Prevent coda accordioning & track clipping.
    Apply post-processing to audio file to ensure proper trailing consonants
    and add safety tail.
    
    Args:
        audio_path: Path to the generated audio file
        word_timestamps: List of word timestamp dicts
        
    Returns:
        Tuple of (new_duration, updated_timestamps)
    """
    from pydub import AudioSegment
    
    audio = AudioSegment.from_file(audio_path)
    
    # 1. Ensure final trailing consonants are fully rendered
    # The TTS sometimes cuts off the last word's ending consonants
    # We add a 100ms silence tail to prevent word-swallowing
    SAFETY_TAIL_MS = 100
    silence_tail = AudioSegment.silent(duration=SAFETY_TAIL_MS)
    audio = audio + silence_tail
    
    # 2. Ensure minimum silence between sentences (150-250ms)
    # This is handled at text level via _enforce_phrasing_pauses, 
    # but we can also enforce at audio level if timestamps available
    
    if word_timestamps and len(word_timestamps) > 1:
        # Check for sentence-ending words and ensure minimum gap after them
        sentence_end_words = []
        for i, wt in enumerate(word_timestamps):
            word = wt['word'].strip().rstrip('.,!?')
            if word and word[-1] in '.!?':
                sentence_end_words.append(i)
        
        # If we have sentence boundaries, we could insert silence
        # But this is complex with pydub - better handled at TTS text level
    
    # 3. Normalize to prevent clipping
    audio = audio.normalize(headroom=0.1)
    
    # 4. Fade in/out to prevent clicks (5ms each)
    audio = audio.fade_in(5).fade_out(5)
    
    # Export back
    audio.export(audio_path, format="wav" if audio_path.endswith(".wav") else "mp3")
    
    new_duration = len(audio) / 1000.0
    
    # Update timestamps for the added safety tail
    updated_timestamps = word_timestamps.copy()
    # The safety tail is at the end, so no timestamp shifts needed for words
    
    print(f"   🔊 Audio pacing applied: +{SAFETY_TAIL_MS}ms safety tail, normalized, fade in/out. Duration: {new_duration:.2f}s")
    
    return new_duration, updated_timestamps

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
    from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
    if not ELEVENLABS_API_KEY:
        print("   ✗ ElevenLabs API Key missing.")
        return None, 0, []
    
    try:
        import requests
        # Voice ID for VJ-like tech persona or custom voice ID from config
        VOICE_ID = ELEVENLABS_VOICE_ID if ELEVENLABS_VOICE_ID else "EXAVITQu4vr4xnSDxMaL"
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
                "stability": 0.3,
                "similarity_boost": 0.85,
                "style": 0.3,
                "use_speaker_boost": True
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
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

class AudioAuditEngine:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def _call_gemini_audio(self, audio_path, script_text):
        try:
            # Upload the audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Requesting a technical audit of the audio
            prompt = f"""AUDIT TASK: Compare this generated AI voiceover with the following technical script.
            
            TECHNICAL SCRIPT:
            {script_text}
            
            CRITERIA:
            1. Correct Pronunciation: Specifically check 'tiktoken', 'GGUF', 'vLLM', 'quantization', 'inference'.
            2. Pacing: Is it too fast for complex concepts?
            3. Persona: Does it sound like an authoritative Staff Engineer or a robotic news anchor?
            
            Return ONLY a JSON object:
            {{
              "score": 0.0-10.0,
              "mispronunciations": ["word1", "word2"],
              "critique": "Draft improvements here",
              "fix_hints": {{"word": "new_phonetic_spelling"}}
            }}"""

            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(data=audio_data, mime_type='audio/wav'),
                    prompt
                ]
            )
            raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"⚠️ [AUDIO LOOP] Audit failed: {e}")
            return None

def generate_voiceover(text, custom_phonetic_map=None, api_key=None):
    """
    Agentic Loop for Voiceover: Plan -> Act -> Observe -> Critique -> Refine
    """
    original_raw_text = text
    current_phonetic_map = custom_phonetic_map.copy() if custom_phonetic_map else {}
    
    today = datetime.now().strftime("%Y-%m-%d")
    mp3_path = os.path.join(OUTPUT_DIR, f"audio_{today}.mp3")
    
    best_path, best_dur, best_ts = None, 0, []
    
    iterations = 0
    max_iters = 1 # Start with 1 audit pass for production safety
    
    while iterations <= max_iters:
        # 1. ACT: Generate Audio
        text_to_speak = clean_tts_text(text, phonetic=True, custom_phonetic_map=current_phonetic_map)
        # Pre-TTS symbol sanitization to prevent glitchy artifacts
        text_to_speak = _sanitize_tts_symbols(text_to_speak)
        print(f"🎙️ [AUDIO LOOP] Act: Generating iteration {iterations}...")
        
        path, dur, word_timestamps = None, 0, []
        try:
            # First Priority: ElevenLabs Cloned Voice
            path, dur, word_timestamps = _generate_elevenlabs(text_to_speak, mp3_path)
        except Exception as e:
            print(f"❌ ElevenLabs failed: {e}")
            
        if not path:
            # Second Priority: Local F5-TTS Voice Cloning (if GPU and package are available)
            try:
                import torch
                has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
                if has_gpu:
                    print("🎙️ ElevenLabs failed. Attempting local GPU F5-TTS fallback...")
                    path, dur, word_timestamps = _generate_f5_clone(text_to_speak, mp3_path)
                else:
                    print("   Local GPU not found. Skipping F5-TTS fallback.")
            except Exception as f5_err:
                print(f"❌ F5-TTS voice cloning failed: {f5_err}")
                path = None

        if not path:
            # Third Priority: Edge-TTS
            try:
                path, dur, word_timestamps = _generate_edge_tts(text_to_speak, mp3_path)
            except Exception as edge_err:
                print(f"❌ Edge-TTS failed: {edge_err}")

        if not path: break
        
        # Post-process for alignment check
        if word_timestamps:
            word_timestamps = restore_original_words(word_timestamps, original_raw_text, custom_phonetic_map=current_phonetic_map)
        if path and word_timestamps:
            dur, word_timestamps = trim_audio_silence(path, word_timestamps)
            dur, word_timestamps = optimize_audio_gaps(path, word_timestamps)
            
            # Phase 2: Apply audio pacing & clipping management
            dur, word_timestamps = apply_audio_pacing(path, word_timestamps)
            
            # Phase 3: Inject context-shift pauses for natural breathing room
            dur, word_timestamps = _inject_context_pauses(path, word_timestamps)

        # 2. OBSERVE & CRITIQUE
        if api_key and iterations < max_iters:
            auditor = AudioAuditEngine(api_key)
            feedback = auditor._call_gemini_audio(path, original_raw_text)
            
            if feedback and feedback.get("score", 0) < 9.0:
                score = feedback.get("score", 0)
                fixes = feedback.get("fix_hints", {})
                print(f"🔄 [AUDIO LOOP] Quality: {score}/10. Refining based on: {feedback.get('mispronunciations', [])}")
                
                # 3. REFINE
                if fixes:
                    current_phonetic_map.update(fixes)
                iterations += 1
                continue
            else:
                score_display = feedback.get('score', 'N/A') if feedback else 'N/A'
                print(f"⭐ [AUDIO LOOP] Quality Score: {score_display}/10. Ready.")
        
        best_path, best_dur, best_ts = path, dur, word_timestamps
        break

    return best_path, best_dur, best_ts
