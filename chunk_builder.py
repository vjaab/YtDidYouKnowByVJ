"""
chunk_builder.py — Groups word timestamps into visual chunks of 4-6 words.

Rules:
  - Each chunk is 3-6 seconds long
  - Never shorter than 2 seconds
  - Never longer than 7 seconds
  - Prefer splitting at punctuation (natural phrase boundaries)
"""

import re

MIN_DURATION  = 2.0
MAX_DURATION  = 7.0
TARGET_WORDS  = 5  # ideal words per chunk


def _has_sentence_end(word):
    """True if the word ends with sentence-ending punctuation."""
    return bool(re.search(r'[.!?]$', word.strip()))


def _has_phrase_break(word):
    """True if the word ends with a comma or dash (natural pause)."""
    return bool(re.search(r'[,;:\-]$', word.strip()))


def build_chunks(word_timestamps):
    """
    Groups word timestamps into visual chunks.

    Input:
        word_timestamps: [{"word": str, "start": float, "end": float}, ...]
    Returns:
        chunks: [
            {
                "chunk_id":  int,
                "text":      str,
                "words":     [{"word", "start", "end"}, ...],
                "start":     float,
                "end":       float,
                "duration":  float,
            },
            ...
        ]
    """
    if not word_timestamps:
        return []

    chunks = []
    current_words = []

    for i, word_data in enumerate(word_timestamps):
        current_words.append(word_data)

        word_count = len(current_words)
        current_start = current_words[0]["start"]
        current_end   = word_data["end"]
        current_dur   = current_end - current_start

        is_last_word = (i == len(word_timestamps) - 1)

        # Decide whether to close this chunk
        should_close = False

        if is_last_word:
            should_close = True
        elif current_dur >= MAX_DURATION:
            should_close = True
        elif word_count >= TARGET_WORDS and current_dur >= MIN_DURATION:
            # Good size — prefer closing at punctuation
            if _has_sentence_end(word_data["word"]):
                should_close = True
            elif _has_phrase_break(word_data["word"]):
                should_close = True
            elif word_count >= TARGET_WORDS + 2:
                # Force close if we're 2 words over target
                should_close = True

        if should_close:
            # Merge too-short chunks into previous if possible
            if current_dur < MIN_DURATION and chunks:
                chunks[-1]["words"].extend(current_words)
                chunks[-1]["text"]     = " ".join(w["word"] for w in chunks[-1]["words"])
                chunks[-1]["end"]      = current_end
                chunks[-1]["duration"] = current_end - chunks[-1]["start"]
            else:
                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "text":     " ".join(w["word"] for w in current_words),
                    "words":    list(current_words),
                    "start":    current_start,
                    "end":      current_end,
                    "duration": current_dur,
                })
            current_words = []

    # Fix last chunk end to audio end (if tiny rounding diff)
    if chunks:
        last = chunks[-1]
        if last["duration"] < 0.5 and len(chunks) > 1:
            # Merge into previous
            prev = chunks[-2]
            prev["words"].extend(last["words"])
            prev["text"]     = " ".join(w["word"] for w in prev["words"])
            prev["end"]      = last["end"]
            prev["duration"] = last["end"] - prev["start"]
            chunks.pop()

    # Re-index
    for idx, ch in enumerate(chunks):
        ch["chunk_id"] = idx + 1

    return chunks


def redistribute_to_audio_duration(chunks, audio_duration):
    """
    Ensures the last chunk's end time equals audio_duration exactly.
    Redistributes any gap proportionally.
    """
    if not chunks:
        return chunks

    total = sum(c["duration"] for c in chunks)
    if abs(total - audio_duration) < 0.1:
        chunks[-1]["end"]      = audio_duration
        chunks[-1]["duration"] = audio_duration - chunks[-1]["start"]
        return chunks

    # Proportional rescaling of all chunk boundaries
    scale = audio_duration / total if total > 0 else 1.0
    current = 0.0
    for ch in chunks:
        new_dur  = ch["duration"] * scale
        ch["start"]    = round(current, 3)
        ch["end"]      = round(current + new_dur, 3)
        ch["duration"] = round(new_dur, 3)
        for w in ch["words"]:
            w["start"] = min(w["start"] * scale, ch["end"] - 0.01)
            w["end"]   = min(w["end"]   * scale, ch["end"])
        current += new_dur

    chunks[-1]["end"]      = audio_duration
    chunks[-1]["duration"] = audio_duration - chunks[-1]["start"]
    return chunks
