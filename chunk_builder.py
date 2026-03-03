"""
chunk_builder.py — Regroups visual chunks using real audio timestamps (stable-ts / Edge boundaries) and Gemini's subtitle_chunks template.
"""

import re

def build_chunks(word_timestamps, subtitle_chunks):
    """
    Groups word timestamps into visual chunks based on the Gemini subtitle_chunks template.
    Overrides Gemini's estimated timestamps with the REAL timestamps from audio.
    """
    if not word_timestamps:
        return []
    
    if not subtitle_chunks or len(subtitle_chunks) < 3:
        # Fallback to audio-based chunks if Gemini gave too few subtitle_chunks
        if subtitle_chunks and len(subtitle_chunks) < 3:
            print(f"WARNING: Gemini only produced {len(subtitle_chunks)} subtitle_chunks. Falling back to audio-based chunking for proper subtitles.")
        return _fallback_build_chunks(word_timestamps)

    def strip_punc(s):
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

    word_idx = 0
    num_words = len(word_timestamps)
    final_chunks = []

    for gc in subtitle_chunks:
        chunk_text = gc.get("text", "")
        # Calculate how many alphanumeric characters are in this chunk
        target_len = sum(len(strip_punc(w)) for w in chunk_text.split())
        
        chunk_words = []
        current_len = 0
        
        # Consume words until we've matched the approximate length of this chunk
        while word_idx < num_words and current_len < target_len * 0.85: # 85% tolerance due to TTS skips
            wdata = word_timestamps[word_idx]
            chunk_words.append(wdata)
            current_len += len(strip_punc(wdata["word"]))
            word_idx += 1
            
        if not chunk_words:
            # If no words matched, use the Gemini provided timestamps as a deep fallback
            # but usually shouldn't happen
            continue
            
        final_gc = dict(gc)  # Copy all Gemini properties (pexels_*, highlight_word, etc.)
        final_gc["words"] = chunk_words
        
        # Override with REAL timestamps
        final_gc["start"] = chunk_words[0]["start"]
        final_gc["end"] = chunk_words[-1]["end"]
        final_gc["duration"] = final_gc["end"] - final_gc["start"]
        
        final_chunks.append(final_gc)

    # Now enforce the non-overlap rule: chunk[i].end <= chunk[i+1].start
    for i in range(len(final_chunks) - 1):
        if final_chunks[i]["end"] > final_chunks[i+1]["start"]:
             # Push next chunk's start to current chunk's end
             final_chunks[i+1]["start"] = final_chunks[i]["end"]
             final_chunks[i+1]["duration"] = max(0.1, final_chunks[i+1]["end"] - final_chunks[i+1]["start"])

    return final_chunks

def _fallback_build_chunks(word_timestamps):
    MIN_DURATION  = 2.0
    MAX_DURATION  = 7.0
    TARGET_WORDS  = 5

    chunks = []
    current_words = []

    for i, word_data in enumerate(word_timestamps):
        current_words.append(word_data)
        word_count = len(current_words)
        current_start = current_words[0]["start"]
        current_end   = word_data["end"]
        current_dur   = current_end - current_start
        is_last_word = (i == len(word_timestamps) - 1)
        should_close = False

        if is_last_word or current_dur >= MAX_DURATION:
            should_close = True
        elif word_count >= TARGET_WORDS and current_dur >= MIN_DURATION:
            if re.search(r'[.!?]$', word_data["word"].strip()) or re.search(r'[,;:\-]$', word_data["word"].strip()) or word_count >= TARGET_WORDS + 2:
                should_close = True

        if should_close:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "text":     " ".join(w["word"] for w in current_words),
                "words":    list(current_words),
                "start":    current_start,
                "end":      current_end,
                "duration": current_dur,
            })
            current_words = []
            
    return chunks

def redistribute_to_audio_duration(chunks, audio_duration):
    if not chunks:
        return chunks

    # Only adjust the last chunk to match audio duration exactly
    last_chunk = chunks[-1]
    last_chunk["end"] = max(last_chunk["end"], audio_duration)
    last_chunk["duration"] = last_chunk["end"] - last_chunk["start"]

    # Verify gap logic one more time
    for i in range(len(chunks) - 1):
        if chunks[i]["end"] > chunks[i+1]["start"]:
             chunks[i+1]["start"] = chunks[i]["end"]
             chunks[i+1]["duration"] = max(0.1, chunks[i+1]["end"] - chunks[i+1]["start"])

    return chunks
