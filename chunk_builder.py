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
        return _fallback_build_chunks(word_timestamps, subtitle_chunks=subtitle_chunks)

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
            # If no words matched, we might have a drift issue. 
            # Skip this chunk but keep word_idx where it is.
            continue
            
        final_gc = dict(gc)  # Copy all Gemini properties (pexels_*, highlight_word, etc.)
        final_gc["words"] = chunk_words
        
        # Override with REAL timestamps
        final_gc["start"] = chunk_words[0]["start"]
        final_gc["end"] = chunk_words[-1]["end"]
        final_gc["duration"] = final_gc["end"] - final_gc["start"]
        
        final_chunks.append(final_gc)

    # TRAILING WORD RECOVERY: Append any remaining unaligned words to the last chunk
    if final_chunks and word_idx < num_words:
        remaining_words = word_timestamps[word_idx:]
        if len(final_chunks[-1]["words"]) < 6:
            final_chunks[-1]["words"].extend(remaining_words)
            final_chunks[-1]["text"] += " " + " ".join(w["word"] for w in remaining_words)
            final_chunks[-1]["end"] = remaining_words[-1]["end"]
            final_chunks[-1]["duration"] = final_chunks[-1]["end"] - final_chunks[-1]["start"]
        else:
            new_chunk = dict(final_chunks[-1])
            new_chunk["chunk_id"] = final_chunks[-1]["chunk_id"] + 1
            new_chunk["text"] = " ".join(w["word"] for w in remaining_words)
            new_chunk["words"] = remaining_words
            new_chunk["start"] = remaining_words[0]["start"]
            new_chunk["end"] = remaining_words[-1]["end"]
            new_chunk["duration"] = new_chunk["end"] - new_chunk["start"]
            final_chunks.append(new_chunk)

    # RECENT PRODUCTION FIX: If alignment failed for more than 50% of intended chunks,
    # it means Gemini's template and the audio script diverged too much. 
    # Fallback to pure audio-based chunking to avoid broken visual timing.
    if len(final_chunks) < len(subtitle_chunks) * 0.5:
        print(f"WARNING: Subtitle alignment poor ({len(final_chunks)}/{len(subtitle_chunks)}). Falling back to audio-based chunks.")
        return _fallback_build_chunks(word_timestamps, subtitle_chunks=subtitle_chunks)

    # Now enforce the non-overlap rule: chunk[i].end <= chunk[i+1].start
    for i in range(len(final_chunks) - 1):
        if final_chunks[i]["end"] > final_chunks[i+1]["start"]:
             # Push next chunk's start to current chunk's end
             final_chunks[i+1]["start"] = final_chunks[i]["end"]
             final_chunks[i+1]["duration"] = max(0.1, final_chunks[i+1]["end"] - final_chunks[i+1]["start"])

    return final_chunks

def _fallback_build_chunks(word_timestamps, subtitle_chunks=None):
    MIN_DURATION  = 1.0
    MAX_DURATION  = 2.5
    TARGET_WORDS  = 2

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
            
    # Map visual metadata from original subtitle_chunks based on start time/index ratio
    if subtitle_chunks:
        num_sc = len(subtitle_chunks)
        num_fc = len(chunks)
        for idx, fc in enumerate(chunks):
            fc_start = fc["start"]
            best_sc = None
            min_dist = float('inf')
            
            # 1. Try to find by timestamp overlap
            for sc in subtitle_chunks:
                sc_start = sc.get("start")
                sc_end = sc.get("end")
                if sc_start is not None and sc_end is not None:
                    if sc_start <= fc_start <= sc_end:
                        best_sc = sc
                        break
                    dist = abs(sc_start - fc_start)
                    if dist < min_dist:
                        min_dist = dist
                        best_sc = sc
            
            # 2. If no close timestamp match, map by proportional index ratio
            if best_sc is None or min_dist > 10.0:
                ratio_idx = min(num_sc - 1, int(idx / num_fc * num_sc))
                best_sc = subtitle_chunks[ratio_idx]
                
            if best_sc:
                # Copy visual metadata keys
                for key in ["pexels_primary", "pexels_fallback", "nano_visual_prompt", "scene_objective", "visual_type"]:
                    if key in best_sc:
                        fc[key] = best_sc[key]

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
