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
                for key in ["pexels_primary", "pexels_fallback", "nano_visual_prompt", "scene_objective", "visual_type", "is_setting_chunk", "has_infographic", "infographic_type", "infographic_data"]:
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


def build_chapter_aware_chunks(word_timestamps, chapters, subtitle_chunks=None, max_beats_per_chapter=6):
    """
    Groups word timestamps into visual chunks using chapter boundaries and visual beats.
    
    This produces ~15-30 total chunks instead of ~200+ sentence-level chunks, which is the
    direct fix for the memory/SIGTERM issue in longform rendering. Each visual beat holds
    8-15 seconds of narration rather than one sentence.
    
    Args:
        word_timestamps: List of {"word": str, "start": float, "end": float}
        chapters: List of chapter dicts from the script, each containing:
            - chapter_number, chapter_title, chapter_text, approx_start_seconds
            - visual_beats: list of {"beat_text": str, "visual_direction": str}
        subtitle_chunks: Optional Gemini subtitle_chunks for visual metadata mapping
        max_beats_per_chapter: Hard cap on visual beats per chapter
    
    Returns:
        List of chunk dicts compatible with the existing visual pipeline
    """
    if not word_timestamps or not chapters:
        print("WARNING: No chapters available for chapter-aware chunking. Falling back to standard chunking.")
        return build_chunks(word_timestamps, subtitle_chunks or [])

    num_words = len(word_timestamps)
    total_audio_dur = word_timestamps[-1]["end"] if word_timestamps else 0
    final_chunks = []
    
    def strip_punc(s):
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

    # Calculate chapter time boundaries from approx_start_seconds
    chapter_boundaries = []
    for i, ch in enumerate(chapters):
        start_sec = ch.get("approx_start_seconds", 0)
        if i + 1 < len(chapters):
            end_sec = chapters[i + 1].get("approx_start_seconds", total_audio_dur)
        else:
            end_sec = total_audio_dur
        chapter_boundaries.append((start_sec, end_sec))

    # Distribute word timestamps into chapters by time
    chapter_words = [[] for _ in chapters]
    for wdata in word_timestamps:
        w_mid = (wdata["start"] + wdata["end"]) / 2
        assigned = False
        for ci, (c_start, c_end) in enumerate(chapter_boundaries):
            if c_start <= w_mid < c_end:
                chapter_words[ci].append(wdata)
                assigned = True
                break
        if not assigned:
            # Assign to last chapter if past all boundaries
            chapter_words[-1].append(wdata)

    # Within each chapter, group words into visual beats
    for ci, ch in enumerate(chapters):
        ch_word_list = chapter_words[ci]
        if not ch_word_list:
            continue

        beats = ch.get("visual_beats", [])
        # Cap beats per chapter
        beats = beats[:max_beats_per_chapter]

        if not beats:
            # No visual beats defined for this chapter, create one big chunk
            chunk = {
                "chunk_id": len(final_chunks) + 1,
                "text": " ".join(w["word"] for w in ch_word_list),
                "words": ch_word_list,
                "start": ch_word_list[0]["start"],
                "end": ch_word_list[-1]["end"],
                "duration": ch_word_list[-1]["end"] - ch_word_list[0]["start"],
                "chapter_number": ch.get("chapter_number", ci + 1),
                "chapter_title": ch.get("chapter_title", f"Chapter {ci + 1}"),
                "nano_visual_prompt": ch.get("visual_beats", [{}])[0].get("visual_direction", "") if ch.get("visual_beats") else "",
            }
            final_chunks.append(chunk)
            continue

        # Distribute chapter words evenly across beats by character count
        total_chars = sum(len(strip_punc(w["word"])) for w in ch_word_list)
        beat_char_targets = []
        for beat in beats:
            beat_text = beat.get("beat_text", "")
            beat_chars = sum(len(strip_punc(w)) for w in beat_text.split())
            beat_char_targets.append(max(beat_chars, 1))

        # Normalize targets to match actual word count
        target_sum = sum(beat_char_targets)
        if target_sum > 0:
            beat_char_targets = [int(t / target_sum * total_chars) for t in beat_char_targets]
            # Ensure at least some characters per beat
            beat_char_targets = [max(t, 5) for t in beat_char_targets]

        word_idx = 0
        for bi, beat in enumerate(beats):
            target_chars = beat_char_targets[bi] if bi < len(beat_char_targets) else total_chars
            beat_words = []
            current_chars = 0

            while word_idx < len(ch_word_list) and current_chars < target_chars * 0.85:
                beat_words.append(ch_word_list[word_idx])
                current_chars += len(strip_punc(ch_word_list[word_idx]["word"]))
                word_idx += 1

            if not beat_words:
                continue

            chunk = {
                "chunk_id": len(final_chunks) + 1,
                "text": " ".join(w["word"] for w in beat_words),
                "words": beat_words,
                "start": beat_words[0]["start"],
                "end": beat_words[-1]["end"],
                "duration": beat_words[-1]["end"] - beat_words[0]["start"],
                "chapter_number": ch.get("chapter_number", ci + 1),
                "chapter_title": ch.get("chapter_title", f"Chapter {ci + 1}"),
                "nano_visual_prompt": beat.get("visual_direction", ""),
                "visual_type": "chapter_beat",
            }
            final_chunks.append(chunk)

        # Append any remaining words from this chapter to the last beat chunk
        if word_idx < len(ch_word_list) and final_chunks:
            remaining = ch_word_list[word_idx:]
            last = final_chunks[-1]
            last["words"].extend(remaining)
            last["text"] += " " + " ".join(w["word"] for w in remaining)
            last["end"] = remaining[-1]["end"]
            last["duration"] = last["end"] - last["start"]

    # Map visual metadata from subtitle_chunks if available
    if subtitle_chunks and final_chunks:
        num_sc = len(subtitle_chunks)
        num_fc = len(final_chunks)
        for idx, fc in enumerate(final_chunks):
            ratio_idx = min(num_sc - 1, int(idx / num_fc * num_sc))
            best_sc = subtitle_chunks[ratio_idx]
            if best_sc:
                for key in ["pexels_primary", "pexels_fallback", "scene_objective",
                            "is_setting_chunk", "has_infographic", "infographic_type", "infographic_data"]:
                    if key in best_sc:
                        fc[key] = best_sc[key]

    # Enforce non-overlap rule
    for i in range(len(final_chunks) - 1):
        if final_chunks[i]["end"] > final_chunks[i + 1]["start"]:
            final_chunks[i + 1]["start"] = final_chunks[i]["end"]
            final_chunks[i + 1]["duration"] = max(0.1, final_chunks[i + 1]["end"] - final_chunks[i + 1]["start"])

    if len(final_chunks) < 3:
        print(f"WARNING: Chapter-aware chunking produced only {len(final_chunks)} chunks. Falling back to standard chunking.")
        return build_chunks(word_timestamps, subtitle_chunks or [])

    print(f"✅ Chapter-aware chunking: {len(final_chunks)} chunks across {len(chapters)} chapters (vs ~{num_words // 2} sentence-level chunks)")
    return final_chunks

