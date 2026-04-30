import os
import stable_whisper
import json
from config import ENABLE_HORMOZI_STYLING

def transcribe_and_segment(audio_path, original_text=None):
    """
    High-fidelity transcription for Hormozi-style word-by-word captions.
    Uses stable-ts for precise word boundaries.
    """
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found")
        return []
    
    print(f"🎯 Transcription Pipeline: Processing {os.path.basename(audio_path)}...")
    model = stable_whisper.load_model('base')
    
    # Use align if we have original text for better accuracy, otherwise transcribe
    if original_text:
        result = model.align(audio_path, original_text, language='en')
    else:
        result = model.transcribe(audio_path, language='en')
        
    word_timestamps = []
    for segment in result.segments:
        for word in segment.words:
            clean_word = word.word.strip()
            if clean_word:
                word_timestamps.append({
                    "word": clean_word,
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "is_power_word": clean_word.isupper() # Simple heuristic for Hormozi highlighting
                })
                
    print(f"✅ {len(word_timestamps)} word-level timestamps extracted for quick-flash captions.")
    return word_timestamps

if __name__ == "__main__":
    # Test script for local verification
    test_wav = os.path.join(os.getcwd(), "vj.wav")
    if os.path.exists(test_wav):
        ts = transcribe_and_segment(test_wav)
        print(f"Sample word: {ts[0] if ts else 'None'}")
