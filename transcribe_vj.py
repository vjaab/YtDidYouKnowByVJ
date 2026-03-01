import os
import stable_whisper

def transcribe():
    path = os.path.join(os.getcwd(), "vj.wav")
    if not os.path.exists(path):
        print("vj.wav not found")
        return
    
    print("Transcribing vj.wav...")
    model = stable_whisper.load_model('base')
    result = model.transcribe(path)
    print("Transcription:")
    print(result.text)

if __name__ == "__main__":
    transcribe()
