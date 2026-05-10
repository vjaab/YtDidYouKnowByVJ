import os
import requests
import sys
import time
from google import genai
from config import GEMINI_API_KEY, VEO_MODEL_ID, OUTPUT_DIR

def test_veo():
    print("🚀 Starting Google Veo Verification Test...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = "A futuristic 3D holographic interface of a neural network, glowing cyan and obsidian, 4k, cinematic, 9:16 vertical."
    output_path = os.path.join(OUTPUT_DIR, "veo_test_video.mp4")
    
    print(f"🎬 Sending generation request for: '{prompt[:50]}...'")
    try:
        operation = client.models.generate_videos(
            model=VEO_MODEL_ID,
            prompt=prompt,
            config=genai.types.GenerateVideosConfig(
                aspect_ratio="9:16",
            )
        )
        
        print(f"⌛ Operation ID: {operation.name}. Polling for completion...")
        
        attempts = 0
        while not operation.done and attempts < 60:
            time.sleep(10)
            operation = client.operations.get(operation)
            attempts += 1
            print(f"   [{attempts*10}s] Status: {'DONE' if operation.done else 'PROCESSING'}...")

        if operation.error:
            print(f"❌ Operation Failed: {operation.error}")
            return

        print(f"DEBUG: Operation Result: {operation.result}")
        if operation.result and hasattr(operation.result, 'generated_videos'):
            videos = operation.result.generated_videos
            if videos:
                gen_video = videos[0]
                video = gen_video.video if hasattr(gen_video, 'video') else gen_video
                
                if hasattr(video, 'video_bytes') and video.video_bytes:
                    with open(output_path, "wb") as f:
                        f.write(video.video_bytes)
                    print(f"✅ Success! Veo video saved to: {output_path}")
                elif hasattr(video, 'uri'):
                    # If it's a URI, we need to download it
                    print(f"⌛ Downloading video from URI: {video.uri}...")
                    r = requests.get(video.uri, headers={"x-goog-api-key": GEMINI_API_KEY})
                    if r.status_code == 200:
                        with open(output_path, "wb") as f:
                            f.write(r.content)
                        print(f"✅ Success! Veo video downloaded and saved to: {output_path}")
                    else:
                        print(f"❌ Failed to download URI. Status: {r.status_code}")
                else:
                    print(f"❓ Video object found but no bytes/uri: {video}")
            else:
                print("❌ Failure: generated_videos list is empty.")
        else:
            print(f"❌ Failure: operation.result is missing or invalid. Result: {operation.result}")
            
    except Exception as e:
        print(f"❌ Error during Veo generation: {e}")

if __name__ == "__main__":
    test_veo()
