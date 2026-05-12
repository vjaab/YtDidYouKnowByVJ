import os
import time
import random
import argparse
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, INTROS_DIR, OUTROS_DIR, AVATARS_DIR, VEO_MODEL_ID

# ── PROMPT LIBRARY ────────────────────────────────────────────────────────────

INTRO_PROMPTS = [
    "Cinematic shot of a futuristic glass-walled laboratory at sunset, glowing holograms in the background, 4k, hyper-realistic, shallow depth of field.",
    "A dark cyberpunk hacker den with multiple glowing monitors, neon green data streams, industrial aesthetic, 4k.",
    "Minimalist white tech studio with floating geometric shapes, soft professional lighting, high-end production look, 4k.",
    "A quantum data center with pulsing blue fiber optic cables, cold atmospheric lighting, futuristic engineering vibe, 4k."
]

OUTRO_PROMPTS = [
    "A hyper-realistic 3D render of a futuristic smartphone floating in a dark void, the screen glows with a soft light, cinematic slow motion.",
    "Close up of a robotic hand waving goodbye in a high-tech workshop, warm lighting, detailed mechanical parts, 4k.",
    "Abstract digital particles forming a circle and then dispersing, elegant tech motion graphics, deep blue and gold colors, 4k."
]

AVATAR_STYLE_PROMPTS = [
    "Cinematic lighting, warm amber tones, laboratory background, professional depth of field.",
    "Cyberpunk aesthetic, neon purple and cyan lighting, dark industrial background, sharp focus.",
    "Clean corporate tech office, bright natural lighting, professional and high-authority vibe.",
    "Underground bunker style, dramatic high-contrast lighting, cold steel and glowing orange accents."
]

# ── GENERATION ENGINE ─────────────────────────────────────────────────────────

def generate_veo_asset(prompt, output_folder, filename_prefix="veo_asset", aspect_ratio="9:16"):
    """
    Generates a video asset using Google Veo and saves it to the specified folder.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    print(f"🚀 [VEO] Generating asset: {prompt[:50]}...")
    
    try:
        operation = client.models.generate_videos(
            model=VEO_MODEL_ID,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                duration_seconds=5,
            )
        )
        
        # Polling for completion
        start_time = time.time()
        while not operation.done:
            elapsed = time.time() - start_time
            print(f"   ⌛ Waiting for Veo... ({int(elapsed)}s elapsed)")
            time.sleep(15)
            
        result = operation.result
        if result and result.generated_videos:
            video = result.generated_videos[0]
            
            timestamp = int(time.time())
            output_filename = f"{filename_prefix}_{timestamp}.mp4"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save the video
            video.video.save(output_path)
            print(f"✅ [VEO] Asset saved to: {output_path}")
            return output_path
        else:
            print("❌ [VEO] Generation failed: No result returned.")
            return None
            
    except Exception as e:
        print(f"❌ [VEO] API Error: {e}")
        return None

# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate high-end video assets using Google Veo.")
    parser.add_argument("--type", choices=["intro", "outro", "avatar"], required=True, help="Type of asset to generate.")
    parser.add_argument("--count", type=int, default=1, help="Number of assets to generate.")
    
    args = parser.parse_args()
    
    if args.type == "intro":
        prompts = INTRO_PROMPTS
        folder = INTROS_DIR
        prefix = "intro_variation"
    elif args.type == "outro":
        prompts = OUTRO_PROMPTS
        folder = OUTROS_DIR
        prefix = "outro_variation"
    else:
        prompts = AVATAR_STYLE_PROMPTS
        folder = AVATARS_DIR
        prefix = "avatar_base"
        
    for i in range(args.count):
        prompt = random.choice(prompts)
        if args.type == "avatar":
            # For avatars, we want to keep the character consistent
            prompt = f"A cinematic video of a futuristic male AI researcher, {prompt}"
            
        generate_veo_asset(prompt, folder, filename_prefix=prefix)

if __name__ == "__main__":
    main()
