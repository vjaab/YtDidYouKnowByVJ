import requests
import urllib.parse
import os

def test_pollinations():
    prompt = "a coding robot looking at a laptop screen, vertical 9:16, cinematic, photorealistic"
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1080&height=1920&nologo=true&private=true"
    print(f"Fetching: {url}")
    try:
        resp = requests.get(url, timeout=30)
        print(f"Status Code: {resp.status_code}")
        print(f"Headers: {resp.headers}")
        if resp.status_code == 200:
            with open("scratch_pollinations.jpg", "wb") as f:
                f.write(resp.content)
            print("Success! Image saved to scratch_pollinations.jpg")
        else:
            print("Failed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pollinations()
