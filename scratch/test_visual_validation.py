import sys
import os
from PIL import Image, ImageDraw
from unittest.mock import MagicMock, patch

# Add workspace directory to path to allow importing screenshot_gen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from screenshot_gen import check_screenshot_validity, capture_article_screenshot

def create_mock_captcha_image(output_path):
    img = Image.new("RGB", (1080, 1920), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((100, 300), "cloudflare", fill=(246, 130, 31))
    draw.text((100, 400), "Checking if the site connection is secure", fill=(0, 0, 0))
    draw.text((100, 500), "Verify you are human", fill=(0, 0, 0))
    img.save(output_path)

def run_tests():
    print("=== RUNNING SCREENSHOT VALIDATION MOCK TESTS ===")
    
    dummy_img = "dummy_screenshot.png"
    if not os.path.exists(dummy_img):
        img = Image.new("RGB", (1080, 1920), (10, 10, 250))
        img.save(dummy_img)
        
    mock_captcha_path = "scratch/mock_captcha.png"
    create_mock_captcha_image(mock_captcha_path)

    # 1. Test standard image (should return True when API returns 'NO')
    print("\n--- Test 1: Standard Image (API returns NO) ---")
    mock_response = MagicMock()
    mock_response.text = "NO"
    
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        is_valid = check_screenshot_validity(dummy_img)
        print(f"Result (expected True): {is_valid}")
        assert is_valid is True, "Test 1 Failed: Standard image should be valid."
        print("✅ Test 1 Passed.")

    # 2. Test CAPTCHA image (should return False when API returns 'YES')
    print("\n--- Test 2: CAPTCHA Image (API returns YES) ---")
    mock_response_yes = MagicMock()
    mock_response_yes.text = "YES"
    
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response_yes
        mock_client_class.return_value = mock_client
        
        is_valid_captcha = check_screenshot_validity(mock_captcha_path)
        print(f"Result (expected False): {is_valid_captcha}")
        assert is_valid_captcha is False, "Test 2 Failed: CAPTCHA should be invalid."
        print("✅ Test 2 Passed.")

    # 3. Test API Rate Limit Fallback (should return True as safe default when API raises exception)
    print("\n--- Test 3: API Error / Quota Exhausted Safe Default ---")
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 Resource Exhausted")
        mock_client_class.return_value = mock_client
        
        is_valid_fallback = check_screenshot_validity(mock_captcha_path)
        print(f"Result on error fallback (expected True): {is_valid_fallback}")
        assert is_valid_fallback is True, "Test 3 Failed: Should fallback to True on rate limit."
        print("✅ Test 3 Passed.")

    # Clean up mock file
    if os.path.exists(mock_captcha_path):
        os.remove(mock_captcha_path)
        
    # 4. Test real browser capture and validation (using example.com)
    print("\n--- Test 4: Real capture_article_screenshot integration ---")
    test_url = "https://example.com"
    test_output = "scratch/test_example_com.png"
    
    # We patch check_screenshot_validity to return True to check execution path
    with patch("screenshot_gen.check_screenshot_validity", return_value=True) as mock_val:
        captured_path = capture_article_screenshot(test_url, "../scratch/test_example_com.png")
        print(f"Captured path: {captured_path}")
        if captured_path:
            assert os.path.exists(captured_path), "File should be written to path."
            os.remove(captured_path)
            print("✅ Test 4 Passed.")
        else:
            print("❌ Test 4 Failed: Could not capture screenshot of example.com")

if __name__ == "__main__":
    run_tests()
