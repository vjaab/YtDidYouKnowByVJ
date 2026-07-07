import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import time

# Add workspace directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemini_script_longform
from gemini_script_longform import (
    get_retry_after,
    execute_with_timeout,
    is_groq_model_near_limit,
    _update_groq_token_usage,
    call_fallback_model
)

class TestLongformFallbackChain(unittest.TestCase):

    def test_get_retry_after_parsing(self):
        """Verify get_retry_after correctly parses different rate limit error formats."""
        # 1. Test standard string parsing
        e1 = Exception("Rate limit hit. Please retry after 15s. Thanks!")
        self.assertEqual(get_retry_after(e1), 15.0)

        e2 = Exception("Resource exhausted: backoff delay of 45 seconds enforced.")
        self.assertEqual(get_retry_after(e2), 45.0)

        e3 = Exception("Temporary overload: wait 500ms before retrying.")
        self.assertEqual(get_retry_after(e3), 3.0)  # Capped at minimum 3.0

        # 2. Test response header parsing
        class MockResponse:
            def __init__(self, headers):
                self.headers = headers

        mock_resp = MockResponse({"Retry-After": "30"})
        e4 = Exception("HTTP Rate Limit")
        e4.response = mock_resp
        self.assertEqual(get_retry_after(e4), 30.0)

        # 3. Fallback when no info is present
        e5 = Exception("Generic connection reset")
        self.assertEqual(get_retry_after(e5), 3.0)

    def test_execute_with_timeout(self):
        """Verify execute_with_timeout halts control flow after timeout is reached."""
        def slow_function():
            time.sleep(2.0)
            return "completed"

        def fast_function():
            return "completed"

        # Fast function should return the correct value
        res_fast = execute_with_timeout(fast_function, 0.5)
        self.assertEqual(res_fast, "completed")

        # Slow function should timeout and return None
        start_time = time.time()
        res_slow = execute_with_timeout(slow_function, 0.3)
        duration = time.time() - start_time
        
        self.assertIsNone(res_slow)
        self.assertLess(duration, 0.6)  # Control must return quickly

    @patch("gemini_script._load_cache")
    def test_groq_token_usage_check(self, mock_load_cache):
        """Verify that models exceeding 90% TPD are pre-emptively skipped."""
        today_str = time.strftime("%Y-%m-%d")
        
        # Mock cache where qwen model is at 460k / 500k limit (92%)
        mock_load_cache.return_value = {
            "groq_token_usage": {
                "qwen/qwen3-32b": {
                    "tokens": 460000,
                    "date": today_str
                },
                "llama-3.3-70b-versatile": {
                    "tokens": 100000,
                    "date": today_str
                }
            }
        }

        # Qwen should be skipped (92% limit reached)
        self.assertTrue(is_groq_model_near_limit("qwen/qwen3-32b"))
        # Llama should not be skipped (10% limit reached)
        self.assertFalse(is_groq_model_near_limit("llama-3.3-70b-versatile"))

    @patch("requests.post")
    @patch("gemini_script.is_model_exhausted", return_value=False)
    @patch("gemini_script_longform.is_groq_model_near_limit", return_value=False)
    @patch("os.getenv")
    def test_full_chain_exhaustion(self, mock_getenv, mock_near_limit, mock_exhausted, mock_post):
        """Verify call_fallback_model cascades through all providers sequentially and returns None if all fail."""
        # Setup environment keys so all providers are evaluated
        mock_getenv.side_effect = lambda key: "mock_api_key" if "KEY" in key or "TOKEN" in key or "ID" in key else None

        # Mock all API endpoints to return error code 500 to simulate full failure
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        print("Testing full-chain exhaustion fallback...")
        res = call_fallback_model("Generate a viral script.")
        
        # Verify it returns None (all providers failed)
        self.assertIsNone(res)
        
        # Verify it hit the API endpoints for different providers:
        # 1. Groq (llama, qwen, openai-gpt, llama-3.1, etc.)
        # 2. Cloudflare (various models)
        # 3. OpenCode Zen
        # 4. Cerebras
        # 5. OpenAI
        # 6. Anthropic
        # 7. DeepSeek
        # 8. OpenRouter
        # Total requests sent should be >= 10 across all the lists
        self.assertGreaterEqual(mock_post.call_count, 10)

if __name__ == "__main__":
    unittest.main()
