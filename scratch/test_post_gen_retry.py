import os
import sys

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemini_script

def test_post_gen_retry():
    print("🧪 Running Pure Unit Test for Post-Generation Retry Loop...")
    
    # Track calls and simulate return values
    call_records = []
    
    def mock_attempt(articles=None, extra_instruction="", forced_article=None, topic_type="research", failed_topics=None, target_country="US", recent_history=None, recent_titles=None):
        if failed_topics is None:
            failed_topics = []
        call_records.append(dict(failed_topics=list(failed_topics)))
        
        # Simulate a duplicate failure for the first call
        if len(call_records) == 1:
            print("🧠 [TEST MOCK] Attempt 1: Simulating post-generation uniqueness failure.")
            failed_topics.append("Duplicate Topic A")
            return None
        # Simulate success for the second call
        else:
            print("🧠 [TEST MOCK] Attempt 2: Simulating post-generation uniqueness success.")
            return {"title": "Unique Topic B", "script": "Script content for Topic B"}
            
    # Mock the helper attempt function
    original_attempt = gemini_script._pick_and_generate_script_attempt
    gemini_script._pick_and_generate_script_attempt = mock_attempt
    
    try:
        # Run pick_and_generate_script
        failed_topics = []
        res = gemini_script.pick_and_generate_script(
            articles=[],
            extra_instruction="test",
            forced_article=None,
            topic_type="research",
            failed_topics=failed_topics,
            target_country="US"
        )
        
        print("\n--- Verifying Asserts (Success Case) ---")
        print(f"Total Attempts Called: {len(call_records)}")
        print(f"Final failed_topics: {failed_topics}")
        print(f"Result: {res}")
        
        assert len(call_records) == 2, f"Expected 2 attempts, got {len(call_records)}"
        assert "Duplicate Topic A" in failed_topics, "Expected Duplicate Topic A to be in failed_topics"
        assert res is not None and res["title"] == "Unique Topic B", "Expected unique Topic B script data"
        print("✅ Success Case Passed!")
        
    finally:
        gemini_script._pick_and_generate_script_attempt = original_attempt

def test_post_gen_retry_terminal_failure():
    print("\n🧪 Running Pure Unit Test for Post-Generation Terminal Failure...")
    
    call_records = []
    
    def mock_attempt_fail_always(articles=None, extra_instruction="", forced_article=None, topic_type="research", failed_topics=None, target_country="US", recent_history=None, recent_titles=None):
        if failed_topics is None:
            failed_topics = []
        call_records.append(dict(failed_topics=list(failed_topics)))
        
        # Always simulate uniqueness failure
        print(f"🧠 [TEST MOCK] Attempt {len(call_records)}: Simulating post-generation uniqueness failure.")
        failed_topics.append(f"Duplicate Topic {len(call_records)}")
        return None
            
    original_attempt = gemini_script._pick_and_generate_script_attempt
    gemini_script._pick_and_generate_script_attempt = mock_attempt_fail_always
    
    try:
        failed_topics = []
        res = gemini_script.pick_and_generate_script(
            articles=[],
            extra_instruction="test",
            forced_article=None,
            topic_type="research",
            failed_topics=failed_topics,
            target_country="US"
        )
        
        print("\n--- Verifying Asserts (Terminal Failure Case) ---")
        print(f"Total Attempts Called: {len(call_records)}")
        print(f"Final failed_topics: {failed_topics}")
        print(f"Result: {res}")
        
        assert len(call_records) == 2, f"Expected exactly 2 attempts total (capped), got {len(call_records)}"
        assert "Duplicate Topic 1" in failed_topics, "Expected Duplicate Topic 1 to be in failed_topics"
        assert "Duplicate Topic 2" in failed_topics, "Expected Duplicate Topic 2 to be in failed_topics"
        assert res is None, "Expected return value to be None on terminal failure"
        print("✅ Terminal Failure Case Passed!")
        
    finally:
        gemini_script._pick_and_generate_script_attempt = original_attempt

if __name__ == "__main__":
    test_post_gen_retry()
    test_post_gen_retry_terminal_failure()
