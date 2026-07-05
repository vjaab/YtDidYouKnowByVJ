import os
import sys

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemini_script
from gemini_script import MultiAgentGenerationEngine

def test_duplicate_retry_pure():
    print("🧪 Running Pure Unit Test for Selector Retry-on-Duplicate...")
    
    # 1. Setup mock check_story_uniqueness
    # It rejects Bolt.new and accepts anything else.
    def mock_uniqueness_check(title, new_headline=None, new_keywords=None, new_url=None, tracker_file=None):
        headline_str = str(title or new_headline or "").lower()
        url_str = str(new_url or "").lower()
        if "bolt" in headline_str or "bolt" in url_str:
            print(f"🔒 [TEST MOCK] Rejected Bolt.new (Duplicate).")
            return False, "Simulated duplicate error for Bolt.new"
        print(f"🔑 [TEST MOCK] Approved unique story: '{headline_str}'")
        return True, "Unique"
        
    gemini_script.check_story_uniqueness = mock_uniqueness_check
    
    # 2. Setup mock MultiAgentGenerationEngine._call_gemini
    # We want to simulate the Selector Agent returning "Bolt.new" first, and "CleanCoder" second.
    # We also mock context sharpening and downstream calls to return mock success data
    # so we don't hit any actual LLM.
    call_counts = {"selector": 0, "sharpener": 0, "researcher": 0}
    
    def mock_call_gemini(self_obj, prompt, model=None):
        prompt_lower = str(prompt).lower()
        if "selector agent" in prompt_lower:
            call_counts["selector"] += 1
            if call_counts["selector"] == 1:
                print("🧠 [TEST MOCK] Selector Agent call 1: Selecting Bolt.new")
                return {
                    "selected_headline": "Bolt.new is a viral AI web dev tool",
                    "selected_url": "https://bolt.new",
                    "keywords": ["bolt", "ai", "browser"],
                    "reason": "Very trending"
                }
            else:
                print("🧠 [TEST MOCK] Selector Agent call 2: Selecting CleanCoder")
                return {
                    "selected_headline": "CleanCoder is a lint fixer agent",
                    "selected_url": "https://github.com/cleancoder",
                    "keywords": ["clean", "coder", "lint"],
                    "reason": "Second best unique option"
                }
        elif "context sharpener" in prompt_lower:
            call_counts["sharpener"] += 1
            print("🧠 [TEST MOCK] Context Sharpener call")
            return {"core_narrative": "CleanCoder is awesome."}
        elif "research agent" in prompt_lower:
            call_counts["researcher"] += 1
            print("🧠 [TEST MOCK] Research Agent call")
            return {"research": "facts about CleanCoder"}
        elif "hook agent" in prompt_lower:
            print("🧠 [TEST MOCK] Hook Agent call")
            return {"hooks": [{"text": "Mock hook text", "curiosity_score": 10, "emotional_trigger_score": 10, "swipe_stop_score": 10}]}
        elif "narrative agent" in prompt_lower or "fact script generator" in prompt_lower:
            print("🧠 [TEST MOCK] Fact Script Generator call")
            return {"narrative": "Mock narrative"}
        elif "retention optimizer" in prompt_lower:
            print("🧠 [TEST MOCK] Retention Optimizer call")
            return {"optimized_script": "Mock optimized script"}
        elif "retention scientist" in prompt_lower:
            print("🧠 [TEST MOCK] Retention Scientist call")
            return {"retention_enhanced_script": "Mock enhanced script", "retention_map": {"curiosity_gap_ratio": 0.7, "open_loops": [1], "pattern_interrupts": [2]}}
        elif "humanizer" in prompt_lower:
            print("🧠 [TEST MOCK] Humanizer Agent call")
            return {"script": "Mock final humanized script", "title": "CleanCoder: AI Lint Fixer", "original_news_headline": "CleanCoder is a lint fixer agent", "original_news_url": "https://github.com/cleancoder"}
        
        # Fallback default response
        return {"script": "Mock script", "title": "Mock title"}

    original_call = MultiAgentGenerationEngine._call_gemini
    MultiAgentGenerationEngine._call_gemini = mock_call_gemini
    
    # 3. Instantiate engine
    failed_topics = []
    engine = MultiAgentGenerationEngine(
        client=None,
        context="Mock Context details...",
        slot="Slot A",
        category="Tech",
        strategy_enhancement="",
        is_longform=False,
        raw_articles=None,
        topic_type="tools",
        failed_topics=failed_topics
    )
    
    # 4. Execute the Selector Agent (which should trigger the loop and succeed on CleanCoder)
    # We also mock the rest of the execute method by stubbing the agents if needed, but wait!
    # Let's check if execute goes all the way to the end and calls other agents.
    # Yes, execute calls: Selector, Sharpener, Researcher, Hook, Writer, Humanizer etc.
    # Since we mocked _call_gemini to handle Selector, Sharpener, Researcher, and default to mock dicts,
    # it should complete execute successfully!
    script_data = engine.execute("selection instruction", "prompt requirements")
    
    # Restore original _call_gemini
    MultiAgentGenerationEngine._call_gemini = original_call
    
    # 5. Asserts
    print("\n--- Verifying Asserts ---")
    print(f"Selector Call Count: {call_counts['selector']}")
    print(f"Failed Topics List: {failed_topics}")
    
    assert call_counts["selector"] == 2, f"Expected 2 selector calls, got {call_counts['selector']}"
    assert any("bolt" in str(x).lower() for x in failed_topics), "Expected Bolt.new to be in failed_topics"
    assert script_data is not None, "Expected execute to return script_data successfully"
    
    print("✅ Pure Unit Test Passed successfully!")

if __name__ == "__main__":
    test_duplicate_retry_pure()
