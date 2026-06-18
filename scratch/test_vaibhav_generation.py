import os
import sys

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_script import pick_and_generate_script

def test_vaibhav_generation():
    print("Testing Vaibhav Sisinty Style Script Generation...")
    
    mock_articles = [
        {
            "title": "Bolt.new is a viral AI web development environment that runs entirely in the browser",
            "description": "StackBlitz launched Bolt.new, a web development tool that lets developers build, run, and deploy fullstack web apps directly in the browser using Claude 3.5 Sonnet. It handles package installation, server startup, and deployment with a single prompt.",
            "url": "https://stackblitz.com/blog/introducing-bolt-new-ai-web-dev",
            "source": {"name": "StackBlitz Blog"},
            "publishedAt": "2026-06-18T10:00:00Z",
            "type": "tools"
        }
    ]
    
    # Run the generator with topic_type="vaibhav"
    script_data = pick_and_generate_script(
        articles=mock_articles,
        topic_type="vaibhav"
    )
    
    if not script_data:
        print("❌ Script generation failed!")
        return
        
    print("\n✅ Script generation succeeded!")
    print("\n--- Output Details ---")
    print(f"Title: {script_data.get('title')}")
    print(f"Sub-category: {script_data.get('sub_category')}")
    print(f"Breaking News Level: {script_data.get('breaking_news_level')}")
    print(f"Original News URL: {script_data.get('original_news_url')}")
    print(f"Use Case Evidence URL: {script_data.get('use_case_evidence_url')}")
    print(f"Comment Hook: {script_data.get('comment_hook')}")
    print(f"Keywords: {script_data.get('keywords')}")
    print(f"Hashtags: {script_data.get('hashtags')}")
    
    script_text = script_data.get('script', '')
    word_count = len(script_text.split())
    print(f"\n--- Unified Script ({word_count} words) ---")
    print(script_text)
    
    print("\n--- Timing Structure Check ---")
    print(f"Hook (<15 words): {script_data.get('hook_script')}")
    print(f"Problem (15-20 words): {script_data.get('problem_context')}")
    print(f"Solution (50-60 words): {script_data.get('solution_tech')}")
    print(f"Proof/Result (30-40 words): {script_data.get('retention_loop')}")
    print(f"CTA (10-15 words): {script_data.get('outro_cta')}")
    
    print("\n--- Subtitle Chunks (First 3) ---")
    subtitles = script_data.get('subtitle_chunks', [])
    for sub in subtitles[:3]:
        print(f"  Chunk {sub.get('chunk_id')} ({sub.get('start')}s - {sub.get('end')}s): '{sub.get('text')}'")
        print(f"    Visual Prompt: {sub.get('nano_visual_prompt')}")
        print(f"    Visual Type: {sub.get('visual_type')}")
        
    print("\n--- Key Entities ---")
    print(script_data.get('key_entities'))
    
if __name__ == "__main__":
    test_vaibhav_generation()
