import os
import sys

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_script_longform import generate_longform_script
from main_longform import format_longform_description

def test_vaibhav_longform():
    print("Testing Vaibhav Sisinty Style Long-Form Script Generation...")
    
    mock_articles = [
        {
            "title": "Bolt.new is a viral AI web development environment that runs entirely in the browser",
            "description": "StackBlitz launched Bolt.new, a web development tool that lets developers build, run, and deploy fullstack web apps directly in the browser using Claude 3.5 Sonnet. It handles package installation, server startup, and deployment with a single prompt.",
            "url": "https://stackblitz.com/blog/introducing-bolt-new-ai-web-dev",
            "source": {"name": "StackBlitz Blog"},
            "publishedAt": "2026-06-18T10:00:00Z",
            "type": "tools"
        },
        {
            "title": "OpenAI releases SearchGPT, an AI-powered search prototype",
            "description": "OpenAI has announced SearchGPT, a temporary prototype of new AI search features designed to combine the strength of our AI models with information from the web to give you fast and timely answers with clear and relevant sources.",
            "url": "https://openai.com/blog/searchgpt-prototype",
            "source": {"name": "OpenAI Blog"},
            "publishedAt": "2026-06-19T10:00:00Z",
            "type": "trending"
        },
        {
            "title": "Anthropic Claude 3.5 Sonnet sets new industry benchmarks for coding",
            "description": "Anthropic released Claude 3.5 Sonnet, which outperforms competitor models and Claude 3 Opus on a wide range of evaluations, with marked improvements in coding, reasoning, and mathematical skills.",
            "url": "https://anthropic.com/claude-3-5-sonnet",
            "source": {"name": "Anthropic Blog"},
            "publishedAt": "2026-06-20T10:00:00Z",
            "type": "research"
        },
        {
            "title": "Google updates NotebookLM with audio overview features",
            "description": "Google announced a new update to NotebookLM that lets users generate a dynamic audio overview where two AI hosts discuss the documents in a conversational format.",
            "url": "https://google.com/notebooklm-audio-overview",
            "source": {"name": "Google Blog"},
            "publishedAt": "2026-06-21T10:00:00Z",
            "type": "tools"
        },
        {
            "title": "Meta Llama 3.1 405B model released as open-source",
            "description": "Meta has officially released Llama 3.1 405B, the first frontier-class open-source AI model, boasting state-of-the-art capabilities in multilingual translation, coding, and logical reasoning.",
            "url": "https://meta.com/llama-3-1-released",
            "source": {"name": "Meta Newsroom"},
            "publishedAt": "2026-06-22T10:00:00Z",
            "type": "trending"
        }
    ]
    
    script_data = generate_longform_script(articles=mock_articles)
    
    if not script_data:
        print("❌ Script generation failed!")
        return
        
    print("\n✅ Script generation succeeded!")
    print("\n--- Output Details ---")
    print(f"Title: {script_data.get('title')}")
    print(f"Longform Format: {script_data.get('longform_format')}")
    print(f"Num Facts/Topics: {script_data.get('num_facts')}")
    print(f"Original News URL: {script_data.get('original_news_url')}")
    
    script_text = script_data.get('script', '')
    word_count = len(script_text.split())
    print(f"\n--- Unified Script ({word_count} words) ---")
    print(script_text[:1000] + "\n...")
    
    print("\n--- Description with Timestamps ---")
    desc = format_longform_description(script_data, script_data.get("hashtags", []))
    print(desc)

if __name__ == "__main__":
    test_vaibhav_longform()
