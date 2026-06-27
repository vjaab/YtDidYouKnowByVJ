import sys
import os

# Add parent directory to path so we can import vidiq_trending
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from vidiq_trending import get_pipeline_topics, _extract_topic
    
    print("🧪 Running vidIQ module tests...")
    
    # 1. Test clean topic extraction
    title = "🤯 Stop Paying for Midjourney! This AI does it FREE! (New generator)"
    clean = _extract_topic(title)
    print(f"Original title: '{title}'")
    print(f"Extracted topic: '{clean}'")
    
    assert "Stop Paying for Midjourney" in clean or "Midjourney" in clean
    print("✅ Topic extraction parsed successfully!")
    
    # 2. Test pipeline topic generation (with mock/fallback)
    print("\n📡 Generating pipeline topics (simulating empty/missing API key fallback)...")
    topics = get_pipeline_topics(category="AI & Tech Tools")
    
    print(f"Retrieved {len(topics)} topics.")
    for idx, item in enumerate(topics[:3], 1):
        print(f"  {idx}. Topic: '{item['title']}' | Source: {item['source']} | Score: {item['score']}")
        
    assert len(topics) > 0
    assert all("title" in t for t in topics)
    assert all("score" in t for t in topics)
    
    print("\n✅ All vidIQ module tests passed successfully!")
except Exception as e:
    print(f"❌ Tests failed: {e}")
    sys.exit(1)
