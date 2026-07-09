import sys
sys.path.append('.')
from tags_helper import get_optimized_metadata, to_clean_hashtag

def run_test():
    print("🧪 Running Dynamic Hashtags and Tags Optimization Tests...\n")
    
    # Test 1: to_clean_hashtag formatting
    print("--- Test 1: Hashtag formatting (PascalCase) ---")
    test_cases = [
        ("Google Colab", "#GoogleColab"),
        ("google colab", "#GoogleColab"),
        ("LLM", "#LLM"),
        ("large language model", "#LargeLanguageModel"),
        ("chatgpt-4o", "#Chatgpt4o"),
        ("Python", "#Python"),
        ("Yann LeCun", "#YannLeCun"),
        ("#AlreadyHasHash", "#AlreadyHasHash"),
        ("A very long name that exceeds the maximum length constraint of twenty-five chars", "")
    ]
    for inp, expected in test_cases:
        out = to_clean_hashtag(inp)
        status = "✅" if out == expected else f"❌ (Got: '{out}')"
        print(f"'{inp}' -> '{out}' {status}")
        
    # Test 2: MiniMind metadata extraction
    print("\n--- Test 2: MiniMind Metadata Hashtags ---")
    title = "Train an LLM in 2 Hours?! 🤯 MiniMind Hack"
    script = "Today we are looking at MiniMind, a project by jingyaogong. With Google Colab, you can train a 64M LLM in 2 hours using Python."
    sub_category = "AI & Tech Tools"
    keywords = ["AI Hacks", "Tech Tips", "Productivity", "LLM", "Machine Learning", "Deep Learning", "AI Training", "Google Colab", "Python"]
    companies = [{"name": "Google Colab"}, {"name": "minimind"}]
    people = []
    initial_hashtags = ["#AIHacks", "#TechTips", "#Productivity", "#VaibhavSisinty"]
    
    res = get_optimized_metadata(
        title=title,
        script=script,
        sub_category=sub_category,
        initial_keywords=keywords,
        initial_companies=[c["name"] for c in companies],
        initial_people=people,
        initial_hashtags=initial_hashtags
    )
    
    # Trace info
    from tags_helper import GENERIC_HASHTAGS
    raw_all = []
    for item in [c["name"] for c in companies] + people + keywords + [sub_category] + initial_hashtags:
        ht = to_clean_hashtag(item)
        if ht: raw_all.append(ht)
    print("Raw Hashtags:", raw_all)
    
    print("Input Title:", title)
    print("Generated Tags:", res["tags"])
    print("Generated Hashtags:", res["hashtags"])
    
    # Assert exactly 4 hyper-targeted hashtags as per rules
    assert res["hashtags"] == ["#GoogleColab", "#Minimind", "#Python", "#MiniMindHack"], f"Unexpected hashtags: {res['hashtags']}"
    print("✅ Successfully verified hyper-targeted 4-hashtag output!")

if __name__ == "__main__":
    run_test()
