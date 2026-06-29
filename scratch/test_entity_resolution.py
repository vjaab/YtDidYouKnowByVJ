import os
import sys

# Add base directory to path so imports work
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from entity_fetcher import fetch_all_entities, resolve_tech_entity, TECH_ENTITY_MAPPING
from config import OUTPUT_DIR

def run_tests():
    print("🚀 Running entity resolution and logo/portrait fetching tests...\n")
    
    # 1. Test tech entity resolution dictionary mapping
    test_names = [
        "ChatGPT",
        "Claude 3.5 Sonnet",
        "Llama 3.2",
        "Sam Altman",
        "Nvidia GPU",
        "Microsoft Copilot",
        "Unmapped Entity"
    ]
    
    print("--- 1. Testing resolution mapping ---")
    for name in test_names:
        resolved = resolve_tech_entity(name)
        if resolved:
            print(f"✅ Resolved '{name}' -> domain: {resolved.get('domain')}, wiki: {resolved.get('wiki_slug')}")
        else:
            print(f"❌ '{name}' did not resolve to mapped tech entity.")
    print()

    # 2. Test fetching logos and portraits
    mock_script_data = {
        "title": "AI Innovation Test",
        "script": "Sam Altman announced ChatGPT. Meanwhile, Anthropic Claude 3.5 Sonnet was released.",
        "original_news_headline": "AI Advancements",
        "original_news_url": "https://example.com",
        "companies": [
            {"name": "ChatGPT", "description": "AI chatbot tool by OpenAI"},
            {"name": "Claude 3.5 Sonnet", "description": "Next-generation LLM from Anthropic"}
        ],
        "people": [
            {"name": "Sam Altman", "description": "CEO of OpenAI"}
        ],
        "key_entities": [
            {"name": "Llama 3.2", "type": "TOOL", "description": "Meta open source model"},
            {"name": "Sundar Pichai", "type": "PEOPLE", "description": "CEO of Google"},
            {"name": "MyCustomFakeCompany", "type": "COMPANY", "description": "A completely custom startup"}
        ]
    }
    
    print("--- 2. Testing fetch_all_entities logic ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_data = fetch_all_entities(mock_script_data)
    
    print("\nFetch results:")
    for company in result_data.get("companies", []):
        logo_path = company.get("local_logo_path") or company.get("local_hq_path")
        print(f"Company: {company['name']} -> Path: {logo_path} (Exists: {os.path.exists(logo_path) if logo_path else False})")
        
    for person in result_data.get("people", []):
        photo_path = person.get("local_image_path")
        print(f"Person: {person['name']} -> Path: {photo_path} (Exists: {os.path.exists(photo_path) if photo_path else False})")
        
    for entity in result_data.get("key_entities", []):
        path = entity.get("local_logo_path") or entity.get("local_hq_path") or entity.get("local_image_path")
        print(f"Key Entity: {entity['name']} ({entity.get('type')}) -> Path: {path} (Exists: {os.path.exists(path) if path else False})")

if __name__ == "__main__":
    run_tests()
