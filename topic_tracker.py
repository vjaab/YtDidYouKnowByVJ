import json
import os
from datetime import datetime
from rapidfuzz import fuzz
from config import TRACKER_FILE

def load_tracker():
    if not os.path.exists(TRACKER_FILE):
        return {
            "used_titles": [],
            "used_keywords": [],
            "used_companies": {},
            "used_subcategories": {},
            "last_7_days_stories": [],
            "last_3_days_subcategories": [],
            "last_3_days_companies": [],
            "total_uploaded": 0,
            "last_upload": None,
            "history": []
        }
    with open(TRACKER_FILE, 'r') as f:
        return json.load(f)

def save_tracker(tracker_data):
    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker_data, f, indent=4)

def check_story_uniqueness(new_title):
    tracker = load_tracker()
    for old_title in tracker.get('used_titles', []):
        if fuzz.ratio(new_title.lower(), old_title.lower()) > 75:
            return False, f"Similar to previously covered story: '{old_title}'"
    return True, "Unique"
    
def check_cooldowns(companies, subcategory):
    tracker = load_tracker()
    last_3_companies = tracker.get('last_3_days_companies', [])
    for comp in companies:
        if comp in last_3_companies:
            return False, f"Company '{comp}' covered in last 3 days."
            
    last_3_subcategories = tracker.get('last_3_days_subcategories', [])
    if last_3_subcategories.count(subcategory) >= 2:
        return False, f"Subcategory '{subcategory}' overused in last 3 days."
        
    return True, "Cooldowns OK"

def record_story(title, news_headline, subcategory, companies, keywords, breaking_news_level, voice_used, youtube_url, news_source_url):
    tracker = load_tracker()
    today = datetime.now().strftime("%Y-%m-%d")
    
    tracker.setdefault("used_titles", []).append(title)
    tracker.setdefault("used_titles", []).append(news_headline)
    
    tracker.setdefault("used_keywords", []).extend(keywords)
    tracker["used_keywords"] = list(set(tracker["used_keywords"]))
    
    tracker.setdefault("used_companies", {})
    for comp in companies:
        tracker["used_companies"][comp] = tracker["used_companies"].get(comp, 0) + 1
        
    tracker.setdefault("used_subcategories", {})
    tracker["used_subcategories"][subcategory] = tracker["used_subcategories"].get(subcategory, 0) + 1
    
    tracker.setdefault("last_7_days_stories", []).append(news_headline)
    if len(tracker["last_7_days_stories"]) > 7:
        tracker["last_7_days_stories"].pop(0)
        
    tracker.setdefault("last_3_days_subcategories", []).append(subcategory)
    if len(tracker["last_3_days_subcategories"]) > 3:
        tracker["last_3_days_subcategories"].pop(0)
        
    for comp in companies:
        tracker.setdefault("last_3_days_companies", []).append(comp)
    if len(tracker["last_3_days_companies"]) > 5:
        tracker["last_3_days_companies"] = tracker["last_3_days_companies"][-5:]
    
    tracker["total_uploaded"] = tracker.get("total_uploaded", 0) + 1
    tracker["last_upload"] = today
    
    history_entry = {
        "date": today,
        "title": title,
        "news_headline": news_headline,
        "sub_category": subcategory,
        "companies": companies,
        "keywords": keywords,
        "breaking_news_level": breaking_news_level,
        "voice_used": voice_used,
        "youtube_url": youtube_url,
        "news_source_url": news_source_url
    }
    tracker.setdefault("history", []).append(history_entry)
    save_tracker(tracker)
