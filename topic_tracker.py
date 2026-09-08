import json
import os
from datetime import datetime
from rapidfuzz import fuzz
from config import TRACKER_FILE

def load_tracker(tracker_file=TRACKER_FILE):
    if not os.path.exists(tracker_file):
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
    try:
        with open(tracker_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrupt_backup = f"{tracker_file}.corrupt_{timestamp}"
        try:
            shutil.copy2(tracker_file, corrupt_backup)
            print(f"⚠️ Warning: {tracker_file} is corrupted. Backed up to {corrupt_backup}")
        except Exception as copy_err:
            print(f"❌ Failed to backup corrupted tracker file: {copy_err}")
        
        print(f"❌ JSON Decode Error reading {tracker_file}: {e}")
        print("💡 Suggestion: Check for Git conflict markers (<<<<<<<, =======, >>>>>>>) or partial writes in the file.")
        raise

def save_tracker(tracker_data, tracker_file=TRACKER_FILE):
    tmp_file = f"{tracker_file}.tmp"
    try:
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(tracker_data, f, indent=4)
        os.replace(tmp_file, tracker_file)
    except Exception as e:
        print(f"❌ Failed to save tracker to {tracker_file}: {e}")
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass
        raise

def check_story_uniqueness(new_title, new_headline=None, new_keywords=None, new_url=None, tracker_file=TRACKER_FILE):
    tracker = load_tracker(tracker_file)
    if not tracker:
        return True, "Unique (Empty tracker)"
    
    # 1. Exact URL Check
    if new_url:
        for entry in tracker.get('history', []):
            if not isinstance(entry, dict): continue
            if entry.get('news_source_url') == new_url:
                return False, f"Exact URL already covered: {new_url}"

    # 2. Semantic Headline Check (Token Set Ratio handles reordering)
    from config import SIMILARITY_THRESHOLD
    headlines_to_check = (tracker.get('used_titles', []) or []) + (tracker.get('last_7_days_stories', []) or [])
    
    search_titles = [new_title]
    if new_headline: search_titles.append(new_headline)
    
    for existing_title in set(headlines_to_check):
        for st in search_titles:
            score = fuzz.token_set_ratio(st.lower(), existing_title.lower())
            if score > SIMILARITY_THRESHOLD: 
                return False, f"Semantic match found (score {score}): '{existing_title}'"
            
    # 3. Keyword Overlap Check (Batch Deduplication)
    if new_keywords:
        recent_keywords = []
        for entry in tracker.get('history', [])[-10:]:  # Look at last 10 stories
            # Skip if same URL (exact same story already recorded)
            if new_url and entry.get('news_source_url') == new_url:
                continue
            recent_keywords.extend([k.lower() for k in entry.get('keywords', [])])
        
        new_k_set = set([k.lower() for k in new_keywords])
        old_k_set = set(recent_keywords)
        intersection = new_k_set.intersection(old_k_set)
        
        # If > 65% of keywords overlap with recent stories, it's likely redundant
        if len(new_k_set) > 0:
            overlap_pct = (len(intersection) / len(new_k_set)) * 100
            if overlap_pct > 65:
                return False, f"High keyword overlap ({overlap_pct:.0f}%) with recent stories."
                
    return True, "Unique"
    
def check_cooldowns(companies, subcategory, tracker_file=TRACKER_FILE):
    tracker = load_tracker(tracker_file)
    last_3_companies = tracker.get('last_3_days_companies', [])
    for comp in companies:
        comp_name = comp.get("name") if isinstance(comp, dict) else comp
        if comp_name in last_3_companies:
            return False, f"Company '{comp_name}' covered in last 3 days."
            
    last_3_subcategories = tracker.get('last_3_days_subcategories', [])
    if last_3_subcategories.count(subcategory) >= 2:
        return False, f"Subcategory '{subcategory}' overused in last 3 days."
        
    return True, "Cooldowns OK"

def record_story(title, news_headline, subcategory, companies, keywords, breaking_news_level, voice_used, youtube_url, news_source_url, topic_type=None, target_country=None, avatar_used=None, student_vector=None, tracker_file=TRACKER_FILE):
    tracker = load_tracker(tracker_file)
    today = datetime.now().strftime("%Y-%m-%d")
    
    tracker.setdefault("used_titles", []).append(title)
    tracker.setdefault("used_titles", []).append(news_headline)
    
    tracker.setdefault("used_keywords", []).extend(keywords)
    tracker["used_keywords"] = list(set(tracker["used_keywords"]))
    
    tracker.setdefault("used_companies", {})
    for comp in companies:
        if isinstance(comp, dict):
            comp_name = comp.get("name")
            if comp_name:
                tracker["used_companies"][comp_name] = tracker["used_companies"].get(comp_name, 0) + 1
        elif isinstance(comp, str):
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
        comp_name = comp.get("name") if isinstance(comp, dict) else comp
        if comp_name:
            tracker.setdefault("last_3_days_companies", []).append(comp_name)
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
        "facebook_post_id": None,
        "news_source_url": news_source_url,
        "target_country": target_country,
        "avatar_used": avatar_used
    }
    if topic_type:
        history_entry["topic_type"] = topic_type
    if student_vector:
        history_entry["student_vector"] = student_vector
        
    tracker.setdefault("history", []).append(history_entry)
    save_tracker(tracker, tracker_file)

def update_youtube_url(news_headline, youtube_url, tracker_file=TRACKER_FILE):
    tracker = load_tracker(tracker_file)
    for entry in tracker.get("history", []):
        if entry.get("news_headline") == news_headline:
            entry["youtube_url"] = youtube_url
            break
    save_tracker(tracker, tracker_file)

def update_facebook_post_id(news_headline, facebook_post_id, tracker_file=TRACKER_FILE):
    tracker = load_tracker(tracker_file)
    for entry in tracker.get("history", []):
        if entry.get("news_headline") == news_headline:
            entry["facebook_post_id"] = facebook_post_id
            break
    save_tracker(tracker, tracker_file)

# ── STUDENT CONTENT VECTORS ────────────────────────────────────────────────
# 4 student sub-vectors that rotate cyclically within the 40% student allocation
STUDENT_VECTORS = [
    "student_academic_ai",    # NotebookLM, Gemini for research, flashcard automation
    "student_dev",            # GitHub Student Pack, Copilot Pro, cloud credits, IDE shortcuts, zsh setup
    "student_capstone",       # Production RAG architectures, multi-agent AI, voice assistants for resume projects
    "student_contrarian",     # "You're studying/coding wrong" myth-busting formats
]

def get_student_sub_vector(tracker_file=TRACKER_FILE):
    """
    Returns the next student sub-vector by cycling through the 4 vectors.
    Checks recent history for the last used student_vector and returns the next one.
    """
    tracker = load_tracker(tracker_file)
    history = tracker.get("history", [])
    
    # Find the last student_vector used
    last_vector = None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        sv = entry.get("student_vector")
        if sv and sv in STUDENT_VECTORS:
            last_vector = sv
            break
    
    if not last_vector:
        return STUDENT_VECTORS[0]
    
    try:
        idx = STUDENT_VECTORS.index(last_vector)
        next_idx = (idx + 1) % len(STUDENT_VECTORS)
        return STUDENT_VECTORS[next_idx]
    except ValueError:
        return STUDENT_VECTORS[0]


def _get_core_topic_type(tracker_file=TRACKER_FILE):
    """
    Internal: Computes the next core topic type using deficit-based ratio balancing.
    Only considers core types (excludes 'student').
    Target ratios (within the 60% core allocation):
      - tools: 35% — Hidden features, AI tools, free apps, tips & tricks
      - news: 20% — Tech myths, privacy scares, common mistakes  
      - research: 10% — Comparisons, AI experiments, educational tech facts
      - quiz: 15% — Interactive tech trivia, history quizzes, multiple choice
      - interview_questions: 20% — Technical interview Q&A
    """
    tracker = load_tracker(tracker_file)
    history = tracker.get("history", [])
    
    target_ratios = {
        "tools": 0.35,
        "news": 0.20,
        "research": 0.10,
        "quiz": 0.15,
        "interview_questions": 0.20
    }
    
    # Only analyze core entries (non-student) from last 30
    recent_entries = history[-30:] if history else []
    
    counts = {"tools": 0, "news": 0, "research": 0, "quiz": 0, "interview_questions": 0}
    total_counted = 0
    
    for entry in recent_entries:
        if not isinstance(entry, dict):
            continue
        ttype = entry.get("topic_type")
        # Skip student entries for core ratio calculation
        if ttype == "student":
            continue
        if ttype in counts:
            counts[ttype] += 1
            total_counted += 1
        else:
            # Heuristics for backward compatibility with existing entries
            sub_cat = str(entry.get("sub_category", "")).lower()
            title = str(entry.get("title", "")).lower()
            headline = str(entry.get("news_headline", "")).lower()
            
            if "tool" in sub_cat or "app" in sub_cat or "feature" in sub_cat or "tip" in title or "trick" in title or "hidden" in title or "hack" in title:
                counts["tools"] += 1
                total_counted += 1
            elif "myth" in sub_cat or "privacy" in sub_cat or "scary" in sub_cat or "wrong" in title or "mistake" in title or "stop" in title or "myth" in title:
                counts["news"] += 1
                total_counted += 1
            elif "quiz" in sub_cat or "trivia" in sub_cat or "quiz" in title or "trivia" in title:
                counts["quiz"] += 1
                total_counted += 1
            elif "interview" in sub_cat or "interview" in title or "interview" in headline:
                counts["interview_questions"] += 1
                total_counted += 1
            else:
                counts["research"] += 1
                total_counted += 1
                
    if total_counted == 0:
        return "tools"
        
    deficits = {}
    for t, target in target_ratios.items():
        current_ratio = counts[t] / total_counted
        deficits[t] = target - current_ratio
        
    selected = max(deficits, key=deficits.get)
    print(f"📊 Core ratio calculation: counts={counts}, deficits={deficits} -> Selected: {selected}")
    return selected


def get_next_topic_type_by_ratio(tracker_file=TRACKER_FILE):
    """
    2-Tier Weighted Allocation System:
    
    TIER 1: Decides between 'student' (40%) and 'core' (60%) content.
    Uses a sliding window of the last 10 entries to enforce the 40/60 split.
    Every 5 generated Shorts should contain exactly 2 student-targeted scripts.
    
    TIER 2: 
      - If student: returns 'student' (sub-vector selected separately via get_student_sub_vector())
      - If core: delegates to _get_core_topic_type() for deficit-based ratio balancing
        among tools/news/research/quiz/interview_questions.
    """
    tracker = load_tracker(tracker_file)
    history = tracker.get("history", [])
    
    # Tier 1: Student vs Core allocation (sliding window of last 10)
    recent_10 = history[-10:] if history else []
    
    student_count = 0
    core_count = 0
    for entry in recent_10:
        if not isinstance(entry, dict):
            continue
        ttype = entry.get("topic_type", "")
        if ttype == "student":
            student_count += 1
        else:
            core_count += 1
    
    total = student_count + core_count
    
    if total == 0:
        # Cold start: begin with student content to seed the ratio
        print("📊 Tier 1: Cold start -> Selecting 'student'")
        return "student"
    
    # Target: 40% student, 60% core
    student_ratio = student_count / total
    student_deficit = 0.40 - student_ratio
    core_deficit = 0.60 - (core_count / total)
    
    if student_deficit > core_deficit:
        # Student content is underrepresented
        print(f"📊 Tier 1: student={student_count}/{total} ({student_ratio:.0%}), deficit={student_deficit:+.2f} -> Selecting 'student'")
        return "student"
    else:
        # Core content is underrepresented or balanced
        core_type = _get_core_topic_type(tracker_file)
        print(f"📊 Tier 1: student={student_count}/{total} ({student_ratio:.0%}), deficit={student_deficit:+.2f} -> Selecting core: '{core_type}'")
        return core_type


def get_next_target_country(tracker_file=TRACKER_FILE):
    """
    Determines the next target country in the sequence:
    US -> GB -> CA -> AU -> NZ -> SG -> KR -> JP -> DE -> FR -> IE
    based on the last recorded story's target country.
    """
    tracker = load_tracker(tracker_file)
    history = tracker.get("history", [])
    
    country_sequence = ["US", "GB", "CA", "AU", "NZ", "SG", "KR", "JP", "DE", "FR", "IE"]
    
    # Traverse history backwards to find the last target country
    last_country = None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        c = entry.get("target_country")
        if c in country_sequence:
            last_country = c
            break
            
    if not last_country:
        return "US"
        
    try:
        idx = country_sequence.index(last_country)
        next_idx = (idx + 1) % len(country_sequence)
        return country_sequence[next_idx]
    except ValueError:
        return "US"


def get_next_avatar(intro_videos, tracker_file=TRACKER_FILE):
    """
    Selects the next avatar from the list of intro videos,
    ensuring we rotate through all of them before repeating.
    """
    if not intro_videos:
        return None
        
    tracker = load_tracker(tracker_file)
    history = tracker.get("history", [])
    
    # Sort intro_videos to guarantee consistent indexing across runs
    sorted_videos = sorted(intro_videos)
    
    # Find the last used avatar path in history
    last_avatar = None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        av = entry.get("avatar_used")
        if av in sorted_videos:
            last_avatar = av
            break
            
    if not last_avatar:
        return sorted_videos[0]
        
    try:
        idx = sorted_videos.index(last_avatar)
        next_idx = (idx + 1) % len(sorted_videos)
        return sorted_videos[next_idx]
    except ValueError:
        return sorted_videos[0]

