"""
tags_helper.py — Dynamic metadata tags and hashtags optimizer for YouTube upload.
Matches title, script, and metadata against curated AI & tech tag categories and key figures.
"""
import os
import re
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# ── CATEGORIES OF TAGS ──

BROAD_AI_TAGS = [
    "artificial intelligence", "AI", "machine learning", "deep learning", "neural network",
    "generative AI", "AI technology", "AI 2025", "AI 2026", "AI tools", "AI news",
    "future of AI", "AI explained", "AI tutorial", "AI for beginners"
]

TRENDING_AI_TOPICS = [
    "ChatGPT", "GPT-4", "Claude AI", "Gemini AI", "Llama", "Mistral", "open source AI",
    "large language model", "LLM", "AI chatbot", "AI assistant", "prompt engineering",
    "AI image generation", "text to image", "text to video", "AI video generator",
    "Sora", "Veo", "Runway", "Midjourney", "Stable Diffusion", "DALL-E", "FLUX"
]

TECHNOLOGY_TAGS = [
    "technology", "tech news", "tech explained", "latest technology", "future technology",
    "emerging technology", "tech trends 2026", "automation", "robotics", "computer science",
    "programming", "software development", "coding", "Python", "developer tools"
]

SPECIFIC_TECH_NICHES = [
    "AI automation", "workflow automation", "no-code AI", "AI agents", "agentic AI",
    "AI productivity", "AI for content creators", "YouTube automation", "AI pipeline",
    "cloud computing", "GPU computing", "edge AI", "on-device AI", "AI hardware",
    "NVIDIA", "AI chips", "semiconductor", "quantum computing"
]

FACTS_STYLE_TAGS = [
    "did you know", "amazing facts", "tech facts", "AI facts", "mind blowing AI",
    "shocking technology", "AI you didn't know", "future tech facts", "science facts",
    "interesting technology", "AI discoveries", "AI breakthroughs"
]

VIRAL_HOOK_TAGS = [
    "AI going viral", "AI vs human", "will AI replace jobs", "AI taking over",
    "AI robots 2026", "scariest AI", "most powerful AI", "AI that shocked the world",
    "AI secrets", "hidden AI", "AI you need to know", "AI changed everything"
]

LONG_TAIL_TAGS = [
    "how does AI work", "what is artificial intelligence", "AI explained simply",
    "best AI tools 2026", "free AI tools", "AI tools for beginners",
    "AI for YouTube creators", "how to use AI", "AI automation tutorial",
    "machine learning explained", "deep learning for beginners"
]

# ── PROMINENT PEOPLE METADATA ──

PEOPLE_METADATA = {
    "sam altman": {
        "tags": ["Sam Altman", "OpenAI CEO", "ChatGPT creator"],
        "hashtag": "#SamAltman"
    },
    "elon musk": {
        "tags": ["Elon Musk", "xAI", "Grok AI", "Tesla AI", "Neuralink"],
        "hashtag": "#ElonMusk"
    },
    "dario amodei": {
        "tags": ["Dario Amodei", "Anthropic CEO", "Claude AI"],
        "hashtag": "#DarioAmodei"
    },
    "daniela amodei": {
        "tags": ["Daniela Amodei", "Anthropic President", "Claude AI"],
        "hashtag": "#DanielaAmodei"
    },
    "demis hassabis": {
        "tags": ["Demis Hassabis", "Google DeepMind CEO", "DeepMind", "AlphaFold"],
        "hashtag": "#DemisHassabis"
    },
    "jensen huang": {
        "tags": ["Jensen Huang", "NVIDIA CEO", "NVIDIA AI", "AI Chips"],
        "hashtag": "#JensenHuang"
    },
    "satya nadella": {
        "tags": ["Satya Nadella", "Microsoft CEO", "Copilot AI"],
        "hashtag": "#SatyaNadella"
    },
    "sundar pichai": {
        "tags": ["Sundar Pichai", "Google CEO", "Gemini AI"],
        "hashtag": "#SundarPichai"
    },
    "mark zuckerberg": {
        "tags": ["Mark Zuckerberg", "Meta CEO", "Llama AI"],
        "hashtag": "#MarkZuckerberg"
    },
    "geoffrey hinton": {
        "tags": ["Geoffrey Hinton", "Godfather of AI", "AI Nobel Prize"],
        "hashtag": "#GeoffreyHinton"
    },
    "yann lecun": {
        "tags": ["Yann LeCun", "Meta AI Chief", "CNN inventor"],
        "hashtag": "#YannLeCun"
    },
    "yoshua bengio": {
        "tags": ["Yoshua Bengio", "deep learning pioneer", "AI safety"],
        "hashtag": "#YoshuaBengio"
    },
    "andrew ng": {
        "tags": ["Andrew Ng", "DeepLearning.AI", "Coursera AI"],
        "hashtag": "#AndrewNg"
    },
    "andrej karpathy": {
        "tags": ["Andrej Karpathy", "Tesla AI", "OpenAI", "AI educator"],
        "hashtag": "#AndrejKarpathy"
    },
    "fei-fei li": {
        "tags": ["Fei-Fei Li", "Stanford AI", "ImageNet", "World Labs"],
        "hashtag": "#FeiFeiLi"
    },
    "ian goodfellow": {
        "tags": ["Ian Goodfellow", "GANs inventor"],
        "hashtag": "#IanGoodfellow"
    },
    "lex fridman": {
        "tags": ["Lex Fridman", "Lex Fridman Podcast", "MIT AI"],
        "hashtag": "#LexFridman"
    },
    "grant sanderson": {
        "tags": ["3Blue1Brown", "Grant Sanderson", "Math AI"],
        "hashtag": "#3Blue1Brown"
    },
    "three blue one brown": {
        "tags": ["3Blue1Brown", "Grant Sanderson", "Math AI"],
        "hashtag": "#3Blue1Brown"
    },
    "two minute papers": {
        "tags": ["Two Minute Papers", "AI research papers"],
        "hashtag": "#TwoMinutePapers"
    },
    "yannic kilcher": {
        "tags": ["Yannic Kilcher", "ML paper breakdown"],
        "hashtag": "#YannicKilcher"
    },
    "sebastian raschka": {
        "tags": ["Sebastian Raschka", "build LLM"],
        "hashtag": "#SebastianRaschka"
    },
    "gary marcus": {
        "tags": ["Gary Marcus", "AI skeptic"],
        "hashtag": "#GaryMarcus"
    },
    "bill gates": {
        "tags": ["Bill Gates", "Microsoft founder"],
        "hashtag": "#BillGates"
    },
    "jeff bezos": {
        "tags": ["Jeff Bezos", "Amazon AWS AI"],
        "hashtag": "#JeffBezos"
    },
    "tim cook": {
        "tags": ["Tim Cook", "Apple Intelligence", "Apple CEO"],
        "hashtag": "#TimCook"
    },
    "linus torvalds": {
        "tags": ["Linus Torvalds", "Linux creator"],
        "hashtag": "#LinusTorvalds"
    },
    "kai-fu lee": {
        "tags": ["Kai-Fu Lee", "AI expert", "01.AI"],
        "hashtag": "#KaiFuLee"
    }
}

# ── HASHTAG LIMITS BY FORMAT ──
SHORTS_MAX_HASHTAGS = 10
LONGFORM_MAX_HASHTAGS = 15

# ── TOP 5 HASHTAGS FOR ENGLISH FACTS CHANNEL ──
DEFAULT_BROAD_HASHTAGS = ["#ArtificialIntelligence", "#Technology", "#MachineLearning"]
DEFAULT_NICHE_HASHTAGS = ["#AITools", "#TechFacts"]

# ── TRENDING HASHTAG CACHE ──
TRENDING_HASHTAGS_CACHE_FILE = os.path.join("logs", "trending_hashtags_cache.json")
TRENDING_HASHTAGS_CACHE_TTL_HOURS = 6

GENERIC_HASHTAGS = {
    "aihacks", "techtips", "productivity", "vaibhavsisinty",
    "didyouknow", "amazingfacts", "techfacts", "aifacts", "futuretech",
    "artificialintelligence", "technology", "machinelearning", "aitools",
    "techusa", "techuk", "techcanada", "techaustralia", "technz",
    "techsingapore", "techsouthkorea", "techjapan", "techeurope", "english",
    "longform", "deeptech", "softwareengineering", "technews", "llm",
    "aivideo", "aifails", "aihacking", "hacks", "tips", "fact", "facts", "tech",
    "sciencefacts", "interestingtechnology", "youtube", "video",
    "how", "why", "what", "where",
    "hack", "tip", "tutorial", "explained", "science", "future",
    "deeplearning", "aitraining", "aitechtools", "techtools"
    # NOTE: Removed #shorts, #ai, #viral, #trending, #fyp, #foryou, #shortsfeed
    # from blocklist — these are discovery-critical for Shorts feed algorithm
}


def fetch_trending_hashtags_from_google(target_country="US") -> list:
    """
    Fetches trending hashtags from Google Trends RSS feed.
    Filters for tech/AI related trends and converts to hashtag format.
    Uses caching to avoid repeated API calls.
    """
    # Check cache first
    if os.path.exists(TRENDING_HASHTAGS_CACHE_FILE):
        try:
            mtime = os.path.getmtime(TRENDING_HASHTAGS_CACHE_FILE)
            age_hours = (time.time() - mtime) / 3600.0
            if age_hours < TRENDING_HASHTAGS_CACHE_TTL_HOURS:
                with open(TRENDING_HASHTAGS_CACHE_FILE, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                print(f"📋 Loaded trending hashtags cache (Age: {age_hours:.1f} hours)")
                return cached
        except Exception as e:
            print(f"⚠️ Failed to load trending hashtags cache: {e}")

    # Tech/AI whitelist for filtering Google Trends
    tech_whitelist = {
        'ai', 'artificial intelligence', 'chatgpt', 'gpt', 'llm', 'llama', 'claude', 'gemini',
        'openai', 'anthropic', 'deepseek', 'mistral', 'ollama', 'agent', 'autogen', 'crewai',
        'langgraph', 'langchain', 'rag', 'vector', 'embedding', 'fine-tune', 'quantization',
        'nvidia', 'gpu', 'cuda', 'h100', 'h200', 'blackwell', 'chip', 'semiconductor',
        'python', 'javascript', 'typescript', 'rust', 'go', 'coding', 'programming',
        'vscode', 'github', 'copilot', 'cursor', 'ide', 'developer', 'devops', 'kubernetes',
        'docker', 'aws', 'cloud', 'serverless', 'lambda', 'api', 'microservice',
        'robot', 'robotics', 'automation', 'workflow', 'productivity', 'tool',
        'startup', 'funding', 'venture', 'ycombinator', 'unicorn', 'ipo',
        'crypto', 'blockchain', 'web3', 'defi', 'bitcoin', 'ethereum', 'solana',
        'ar', 'vr', 'xr', 'metaverse', 'spatial', 'vision pro', 'quest',
        'quantum', 'quantum computing', 'qubit', 'ibm', 'google quantum',
        'security', 'hack', 'exploit', 'vulnerability', 'zero-day', 'malware',
        'data', 'analytics', 'bi', 'tableau', 'powerbi', 'sql', 'nosql', 'database',
        'ml', 'machine learning', 'deep learning', 'neural', 'transformer', 'attention',
        'diffusion', 'stable diffusion', 'midjourney', 'dall-e', 'sora', 'veo', 'runway',
        'flux', 'image generation', 'video generation', 'text-to-video', 'text-to-image'
    }

    url = f"https://trends.google.com/trending/rss?geo={target_country}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0'}
    trending_hashtags = []

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        items = root.findall('.//item')

        for item in items:
            title = item.find('title').text or ""
            desc = ""
            desc_elem = item.find('description')
            if desc_elem is not None:
                desc = desc_elem.text or ""

            title_lower = title.lower()
            desc_lower = desc.lower()
            combined = f"{title_lower} {desc_lower}"

            # Check if any tech keyword matches
            is_tech = False
            matched_keyword = None
            for keyword in tech_whitelist:
                if keyword in combined:
                    is_tech = True
                    matched_keyword = keyword
                    break

            if is_tech:
                # Convert to hashtag format
                hashtag = to_clean_hashtag(matched_keyword or title)
                if hashtag and len(hashtag) > 2:
                    trending_hashtags.append(hashtag)

        # Deduplicate while preserving order
        seen = set()
        unique_hashtags = []
        for ht in trending_hashtags:
            ht_lower = ht.lower()
            if ht_lower not in seen:
                seen.add(ht_lower)
                unique_hashtags.append(ht)

        # Cache the results
        try:
            os.makedirs(os.path.dirname(TRENDING_HASHTAGS_CACHE_FILE), exist_ok=True)
            with open(TRENDING_HASHTAGS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(unique_hashtags, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save trending hashtags cache: {e}")

        print(f"✅ Google Trends: Found {len(unique_hashtags)} tech-related trending hashtags for {target_country}")
        return unique_hashtags

    except Exception as e:
        print(f"⚠️ Google Trends hashtag fetch failed: {e}")
        return []


def get_trending_hashtags_for_content(title, script, target_country="US", max_count=5) -> list:
    """
    Returns trending hashtags relevant to the content.
    Filters Google Trends results against the video's title/script.
    """
    trending = fetch_trending_hashtags_from_google(target_country)
    
    if not trending:
        return []
    
    full_text = f"{title} {script}".lower()
    relevant = []
    
    for ht in trending:
        # Extract the keyword from hashtag (remove # and convert from PascalCase)
        keyword = ht[1:].lower()
        # Check if keyword appears in content (flexible matching)
        if keyword in full_text or any(word in full_text for word in keyword.split()):
            relevant.append(ht)
    
    return relevant[:max_count]


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could",
    "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes",
    "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im",
    "ive", "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt",
    "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
    "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent",
    "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys",
    "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
    "yourselves", "here", "there", "this", "that", "these", "those", "simple", "hidden", "setting", "using", "just",
    "want", "know", "how", "make", "take", "show", "unveil", "claim", "hint", "skeptical", "critics", "coming",
    "sooner", "think", "change", "forever", "watch", "moves", "daily", "setting", "settings", "can", "will", "shall", "today", "hours", "project", "looking", "year", "years", "month", "months", "day", "days", "train"
}
def extract_capitalized_keywords(title, script):
    """Extract sequences of capitalized words from the title and script."""
    text = f"{title} {script}"
    words = re.findall(r'\b[A-Z][a-zA-Z0-9\-]*\b(?:\s+[A-Z][a-zA-Z0-9\-]*\b)?', text)
    cleaned = []
    for w in words:
        w_strip = w.strip()
        if not w_strip:
            continue
        w_parts = re.split(r'[\s\-]+', w_strip.lower())
        if all(part in STOPWORDS for part in w_parts):
            continue
        cleaned.append(w_strip)
    return cleaned

def extract_key_phrases(text):
    """Extract non-stopword words and two-word phrases from the text."""
    clean_text = re.sub(r'[^\w\s\-]', ' ', text)
    words = re.findall(r'\b[a-zA-Z0-9\-]+\b', clean_text.lower())
    candidates = []
    
    # 1. Two-word phrases (higher priority)
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        if w1 not in STOPWORDS and w2 not in STOPWORDS:
            candidates.append(f"{w1} {w2}")
            
    # 2. Single words (lower priority)
    for w in words:
        if w not in STOPWORDS and len(w) > 2 and not w.isdigit():
            candidates.append(w)
            
    return candidates

def to_clean_hashtag(s):
    """Converts a phrase or tag string into a clean PascalCase hashtag."""
    if not s:
        return ""
    # Strip existing hash prefix if present to clean it up
    s_clean = s.strip()
    if s_clean.startswith("#"):
        s_clean = s_clean[1:]
    # Split by spaces, hyphens, underscores, slashes
    parts = re.split(r'[\s\-_\/]+', s_clean)
    capitalized_parts = []
    for p in parts:
        clean_part = re.sub(r'[^\w]', '', p)
        if clean_part:
            # Capitalize first letter, keep the rest (if all lowercase, capitalize first letter)
            if clean_part.islower():
                capitalized_parts.append(clean_part.capitalize())
            else:
                capitalized_parts.append(clean_part)
    hashtag = "".join(capitalized_parts)
    if not hashtag or len(hashtag) < 2 or len(hashtag) > 25:
        return ""
    if hashtag.isdigit():
        return ""
    return f"#{hashtag}"

def generate_specific_hashtags(
    title,
    script,
    sub_category="",
    initial_keywords=None,
    initial_companies=None,
    initial_people=None,
    initial_hashtags=None,
    is_shorts=True,
    target_country="US"
):
    """
    Generates specific keyword-based hashtags based on proper nouns, entities,
    and non-generic phrases found in the content.
    Includes trending hashtags from Google Trends for discovery.
    """
    initial_keywords = initial_keywords or []
    initial_companies = initial_companies or []
    initial_people = initial_people or []
    initial_hashtags = initial_hashtags or []
    
    # Determine max hashtags based on format
    max_hashtags = SHORTS_MAX_HASHTAGS if is_shorts else LONGFORM_MAX_HASHTAGS
    
    # Normalize initial lists
    norm_companies = []
    for c in initial_companies:
        if isinstance(c, dict):
            name = c.get("name")
            if name: norm_companies.append(name)
        elif isinstance(c, str):
            norm_companies.append(c)
            
    norm_people = []
    for p in initial_people:
        if isinstance(p, dict):
            name = p.get("name")
            if name: norm_people.append(name)
        elif isinstance(p, str):
            norm_people.append(p)

    full_text = f"{title} {script} {sub_category}".lower()
    raw_candidates = []

    # 1. Add initial lists (highest priority as they are curated)
    raw_candidates.extend(norm_people)
    raw_candidates.extend(norm_companies)
    raw_candidates.extend(initial_keywords)
    raw_candidates.extend(initial_hashtags)
    if sub_category:
        raw_candidates.append(sub_category)

    # 2. Match Prominent People (PEOPLE_METADATA)
    for person_key, meta in PEOPLE_METADATA.items():
        pattern = rf"\b{re.escape(person_key)}\b"
        last_name = person_key.split()[-1]
        last_name_pattern = rf"\b{re.escape(last_name)}\b"
        if re.search(pattern, full_text) or (len(last_name) > 3 and re.search(last_name_pattern, full_text)):
            raw_candidates.append(meta["hashtag"])

    # 3. Match Specific Tools & Platforms (TRENDING_AI_TOPICS)
    for topic in TRENDING_AI_TOPICS:
        if re.search(rf"\b{re.escape(topic.lower())}\b", full_text):
            raw_candidates.append(topic)

    # 4. Match Tech Sub-niches (SPECIFIC_TECH_NICHES)
    for niche in SPECIFIC_TECH_NICHES:
        if re.search(rf"\b{re.escape(niche.lower())}\b", full_text):
            raw_candidates.append(niche)

    # 5. Extract capitalized proper nouns from title/script
    cap_keywords = extract_capitalized_keywords(title, script)
    raw_candidates.extend(cap_keywords)

    # 6. Extract key phrases from title and script (lowest priority)
    key_phrases = extract_key_phrases(f"{title} {script}")
    raw_candidates.extend(key_phrases)

    # 7. Add trending hashtags from Google Trends (for discovery)
    trending_for_content = get_trending_hashtags_for_content(title, script, target_country, max_count=5)
    raw_candidates.extend(trending_for_content)

    # Clean, normalize, deduplicate, and filter generic hashtags
    seen_ht_lower = set()
    final_hashtags = []
    for item in raw_candidates:
        ht = to_clean_hashtag(item)
        if not ht:
            continue
        ht_lower = ht.lower()
        # Remove hash character to check against GENERIC_HASHTAGS
        clean_word = ht_lower[1:] if ht_lower.startswith("#") else ht_lower
        if clean_word not in GENERIC_HASHTAGS and ht_lower not in seen_ht_lower:
            seen_ht_lower.add(ht_lower)
            final_hashtags.append(ht)

    return final_hashtags[:max_hashtags]

def get_hyper_targeted_hashtags(title, script, is_shorts=True, target_country="US"):
    """
    Generates keyword-based hashtags by extracting proper nouns and key phrases
    from the title and script.
    """
    return generate_specific_hashtags(title, script, is_shorts=is_shorts, target_country=target_country)

def get_optimized_metadata(
    title, 
    script, 
    sub_category="", 
    initial_keywords=None, 
    initial_companies=None, 
    initial_people=None, 
    initial_hashtags=None,
    is_shorts=True,
    target_country="US",
    editorial_perspective=None,
    content_fingerprint=None
):
    """
    Computes an optimized list of 8-15 unique tags and 8-15 unique hashtags
    based on the content of the video.
    Shorts: 8-10 hashtags, Longform: 12-15 hashtags
    Includes trending hashtags from Google Trends for discovery.
    """
    initial_keywords = initial_keywords or []
    initial_companies = initial_companies or []
    initial_people = initial_people or []
    initial_hashtags = initial_hashtags or []
    
    # Normalize initial_companies to list of strings (handle both string and dict formats)
    norm_companies = []
    for c in initial_companies:
        if isinstance(c, dict):
            name = c.get("name")
            if name:
                norm_companies.append(name)
        elif isinstance(c, str):
            norm_companies.append(c)
    initial_companies = norm_companies
    
    # Normalize initial_people to list of strings
    norm_people = []
    for p in initial_people:
        if isinstance(p, dict):
            name = p.get("name")
            if name:
                norm_people.append(name)
        elif isinstance(p, str):
            norm_people.append(p)
    initial_people = norm_people
    
    # Add editorial perspective to keywords for metadata diversity
    if editorial_perspective:
        initial_keywords.append(editorial_perspective.lower().replace(" ", "_"))
    if content_fingerprint:
        initial_keywords.append(f"fp_{content_fingerprint[:8]}")
    
    full_text = f"{title} {script} {sub_category}".lower()
    
    matched_tags = []
    
    # 1. Match Prominent People
    for person_key, meta in PEOPLE_METADATA.items():
        # Match complete name or surname with word boundary
        pattern = rf"\b{re.escape(person_key)}\b"
        # Also check just the last name if it is unique enough
        last_name = person_key.split()[-1]
        last_name_pattern = rf"\b{re.escape(last_name)}\b"
        
        if re.search(pattern, full_text) or (len(last_name) > 3 and re.search(last_name_pattern, full_text)):
            matched_tags.extend(meta["tags"])
            
    # 2. Add Initial Metadata to Tags
    for tag in initial_keywords + initial_companies + initial_people:
        if tag and len(tag) > 1:
            matched_tags.append(tag.strip())
            
    # 3. Match Specific Tools & Platforms
    for topic in TRENDING_AI_TOPICS:
        if re.search(rf"\b{re.escape(topic.lower())}\b", full_text):
            matched_tags.append(topic)
            
    # 4. Match Tech Sub-niches
    for niche in SPECIFIC_TECH_NICHES:
        if re.search(rf"\b{re.escape(niche.lower())}\b", full_text):
            matched_tags.append(niche)
 
    # 5. Extract additional category matches to fill tag slots
    has_ai = any(term in full_text for term in ["ai", "intelligence", "learning", "neural", "chatbot", "model", "llm"])
    has_tech = any(term in full_text for term in ["tech", "coding", "software", "developer", "programming", "python", "automation", "robot"])
    has_facts = any(term in full_text for term in ["fact", "know", "amazing", "discover", "breakthrough", "scary", "secret"])
 
    if has_ai:
        matched_tags.extend(BROAD_AI_TAGS[:4])
        matched_tags.extend(LONG_TAIL_TAGS[:2])
    if has_tech:
        matched_tags.extend(TECHNOLOGY_TAGS[:4])
    if has_facts:
        matched_tags.extend(FACTS_STYLE_TAGS[:3])
        matched_tags.extend(VIRAL_HOOK_TAGS[:2])
        
    # Fallback to general high-value tags to ensure variety and count
    matched_tags.extend(FACTS_STYLE_TAGS[:2])
    matched_tags.extend(BROAD_AI_TAGS[:2])
    matched_tags.extend(TECHNOLOGY_TAGS[:2])
 
    # 6. Deduplicate and Clean Tags
    seen_tags_lower = set()
    cleaned_tags = []
    for tag in matched_tags:
        # remove punctuation, extra whitespace
        tag_clean = re.sub(r'[^\w\s\-\.]', '', tag).strip()
        if not tag_clean or len(tag_clean) < 2:
            continue
        tag_lower = tag_clean.lower()
        if tag_lower not in seen_tags_lower:
            seen_tags_lower.add(tag_lower)
            cleaned_tags.append(tag_clean)
            
# YouTube Studio tags field constraint: Limit to 8-15 tags
    final_tags = cleaned_tags[:15]
    if len(final_tags) < 8:
        # Force fill up to 8 minimum
        defaults = ["did you know", "amazing facts", "tech facts", "AI", "technology", "artificial intelligence", "future tech"]
        for d in defaults:
            if d.lower() not in seen_tags_lower:
                final_tags.append(d)
                seen_tags_lower.add(d.lower())
                if len(final_tags) >= 8:
                    break
  
    # 7. Generate keyword-based hashtags (8-10 for Shorts, 12-15 for Longform)
    final_hashtags = generate_specific_hashtags(
        title=title,
        script=script,
        sub_category=sub_category,
        initial_keywords=initial_keywords,
        initial_companies=initial_companies,
        initial_people=initial_people,
        initial_hashtags=initial_hashtags,
        is_shorts=is_shorts,
        target_country=target_country
    )
    
    return {
        "tags": final_tags,
        "hashtags": final_hashtags
    }

