"""
tags_helper.py — Dynamic metadata tags and hashtags optimizer for YouTube upload.
Matches title, script, and metadata against curated AI & tech tag categories and key figures.
"""
import re

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

# ── TOP 5 HASHTAGS FOR ENGLISH FACTS CHANNEL ──
DEFAULT_BROAD_HASHTAGS = ["#ArtificialIntelligence", "#Technology", "#MachineLearning"]
DEFAULT_NICHE_HASHTAGS = ["#AITools", "#TechFacts"]

def get_optimized_metadata(
    title, 
    script, 
    sub_category="", 
    initial_keywords=None, 
    initial_companies=None, 
    initial_people=None, 
    initial_hashtags=None
):
    """
    Computes an optimized list of 8-15 unique tags and 3-5 unique hashtags
    based on the content of the video.
    """
    initial_keywords = initial_keywords or []
    initial_companies = initial_companies or []
    initial_people = initial_people or []
    initial_hashtags = initial_hashtags or []
    
    full_text = f"{title} {script} {sub_category}".lower()
    
    matched_tags = []
    matched_hashtags = []
    
    # 1. Match Prominent People
    for person_key, meta in PEOPLE_METADATA.items():
        # Match complete name or surname with word boundary
        pattern = rf"\b{re.escape(person_key)}\b"
        # Also check just the last name if it is unique enough
        last_name = person_key.split()[-1]
        last_name_pattern = rf"\b{re.escape(last_name)}\b"
        
        if re.search(pattern, full_text) or (len(last_name) > 3 and re.search(last_name_pattern, full_text)):
            matched_tags.extend(meta["tags"])
            matched_hashtags.append(meta["hashtag"])
            
    # 2. Add Initial Metadata
    for tag in initial_keywords + initial_companies + initial_people:
        if tag and len(tag) > 1:
            matched_tags.append(tag.strip())
            
    for ht in initial_hashtags:
        ht_clean = ht.strip()
        if ht_clean.startswith("#"):
            matched_hashtags.append(ht_clean)
        elif ht_clean:
            matched_hashtags.append(f"#{ht_clean}")
            
    # 3. Match Specific Tools & Platforms
    for topic in TRENDING_AI_TOPICS:
        if re.search(rf"\b{re.escape(topic.lower())}\b", full_text):
            matched_tags.append(topic)
            # Add topic hashtag if relevant
            matched_hashtags.append(f"#{topic.replace(' ', '').replace('-', '')}")
            
    # 4. Match Tech Sub-niches
    for niche in SPECIFIC_TECH_NICHES:
        if re.search(rf"\b{re.escape(niche.lower())}\b", full_text):
            matched_tags.append(niche)
            matched_hashtags.append(f"#{niche.replace(' ', '').replace('-', '')}")

    # 5. Extract additional category matches to fill tag slots
    # We will score the category match
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

    # 7. Deduplicate and Clean Hashtags (Broad, Niche, Specific)
    seen_ht_lower = set()
    cleaned_hashtags = []
    
    # Standardize matched hashtags
    for ht in matched_hashtags:
        ht_clean = "#" + re.sub(r'[^\w]', '', ht).strip()
        if len(ht_clean) <= 2:
            continue
        ht_lower = ht_clean.lower()
        if ht_lower not in seen_ht_lower:
            seen_ht_lower.add(ht_lower)
            cleaned_hashtags.append(ht_clean)
            
    # Construct dynamic hashtags selection (3-5 items):
    # - We want a combination of:
    #   * Specific (matched from content)
    #   * Niche (from our pool: #AITools, #TechFacts, etc.)
    #   * Broad (from our pool: #ArtificialIntelligence, #Technology, #MachineLearning)
    
    specific_hashtags = [h for h in cleaned_hashtags if h not in DEFAULT_BROAD_HASHTAGS and h not in DEFAULT_NICHE_HASHTAGS]
    
    final_hashtags = []
    
    # A. Select 1-2 Specific hashtags first
    for sh in specific_hashtags:
        if len(final_hashtags) < 2:
            final_hashtags.append(sh)
            
    # B. Select 1-2 Niche hashtags
    for nh in DEFAULT_NICHE_HASHTAGS:
        nh_lower = nh.lower()
        # Prefer niche hashtags not already in final_hashtags
        if nh_lower not in [h.lower() for h in final_hashtags]:
            if len(final_hashtags) < 3:
                final_hashtags.append(nh)
                
    # C. Select 1-2 Broad hashtags
    for bh in DEFAULT_BROAD_HASHTAGS:
        bh_lower = bh.lower()
        if bh_lower not in [h.lower() for h in final_hashtags]:
            if len(final_hashtags) < 5:
                final_hashtags.append(bh)
                
    # D. Fillers if still less than 3
    if len(final_hashtags) < 3:
        for nh in DEFAULT_NICHE_HASHTAGS + DEFAULT_BROAD_HASHTAGS:
            if nh.lower() not in [h.lower() for h in final_hashtags]:
                final_hashtags.append(nh)
                if len(final_hashtags) >= 3:
                    break
                    
    # Limit to maximum 5
    final_hashtags = final_hashtags[:5]
    
    return {
        "tags": final_tags,
        "hashtags": final_hashtags
    }
