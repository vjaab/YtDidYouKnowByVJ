import datetime
from config import ENABLE_LONGFORM

def get_slot_info():
    """
    Returns (day_name, slot, category) based on current UTC time and the 2026 Mass-Appeal Strategy.
    With 1 upload/day, we always use Slot A. Category rotates by weekday.
    """
    now = datetime.datetime.utcnow()
    day_name = now.strftime("%a") # Mon, Tue, etc.
    
    # Single daily upload — always Slot A
    slot = "Slot A (Discovery)"
        
    # Daily category rotation — High Views & Subscribers Strategy
    daily_categories = {
        "Mon": "AI & Tech Tools",          # Massive trend, AI demos
        "Tue": "Facts & Trivia",           # Binge-worthy, Did you know
        "Wed": "Money & Side Hustles",     # High CPC, quick income ideas
        "Thu": "Motivation & Self-Help",   # High engagement, viral potential
        "Fri": "Life Hacks",               # Clever tips, productivity
        "Sat": "Agentic AI Facts",         # AI expertise, niche but high CPC
        "Sun": "Coding Hacks"              # AI expertise, Python/AI shortcuts
    }
    
    category = daily_categories.get(day_name, "AI & Tech Tools")
    
    return day_name, slot, category

def get_longform_slot_info():
    """
    Returns slot info for the daily long-form "Did You Know" pipeline.
    Always returns Slot L with a fixed category.
    """
    now = datetime.datetime.utcnow()
    day_name = now.strftime("%a")
    slot = "Slot L (Long-form)"
    category = "AI Did You Know"
    return day_name, slot, category

SERIES_MAP = {
    "Slot A": {"name": "⚡ TECH INSIDER",     "tagline": "Tips, tricks, and facts you need to know."},
    "Slot B": {"name": "🔬 HYBRID AI",       "tagline": "Local models. Production scale."},
    "Slot L": {"name": "🧠 DID YOU KNOW",    "tagline": "5 mind-blowing AI facts daily."},
}

def get_series_identity(slot):
    for key, val in SERIES_MAP.items():
        if key in slot:
            return val
    return {"name": "VJ TECH TIPS", "tagline": "Daily tech tips and tricks."}

def get_next_slot(current_slot):
    if "Slot A" in current_slot: return "Slot B"
    return "Slot A"

def get_category_prompt_enhancement(category, slot):
    """
    Returns specific instructions and formatting for the given category/slot.
    Targeting mass-market audiences for high views, subscribers, and CPC.
    """
    base_discovery = "FOCUS: Viral Discovery. Use broad, mass-appeal hooks. Target EVERYONE with a smartphone or computer."
    
    enhancements = {
        "AI & Tech Tools": f"""
            CATEGORY: AI & Tech Tools (Massive trend, viewers love AI demos).
            STRATEGY: {base_discovery}
            AUDIENCE: Anyone wanting to save time, increase productivity, or try cool AI. 
            GOAL: Show ONE specific free AI tool or hack that solves a real problem or does something amazing. (e.g., '3 Free AI Tools for Students', 'ChatGPT Hack You Didn't Know').
            HOOK STYLE: 'Stop paying for this app. This AI does it FREE.' or 'This secret AI tool will replace your job.'
            EMOTIONAL TRIGGER: "I NEED to try this right now" + time/money savings.
            CONTENT FORMAT: Quick demo (result-first). Show the tool working. No jargon.
        """,
        "Facts & Trivia": f"""
            CATEGORY: Facts & Trivia (Highly shareable, binge-worthy).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE. People who love learning surprising, weird, or hidden facts.
            GOAL: Share a "Did you know?" tech fact, a weird historical tech story, or debunk a common myth.
            HOOK STYLE: '99% of people don't know what this iPhone button actually does.' or 'The terrifying truth about Google's servers.'
            EMOTIONAL TRIGGER: "Wait, really?!" + curiosity + shareability.
            CONTENT FORMAT: Bold claim/hook → fascinating story/proof → mind-blown payoff.
        """,
        "Money & Side Hustles": f"""
            CATEGORY: Money & Side Hustles (Everyone wants quick income, High CPC).
            STRATEGY: {base_discovery}
            AUDIENCE: People looking for passive income, side hustles, or financial growth using tech/AI.
            GOAL: Provide actionable, realistic ways to make money online using AI or tech tools.
            HOOK STYLE: '3 AI side hustles to make $100 a day.' or 'How to make passive income using ChatGPT in 2026.'
            EMOTIONAL TRIGGER: Financial freedom + greed/ambition + "I can do that easily".
            CONTENT FORMAT: Hook → Step-by-step breakdown of the hustle → Proof/potential earnings → CTA.
        """,
        "Motivation & Self-Help": f"""
            CATEGORY: Motivation & Self-Help (High engagement, viral potential).
            STRATEGY: {base_discovery}
            AUDIENCE: Ambitious individuals, tech entrepreneurs, students, anyone needing inspiration.
            GOAL: Share a powerful success story, mindset tip, or motivational quote related to tech, AI, or startups.
            HOOK STYLE: 'How a 19-year-old built a $1M AI startup in his bedroom.' or 'The ONE habit that made Steve Jobs successful.'
            EMOTIONAL TRIGGER: Inspiration + aspiration + "I need to get to work".
            CONTENT FORMAT: Hook → Story of struggle/success → The core lesson/mindset shift → Motivational CTA.
        """,
        "Life Hacks": f"""
            CATEGORY: Life Hacks (Continued viewer interest, productivity).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE. Universal appeal for making daily life easier.
            GOAL: Share a clever tech tip, productivity hack, or DIY solution that saves time and effort.
            HOOK STYLE: 'This 10-second tech hack will save you 5 hours a week.' or 'The ultimate productivity trick for your laptop.'
            EMOTIONAL TRIGGER: Immediate utility + FOMO.
            CONTENT FORMAT: Fast, punchy, result-first. Show the hack in action immediately.
        """,
        "Agentic AI Facts": f"""
            CATEGORY: Agentic AI Facts (Niche but High CPC, AI/ML Background).
            STRATEGY: Leverage your Senior AI Engineer expertise while keeping it accessible.
            AUDIENCE: Tech enthusiasts, professionals, students, and curious minds.
            GOAL: Explain autonomous AI agents or complex AI facts fast. Make it sound mind-blowing but easy to grasp.
            HOOK STYLE: 'AI agents are about to run the internet. Here is what they actually do.' or 'What is an autonomous AI agent? Explained in 30 seconds.'
            EMOTIONAL TRIGGER: Future-shock + "I'm learning from an expert" + awe.
            CONTENT FORMAT: Hook → Simple everyday analogy for the AI concept → Real-world implication → CTA.
        """,
        "Coding Hacks": f"""
            CATEGORY: Coding Hacks (Niche but High CPC, AI/ML Background).
            STRATEGY: Leverage your Senior AI Engineer expertise. Provide extreme value quickly.
            AUDIENCE: Developers, CS students, tech workers, and people learning to code.
            GOAL: Share a Python trick, AI coding shortcut, or workflow improvement.
            HOOK STYLE: 'This Python trick will save you 100 lines of code.' or 'How to use AI to write your backend in 40 seconds.'
            EMOTIONAL TRIGGER: "I've been doing it the hard way!" + immediate utility.
            CONTENT FORMAT: Problem setup → Show the long way vs the Hack/AI way → Payoff/Result.
        """
    }
    
    return enhancements.get(category, enhancements.get("AI & Tech Tools", ""))

