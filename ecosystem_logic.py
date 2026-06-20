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
        "Mon": "AI & Tech Tools",             # Massive trend, AI demos
        "Tue": "Tech Gadgets & Inventions",   # Consumer tech gadgets, smart devices, new hardware
        "Wed": "Finance & Tech Economy",      # High-CPC personal finance, Fintech, crypto, market trends
        "Thu": "Facts & Trivia",              # Binge-worthy, Did you know
        "Fri": "Life Hacks & Productivity",   # Clever tips, device optimization
        "Sat": "Agentic AI Facts",            # AI expertise, niche but high CPC
        "Sun": "Coding & Development Hacks"   # Python/AI shortcuts, developer tools
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
        "Tech Gadgets & Inventions": f"""
            CATEGORY: Tech Gadgets & Inventions (High mass-market curiosity).
            STRATEGY: {base_discovery}
            AUDIENCE: Tech consumers, gadget lovers, early adopters, anyone wanting smart lifestyle upgrades.
            GOAL: Showcase a revolutionary new gadget, consumer tech hardware release, smart device leak, or physical tech invention.
            HOOK STYLE: 'This tiny gadget makes your home completely offline.' or 'Apple is secretly planning this new device...'
            EMOTIONAL TRIGGER: Futuristic awe + "I want to buy this" + curiosity gap.
            CONTENT FORMAT: Hook → Describe the gadget's unique feature/mechanism → How it improves daily life → Engagement/subscription CTA.
        """,
        "Finance & Tech Economy": f"""
            CATEGORY: Finance & Tech Economy (High-CPC personal finance, Fintech, tech economy trends).
            STRATEGY: {base_discovery}
            AUDIENCE: Retail investors, tech professionals, anyone seeking financial growth, wealth protection, and understanding tech markets.
            GOAL: Provide tech-driven personal finance tips, fintech tools, tech market breakdowns, crypto/market updates, or AI wealth hacks.
            HOOK STYLE: 'How to use AI to track your monthly budget automatically.' or 'The fintech tool banks don't want you to know about.'
            EMOTIONAL TRIGGER: Wealth maximization + immediate monetary utility + financial curiosity.
            CONTENT FORMAT: Hook → Specific fintech/investment tip or tool breakdown → Clear step-by-step application → Low-friction engagement CTA.
        """,
        "Life Hacks & Productivity": f"""
            CATEGORY: Life Hacks & Productivity (Universal appeal for efficiency).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE. Universal appeal for device optimization and time-saving workflows.
            GOAL: Share a clever phone/computer tip, productivity system, or workflow hack that saves hours.
            HOOK STYLE: 'Turn this setting off immediately to double your battery life.' or 'This 30-second workflow hack saves 5 hours of work.'
            EMOTIONAL TRIGGER: Immediate utility + FOMO.
            CONTENT FORMAT: Fast-paced, result-first. Show the settings/trick step-by-step immediately.
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
        "Coding & Development Hacks": f"""
            CATEGORY: Coding & Development Hacks (Leverages engineering background, high CPC).
            STRATEGY: Leverage your Senior AI Engineer expertise. Provide extreme programming value.
            AUDIENCE: Software engineers, developers, computer science students, and tech creators.
            GOAL: Share an advanced coding shortcut, Python optimization trick, or AI developer tool workflow.
            HOOK STYLE: 'Stop writing boilerplate code. Use this Python decorator trick instead.' or 'The open-source library that replaces 80% of your backend.'
            EMOTIONAL TRIGGER: "I've been doing it the hard way!" + engineering authority.
            CONTENT FORMAT: Problem setup → The traditional way vs the optimized/AI Hack way → Payoff/performance benchmark.
        """
    }
    
    return enhancements.get(category, enhancements.get("AI & Tech Tools", ""))

