import datetime
from config import ENABLE_LONGFORM

def get_slot_info():
    """
    Returns (day_name, slot, category) based on current UTC time and the 2026 Ecosystem Strategy.
    With 1 upload/day, we always use Slot A. Category rotates by weekday.
    """
    now = datetime.datetime.utcnow()
    day_name = now.strftime("%a") # Mon, Tue, etc.
    
    # Single daily upload — always Slot A
    slot = "Slot A (Discovery)"
        
    # Daily category rotation (one per day) — biased toward viral, mass-appeal topics
    daily_categories = {
        "Mon": "AI Industry Drama",          # Lawsuits, scandals, firings — highest viral potential
        "Tue": "AI vs Humans",               # Job replacement, AI outperforming humans
        "Wed": "Breakthrough AI Models",     # New model launches, benchmarks, comparisons
        "Thu": "AI Privacy & Security",      # Data leaks, surveillance, deepfakes
        "Fri": "Big Tech AI Wars",           # Google vs OpenAI vs Meta competition
        "Sat": "Open Source AI",             # Free models, community breakthroughs
        "Sun": "AI Weekly Recap"             # Week's biggest stories
    }
    
    category = daily_categories.get(day_name, "AI News")
    
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
    "Slot A": {"name": "⚡ AGENTIC OPS",     "tagline": "Architecting autonomous systems."},
    "Slot B": {"name": "🔬 HYBRID AI",       "tagline": "Local models. Production scale."},
    "Slot L": {"name": "🧠 DID YOU KNOW",    "tagline": "5 mind-blowing AI facts daily."},
}

def get_series_identity(slot):
    for key, val in SERIES_MAP.items():
        if key in slot:
            return val
    return {"name": "VJ AI NEWS", "tagline": "Daily AI intelligence."}

def get_next_slot(current_slot):
    if "Slot A" in current_slot: return "Slot B"
    return "Slot A"

def get_category_prompt_enhancement(category, slot):
    """
    Returns specific instructions and formatting for the given category/slot.
    """
    base_discovery = "FOCUS: Viral/News. Use broad, 'scary/exciting' high-velocity hooks. Target discovery and reach."
    base_utility = "FOCUS: Retention/Value. Use specific 'how-to' or interactive content. Target deep utility and subscriber loyalty."
    
    enhancements = {
        "AI Industry Drama": f"""
            CATEGORY: AI Industry Drama (Lawsuits, Scandals, Firings).
            STRATEGY: {base_discovery}
            GOAL: Focus on corporate lawsuits, executive firings, internal leaks, or ethical scandals in AI companies. These stories trigger strong emotional reactions.
            HOOK: 'This company just got DESTROYED by their own AI. Here is the $100M lawsuit nobody saw coming.'
        """,
        "AI vs Humans": f"""
            CATEGORY: AI vs Humans (Job Displacement, AI Outperformance).
            STRATEGY: {base_discovery}
            GOAL: Stories about AI replacing human workers, outperforming experts, or threatening entire industries. Make it PERSONAL to the viewer.
            HOOK: 'Your job might be next. This AI just replaced 700 workers... overnight.'
        """,
        "Breakthrough AI Models": f"""
            CATEGORY: Breakthrough AI Models (New Releases & Benchmarks).
            STRATEGY: {base_discovery}
            GOAL: Cover new model launches, shocking benchmark results, or unexpected capabilities. Focus on why this changes everything.
            HOOK: 'This new model just DESTROYED GPT-4 on every benchmark. And it's free.'
        """,
        "AI Privacy & Security": f"""
            CATEGORY: AI Privacy & Security (Data Leaks, Surveillance, Deepfakes).
            STRATEGY: {base_discovery}
            GOAL: Expose AI-powered surveillance, data breaches, deepfake dangers, or privacy violations. Make the viewer feel personally at risk.
            HOOK: 'Your face is in an AI database right now. Here is how they got it... without your consent.'
        """,
        "Big Tech AI Wars": f"""
            CATEGORY: Big Tech AI Wars (Google vs OpenAI vs Meta).
            STRATEGY: {base_discovery}
            GOAL: Cover the competitive moves between tech giants. Acquisitions, poaching, one-upmanship, and strategic pivots.
            HOOK: 'Google just declared WAR on OpenAI. Here is the nuclear weapon they're building.'
        """,
        "Open Source AI": f"""
            CATEGORY: Open Source AI (Community Breakthroughs).
            STRATEGY: {base_discovery}
            GOAL: Highlight open-source models matching or beating paid APIs. Sovereignty, privacy, zero-cost alternatives.
            HOOK: 'Open source just passed a new milestone. You can now run THIS on a laptop... for free.'
        """,
        "AI Weekly Recap": """
            CATEGORY: AI Weekly Recap (The Wrap-up).
            STRATEGY: FOCUS: Deep Authority. Target 50-58 seconds.
            GOAL: Summarize the single biggest technical movement of the week that affects everyday people.
            HOOK: 'If you missed THIS story, you missed the biggest AI shift of the year.'
        """,
        "AI News": f"""
            CATEGORY: AI News (General/Breaking).
            STRATEGY: {base_discovery}
            GOAL: Cover the single most surprising, controversial, or impactful AI story of the day. Prioritize mass-appeal over niche technical depth.
            HOOK: 'Nobody is talking about this... but it changes EVERYTHING about AI.'
        """
    }
    
    return enhancements.get(category, enhancements.get("AI News", ""))
