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
        
    # Daily category rotation — Mass-Appeal Tech (audience pool: 100M+ per category)
    daily_categories = {
        "Mon": "Hidden Phone Features",        # iPhone/Android hidden settings — everyone with a phone
        "Tue": "AI Tools You Need",            # Practical AI tool demos — saves time/money
        "Wed": "Tech Myths Busted",            # Debunking misconceptions — triggers comments
        "Thu": "Scary Tech Facts",             # Privacy/security warnings — fear-driven engagement
        "Fri": "You're Using It Wrong",        # Common tech mistakes — FOMO + ego trigger
        "Sat": "Free vs Paid Apps",            # Budget alternatives — universal appeal
        "Sun": "AI vs Reality"                 # AI transformation mashups — shareable entertainment
    }
    
    category = daily_categories.get(day_name, "Tech Tips")
    
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
    All categories now target MASS-MARKET audiences (not just ML engineers).
    """
    base_discovery = "FOCUS: Viral Discovery. Use broad, mass-appeal hooks. Target EVERYONE with a phone/computer."
    
    enhancements = {
        "Hidden Phone Features": f"""
            CATEGORY: Hidden Phone Features (iPhone/Android/Windows Secret Settings).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE who owns a smartphone or computer. NOT just engineers.
            GOAL: Reveal hidden settings, secret menus, or unknown features in popular devices (iPhone, Android, Windows, Chrome, WhatsApp). The viewer must feel like they discovered a "cheat code" for their phone.
            HOOK STYLE: 'Your iPhone has a secret menu Apple doesn't want you to find.' or 'Change this ONE setting and your phone gets 2x faster.'
            EMOTIONAL TRIGGER: FOMO (Fear of Missing Out) + feeling like a "power user"
            CONTENT FORMAT: Screen recording style. Show the exact steps. Result-first (show the outcome, then explain how).
        """,
        "AI Tools You Need": f"""
            CATEGORY: AI Tools You Need (Practical AI for Everyday People).
            STRATEGY: {base_discovery}
            AUDIENCE: Anyone who wants to save time or money. Students, professionals, small business owners. NOT just programmers.
            GOAL: Show ONE specific free AI tool that solves a real everyday problem. The viewer must be able to try it themselves immediately.
            HOOK STYLE: 'Stop paying for Photoshop. This AI does it FREE.' or 'I wrote a 10-page report in 30 seconds using this AI tool.'
            EMOTIONAL TRIGGER: "I NEED to try this right now" + time/money savings
            CONTENT FORMAT: Quick demo (result-first). Show the tool working. No jargon.
        """,
        "Tech Myths Busted": f"""
            CATEGORY: Tech Myths Busted (Debunking Common Misconceptions).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE. These are beliefs that 90% of people hold incorrectly.
            GOAL: Challenge a widely-held tech belief with proof. Make the viewer feel smart for learning the truth. Drive massive comment engagement from people arguing.
            HOOK STYLE: 'Closing apps does NOT save battery. Here's proof.' or 'Incognito mode does NOT make you private. Here's what actually happens.'
            EMOTIONAL TRIGGER: "Wait, I've been doing this wrong my whole life?!" + ego challenge
            CONTENT FORMAT: Bold claim → visual proof → mind-blown payoff. Extremely controversial hooks drive comments.
        """,
        "Scary Tech Facts": f"""
            CATEGORY: Scary Tech Facts (Privacy, Security, and Surveillance Warnings).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE who owns a device connected to the internet. This affects ALL viewers personally.
            GOAL: Reveal a terrifying privacy or security fact that makes the viewer feel personally at risk. Then show them how to protect themselves.
            HOOK STYLE: 'Your phone is recording you RIGHT NOW. Here's proof.' or 'Google knows where you were 3 years ago. Check this setting.'
            EMOTIONAL TRIGGER: Fear + urgency + "I need to fix this NOW"
            CONTENT FORMAT: Dramatic hook → proof/demonstration → "Here's how to protect yourself" (actionable fix).
        """,
        "You're Using It Wrong": f"""
            CATEGORY: You're Using It Wrong (Common Tech Mistakes Everyone Makes).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE. Target universal behaviors that most people do incorrectly.
            GOAL: Show ONE common tech mistake and the correct way to do it. The viewer must feel personally called out.
            HOOK STYLE: 'Stop charging your phone like this. You're killing your battery.' or 'You've been using Google wrong your entire life.'
            EMOTIONAL TRIGGER: Ego challenge + curiosity + "Am I doing this wrong too?"
            CONTENT FORMAT: Call-out the mistake → explain why it's wrong → show the correct way. Keep it punchy.
        """,
        "Free vs Paid Apps": f"""
            CATEGORY: Free vs Paid Apps (Budget Alternatives That Are Actually Better).
            STRATEGY: {base_discovery}
            AUDIENCE: Students, budget-conscious users, anyone paying for subscriptions they don't need.
            GOAL: Compare an expensive paid app/service with a FREE alternative that's just as good (or better). Make the viewer feel like they've been wasting money.
            HOOK STYLE: 'Stop paying $10/month for this app. This free one is BETTER.' or 'I replaced 5 paid apps with this ONE free tool.'
            EMOTIONAL TRIGGER: "I've been wasting money this whole time!" + immediate savings
            CONTENT FORMAT: Side-by-side comparison. Show the free alternative actually working. Name specific apps.
        """,
        "AI vs Reality": f"""
            CATEGORY: AI vs Reality (AI Transformation Mashups & Experiments).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE. Pure entertainment with a tech twist. Highly shareable.
            GOAL: Use AI to create a surprising, visual, or funny transformation. The viewer must want to share this with friends or try it themselves.
            HOOK STYLE: 'I asked AI to predict what I'll look like in 30 years. The result is INSANE.' or 'AI turned my bedroom photo into a luxury penthouse.'
            EMOTIONAL TRIGGER: "Wow, I want to try this!" + shareability + visual amazement
            CONTENT FORMAT: Visual transformation (before → AI result). Show the result FIRST, then how you did it. 15-25 seconds ideal.
        """,
        "Tech Tips": f"""
            CATEGORY: Tech Tips (General Technology Tips & Tricks).
            STRATEGY: {base_discovery}
            GOAL: Cover the single most useful, surprising, or mind-blowing tech tip of the day. Must be applicable to EVERYONE, not just engineers.
            HOOK STYLE: 'This ONE trick will change how you use your phone forever.' or 'I wish I knew this sooner. Here's the tech hack that saved me hours.'
            EMOTIONAL TRIGGER: Immediate utility + FOMO
            CONTENT FORMAT: Fast, punchy, result-first. Show the tip working.
        """
    }
    
    return enhancements.get(category, enhancements.get("Tech Tips", ""))

