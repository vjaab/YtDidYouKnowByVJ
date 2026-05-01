import datetime

def get_slot_info():
    """
    Returns (day_name, slot, category) based on current UTC time and the 2026 Ecosystem Strategy.
    """
    now = datetime.datetime.utcnow()
    day_name = now.strftime("%a") # Mon, Tue, etc.
    hour = now.hour
    
    # Slot A: Morning (approx 5:30 UTC)
    # Slot B: Evening (approx 11:30 UTC)
    # We'll use a threshold of 9:00 UTC to split A and B
    if hour < 9:
        slot = "Slot A (Discovery)"
    elif hour < 15:
        slot = "Slot B (Utility)"
    else:
        slot = "Slot C (Deep-Dive)"
        
    # Schedule Matrix
    # (Day, Slot A Category, Slot B Category)
    matrix = {
        "Mon": ("AI News", "AI Money"),
        "Tue": ("AI vs AI", "AI Hands-on"),
        "Wed": ("AI Concepts", "AI Fails"),       # Fails mid-week for virality
        "Thu": ("AI Predictions", "AI Ethics"),   # Adds human angle
        "Fri": ("AI Quiz", "AI Career"),
        "Sat": ("AI Build", "AI Tool"),            # Project continuity on weekends
        "Sun": ("AI News", "AI Fails")             # Fails again — strong Sunday closer
    }
    
    categories = matrix.get(day_name, ("AI News", "AI Tool", "AI Deep-Dive"))
    
    if "Slot A" in slot:
        category = categories[0]
    elif "Slot B" in slot:
        category = categories[1]
    else:
        category = "AI Deep-Dive" # Dedicated Long-form slot
    
    return day_name, slot, category

SERIES_MAP = {
    "Slot A": {"name": "⚡ TOOL DROP",     "tagline": "New AI tool. 60 seconds."},
    "Slot B": {"name": "🔬 MODEL INTEL",   "tagline": "What the labs don't tell you."},
    "Slot C": {"name": "🧠 DEEP DECODE",   "tagline": "One concept. Fully broken down."},
}

def get_series_identity(slot):
    for key, val in SERIES_MAP.items():
        if key in slot:
            return val
    return {"name": "VJ AI NEWS", "tagline": "Daily AI intelligence."}

def get_next_slot(current_slot):
    if "Slot A" in current_slot: return "Slot B"
    if "Slot B" in current_slot: return "Slot C"
    return "Slot A"

def get_category_prompt_enhancement(category, slot):
    """
    Returns specific instructions and formatting for the given category/slot.
    """
    base_discovery = "FOCUS: Viral/News. Use broad, 'scary/exciting' high-velocity hooks. Target discovery and reach."
    base_utility = "FOCUS: Retention/Value. Use specific 'how-to' or interactive content. Target deep utility and subscriber loyalty."
    
    enhancements = {
        "AI News": f"""
            CATEGORY: AI News (The Sunday/Monday/Mid-week Leak).
            STRATEGY: {base_discovery}
            GOAL: Summarize the biggest breakthrough or leak. Why it matters to humans.
            HOOK: Start with 'AI News is moving faster than you think...' or similar high-intensity alarm.
        """,
        "AI Tool": f"""
            CATEGORY: AI Tool (Showcase/Spotlight).
            STRATEGY: {base_utility}
            GOAL: Spotlight 1 specific tool solving a real problem. Niche tools (Law, Medicine, Research) are encouraged.
            HOOK: 'Stop wasting hours on [Task]. I found a tool that does it in 1 click.'
        """,
        "AI Concepts": f"""
            CATEGORY: AI Concepts (The 'So What?').
            STRATEGY: {base_discovery}
            GOAL: Explain deep concepts like 'Agentic Workflows' or 'Serverless AI' in a simple but profound way.
            HOOK: 'You keep hearing about [Concept], but here is why it actually changes your life.'
        """,
        "AI Hands-on": f"""
            CATEGORY: AI Hands-on (Speed-run Coding/Fixes).
            STRATEGY: {base_utility}
            GOAL: Show a quick fix, safe-run code, or rapid-fire app building. 
            HOOK: 'I fixed 50 bugs in 10 seconds using Antigravity. Here is the secret workflow.'
        """,
        "AI Quiz": f"""
            CATEGORY: AI Quiz (Engagement/Bait).
            STRATEGY: {base_discovery if 'Slot A' in slot else base_utility}
            GOAL: 'AI or Real?' or 'Predict the Prompt'. Ask high-contrast questions.
            HOOK: 'Is this video real or generated? Comment your guess now!'
            COMMENT BAITING: NEVER reveal the answer until the last 3-5 seconds. Explicitly tell users to 'Pause and comment your guess now!'
        """,
        "AI vs AI": f"""
            CATEGORY: AI vs AI (Head-to-Head Comparison).
            STRATEGY: {base_discovery}
            GOAL: Compare two major AI models or tools directly (e.g., ChatGPT vs Gemini, Midjourney vs DALL-E). Focus on who wins and why.
            HOOK: '[Model A] just destroyed [Model B] in the ultimate test. Here is the proof.'
        """,
        "AI Money": f"""
            CATEGORY: AI Money (Monetization & Side Hustles).
            STRATEGY: {base_discovery}
            GOAL: Break down realistic ways people are making money with AI tools today.
            HOOK: 'This simple AI workflow is printing money for creators in 2026. Here is how it works.'
        """,
        "AI Fails": f"""
            CATEGORY: AI Fails (Humor & Limitations).
            STRATEGY: {base_discovery}
            GOAL: Highlight hilarious or consequential mistakes made by AI models. Keep it light but informative.
            HOOK: 'I asked AI to do [Task] and it completely lost its mind. Watch this disaster.'
        """,
        "AI Predictions": f"""
            CATEGORY: AI Predictions (Futurism).
            STRATEGY: {base_utility}
            GOAL: Discuss what AI might look like in 6 months to 5 years. Focus on AGI, agentic behaviors, or hardware.
            HOOK: 'If you think AI is crazy now, wait until you see what happens in [Year].'
        """,
        "AI Career": f"""
            CATEGORY: AI Career (Future of Work).
            STRATEGY: {base_utility}
            GOAL: Discuss jobs, skills to learn, and how to survive the AI boom without being replaced.
            HOOK: 'These 5 jobs will be gone by next year because of AI. Are you on the list?'
        """,
        "AI Ethics": f"""
            CATEGORY: AI Ethics (Human Angle).
            STRATEGY: {base_discovery}
            GOAL: Highlight the moral, legal, or societal implications of a new AI development.
            HOOK: 'AI just crossed a line we can never go back from. Here is what happened.'
        """,
        "AI Build": f"""
            CATEGORY: AI Build (Project/Code).
            STRATEGY: {base_utility}
            GOAL: Show a tangible project being built with AI. Focus on actionable insights for developers or creators.
            HOOK: 'I built a fully functional [App/Tool] in 5 minutes using AI. Watch how.'
            """,
        "AI Deep-Dive": """
            CATEGORY: AI Deep-Dive (Long-form Authority).
            STRATEGY: FOCUS: Deep Authority. Use 16:9 aspect ratio. Target 120-180 seconds.
            GOAL: Provide a comprehensive breakdown of a major AI movement or a weekly wrap-up. 
            HOOK: 'In today's Deep-Dive, we're looking at why the current AI cycle is actually changing...'
            """
    }
    
    return enhancements.get(category, "")
