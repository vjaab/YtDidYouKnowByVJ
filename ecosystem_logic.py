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
    else:
        slot = "Slot B (Utility)"
        
    # Schedule Matrix
    # (Day, Slot A Category, Slot B Category)
    matrix = {
        "Mon": ("AI News", "AI Tool"),
        "Tue": ("AI Concepts", "AI Hands-on"),
        "Wed": ("AI Quiz", "AI Tool"),
        "Thu": ("AI News", "AI Quiz"),
        "Fri": ("AI Concepts", "AI Hands-on"),
        "Sat": ("AI Quiz", "AI Tool"),
        "Sun": ("AI News", "AI Hands-on")
    }
    
    categories = matrix.get(day_name, ("AI News", "AI Tool"))
    category = categories[0] if "Slot A" in slot else categories[1]
    
    return day_name, slot, category

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
        """
    }
    
    return enhancements.get(category, "")
