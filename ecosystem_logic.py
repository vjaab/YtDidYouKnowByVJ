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
        "Mon": ("Model Benchmarks", "Prompt Engineering"),
        "Tue": ("Dev Tooling", "Workflow Optimization"),
        "Wed": ("Architecture Patterns", "AI Infrastructure"),
        "Thu": ("Library Spotlight", "API Deep-Dive"),
        "Fri": ("Paper Breakdown", "SOTA Analysis"),
        "Sat": ("AI Build", "Open Source Focus"),
        "Sun": ("Weekly Recap (Dev)", "Future Roadmap")
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
    "Slot A": {"name": "⚡ DEV STACK",      "tagline": "Tooling the future of AI."},
    "Slot B": {"name": "🔬 ENGINE ROOM",    "tagline": "How the models actually work."},
    "Slot C": {"name": "🧠 ARCHITECTURE",   "tagline": "Deep-dives for systems engineers."},
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
        "Model Benchmarks": f"""
            CATEGORY: Model Benchmarks (The Truth Layer).
            STRATEGY: {base_discovery}
            GOAL: Break down new model performance. Forget the marketing; focus on token speed, reasoning accuracy, and code-gen reliability.
            HOOK: 'Everyone is hype about [Model], but here's how it actually handles a real-world edge case.'
        """,
        "Prompt Engineering": f"""
            CATEGORY: Prompt Engineering (The Interface).
            STRATEGY: {base_utility}
            GOAL: Share a specific advanced prompting technique like Chain-of-Thought, Metaprompting, or XML-structured inputs.
            HOOK: 'Stop using basic prompts. This structural change boosted my AI's accuracy by 40%.'
        """,
        "Dev Tooling": f"""
            CATEGORY: Dev Tooling (The Workbench).
            STRATEGY: {base_utility}
            GOAL: Spotlight a specific IDE extension, CLI tool, or SDK that speeds up AI-assisted development.
            HOOK: 'I found the missing link in my dev stack. Every AI engineer needs [Tool].'
        """,
        "Workflow Optimization": f"""
            CATEGORY: Workflow Optimization (Scale).
            STRATEGY: {base_discovery}
            GOAL: Discuss how to integrate AI agents into CI/CD, testing, or documentation pipelines.
            HOOK: 'Don't just use AI to write code. Here is how I automated my entire [Workflow].'
        """,
        "Architecture Patterns": f"""
            CATEGORY: Architecture Patterns (The Blueprint).
            STRATEGY: {base_discovery}
            GOAL: Explain deep patterns like RAG, Agentic Orchestration, or Vector Database sharding.
            HOOK: 'Retrieval is easy. RAG at scale is hard. Here's how top labs solve [Problem].'
        """,
        "AI Infrastructure": f"""
            CATEGORY: AI Infrastructure (The Metal).
            STRATEGY: {base_utility}
            GOAL: Discuss GPU orchestration, serverless inference, or cost-optimization strategies for LLMs.
            HOOK: 'Your inference costs are killing your margins. Use this [Strategy] to save 70%.'
        """,
        "Library Spotlight": f"""
            CATEGORY: Library Spotlight (The Building Blocks).
            STRATEGY: {base_utility}
            GOAL: Quick-fire breakdown of a new Python/JS library that every developer should know about.
            HOOK: 'If you're not using [Library] for your AI apps, you're building them the hard way.'
        """,
        "API Deep-Dive": f"""
            CATEGORY: API Deep-Dive (Integration).
            STRATEGY: {base_discovery}
            GOAL: Real-world integration guide for major APIs (OpenAI, Anthropic, Gemini). Focus on rate limits and error handling.
            HOOK: 'Most devs ignore [API Feature]. Here's how to use it to build [App Type].'
        """,
        "Paper Breakdown": f"""
            CATEGORY: Paper Breakdown (The Foundation).
            STRATEGY: {base_discovery}
            GOAL: Translate a complex Arxiv paper into 3 actionable takeaways for working developers.
            HOOK: 'This new research from [Lab] just made [Previous Tech] obsolete. Here's what changed.'
        """,
        "SOTA Analysis": f"""
            CATEGORY: SOTA Analysis (Edge).
            STRATEGY: {base_utility}
            GOAL: State-of-the-art analysis on specific domains: Vision, Audio, or Multi-modal models.
            HOOK: 'We just hit a new ceiling in [Domain]. Here is the benchmark that proves it.'
        """,
        "AI Build": f"""
            CATEGORY: AI Build (The Grind).
            STRATEGY: {base_utility}
            GOAL: Show a tangible project being built with AI. Focus on the actual implementation steps.
            HOOK: 'I built a [Project] using [Stack] in just 10 minutes. Watch the full build.'
        """,
        "Open Source Focus": f"""
            CATEGORY: Open Source Focus (Community).
            STRATEGY: {base_discovery}
            GOAL: Highlight major movements in the open-source AI world (Llama, Mistral, ComfyUI).
            HOOK: 'Open source just caught up to [Proprietary Model]. Here is how you can run it locally.'
        """,
        "Weekly Recap (Dev)": """
            CATEGORY: Weekly Recap (The Wrap-up).
            STRATEGY: FOCUS: Deep Authority. Target 120-180 seconds.
            GOAL: Summarize the 3 biggest technical movements of the week for working developers.
            HOOK: 'If you missed the [Event] leak, you missed the biggest change in AI dev this year.'
        """,
        "Future Roadmap": """
            CATEGORY: Future Roadmap (Roadmap).
            STRATEGY: FOCUS: Deep Authority. Target 120-180 seconds.
            GOAL: Long-term technical predictions. AGI timelines, hardware shifts, and language evolution.
            HOOK: 'In today's Deep-Dive, we're looking at the technical roadmap to AGI...'
        """
    }
    
    return enhancements.get(category, "")
