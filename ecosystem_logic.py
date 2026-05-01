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
        "Mon": ("Model Benchmarks", "Agentic System Design"),
        "Tue": ("Cost-Optimized AI", "Local LMM Apps"),
        "Wed": ("Research-to-Production", "AI Infrastructure"),
        "Thu": ("Hybrid Architectures", "API Deep-Dive"),
        "Fri": ("SOTA Benchmarking", "Agentic Orchestration"),
        "Sat": ("Open Source Focus", "Build-with-VJ"),
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
    "Slot A": {"name": "⚡ AGENTIC OPS",     "tagline": "Architecting autonomous systems."},
    "Slot B": {"name": "🔬 HYBRID AI",       "tagline": "Local models. Production scale."},
    "Slot C": {"name": "🧠 SYSTEM DESIGN",   "tagline": "High-level deep dives for architects."},
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
        "Agentic System Design": f"""
            CATEGORY: Agentic System Design (The New Standard).
            STRATEGY: {base_discovery}
            GOAL: Focus on autonomous loops, self-correction, and tool-use architecture.
            HOOK: 'Static prompts are dead. Agentic loops are how you actually build production AI.'
        """,
        "Cost-Optimized AI": f"""
            CATEGORY: Cost-Optimized AI (The Margin Layer).
            STRATEGY: {base_utility}
            GOAL: Show how to drop API costs by 80% using local quantization (vLLM, Ollama, GGUF).
            HOOK: 'Your Claude bill is too high. Here is the local stack that matches its performance for $0.'
        """,
        "Local LMM Apps": f"""
            CATEGORY: Local LMM Apps (Privacy & Speed).
            STRATEGY: {base_utility}
            GOAL: Showcase Kokoro TTS, Whisper local, or Llama.cpp implementations.
            HOOK: 'SOTA performance with zero latency and total privacy. Let\'s build [App] locally.'
        """,
        "Research-to-Production": f"""
            CATEGORY: Research-to-Production (Theory to Code).
            STRATEGY: {base_discovery}
            GOAL: Summarize a new Arxiv paper and show the exact Python class needed to implement it.
            HOOK: 'This [Lab] paper just solved [Problem]. Here is the 10-line implementation.'
        """,
        "Hybrid Architectures": f"""
            CATEGORY: Hybrid Architectures (Professional Scale).
            STRATEGY: {base_discovery}
            GOAL: Explain combining local small models with frontier cloud models for efficiency.
            HOOK: 'Stop sending everything to GPT-4. This hybrid routing logic saves thousands.'
        """,
        "SOTA Benchmarking": f"""
            CATEGORY: SOTA Benchmarking (Data-Driven).
            STRATEGY: {base_utility}
            GOAL: Comparative analysis of the latest models on MMLU, HumanEval, and real-world dev tasks.
            HOOK: 'The benchmarks lie. Here is the real-world coding performance of [Model].'
        """,
        "Agentic Orchestration": f"""
            CATEGORY: Agentic Orchestration (Multi-Agent).
            STRATEGY: {base_discovery}
            GOAL: Focus on LangGraph, AutoGen, or CrewAI patterns for complex task solving.
            HOOK: 'One agent isn\'t enough. Here\'s how to choreograph a team of AI experts.'
        """,
        "Build-with-VJ": f"""
            CATEGORY: Build-with-VJ (Implementation).
            STRATEGY: {base_utility}
            GOAL: Rapid-fire coding session. Deploying a GitHub Action or a Python automated factory.
            HOOK: 'Let\'s build an automated [System] in under 60 seconds. Code in the bio.'
        """,
        "Open Source Focus": f"""
            CATEGORY: Open Source Focus (Freedom).
            STRATEGY: {base_discovery}
            GOAL: Highlighting local model sovereignty and community-lead breakthroughs.
            HOOK: 'Open source just passed a new milestone. You can now run [Model] on a laptop.'
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
