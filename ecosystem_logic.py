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
    Now includes AI_HACK_STRATEGY, AI_HACK_HOOKS, AI_TOOL_DEMO_FORMAT for all categories.
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
            
            AI_HACK_STRATEGY: Focus on immediately accessible, zero-cost AI workflows.
                • Local LLMs: Ollama + Llama 3.1/Phi-3/Mistral — run entirely offline, no API keys
                • Free tier APIs: Groq (instant Llama 3.1), Together AI, Hugging Face Inference API
                • Multi-agent in 5 min: AutoGen / CrewAI / LangGraph templates for research, coding, writing
                • RAG in 5 min: LlamaIndex / Chroma / FAISS + local embeddings — chat with your PDFs/Notion
                • AI coding: Cursor tab completion, Cline (autonomous), Aider (terminal), Claude Code
                • Voice AI: Whisper.cpp (local STT) + Llama 3.1 + Piper/Coqui TTS = fully local voice assistant
            
            AI_HACK_HOOKS:
                1. "I ran Llama 3.1 70B on my LAPTOP for FREE — here's how"
                2. "This 5-minute setup lets you chat with ANY PDF offline"
                3. "Stop paying for ChatGPT Plus — Groq runs Llama 3.1 INSTANTLY for free"
                4. "I built an AI agent team that researches/writes/codes while I sleep"
                5. "Your phone can run an LLM NOW. No internet needed."
            
            AI_TOOL_DEMO_FORMAT (15-30s Short):
                1. Hook (0-1s): Show jaw-dropping result first (e.g., agent writing full app)
                2. "Here's the exact setup" (1-3s): One-liner install command on screen
                3. Live demo (3-20s): Real-time screen capture — prompt → result
                4. "It's completely free/local" (20-22s): Emphasize zero cost/privacy
                5. CTA (22-30s): "Comment 'OLLAMA' for my config file" / "Link in bio"
""",
        "Tech Gadgets & Inventions": f"""
            CATEGORY: Tech Gadgets & Inventions (High mass-market curiosity).
            STRATEGY: {base_discovery}
            AUDIENCE: Tech consumers, gadget lovers, early adopters, anyone wanting smart lifestyle upgrades.
            GOAL: Showcase a revolutionary new gadget, consumer tech hardware release, smart device leak, or physical tech invention.
            HOOK STYLE: 'This tiny gadget makes your home completely offline.' or 'Apple is secretly planning this new device...'
            EMOTIONAL TRIGGER: Futuristic awe + "I want to buy this" + curiosity gap.
            CONTENT FORMAT: Hook → Describe the gadget's unique feature/mechanism → How it improves daily life → Engagement/subscription CTA.
            
            AI_HACK_STRATEGY: Focus on AI-powered hardware, local inference on devices, and smart home automation with LLMs.
                • Home Assistant + Local LLM: Fully private voice control (Ollama + Wyoming Satellite + Piper TTS)
                • Raspberry Pi 5 / Jetson Orin Nano: Run Llama 3.1 8B / Phi-3 locally for home automation
                • ESP32-S3 / ESP-EYE + LLM: Microcontroller-level AI (keyword spotting, simple commands)
                • AI Wearables: Limitless Pendant, Humane Ai Pin, PLAUD Note — what actually works
                • Frame AI glasses / Brilliant Labs: Open-source AI glasses development
                • NVIDIA Jetson + Isaac ROS: Robotics + AI for home robots
                • Local computer vision: YOLOv8/YOLO-NAS on Pi for person/package/pet detection
            
            AI_HACK_HOOKS:
                1. "I replaced Alexa with a LOCAL LLM — zero cloud, total privacy"
                2. "This $35 Raspberry Pi runs an AI that controls my ENTIRE house"
                3. "AI glasses that actually work? Brilliant Labs Frame hands-on"
                4. "ESP32 + Llama.cpp = AI on a $6 microcontroller. Wild."
                5. "I built a local AI security camera that detects packages/people — no cloud"
            
            AI_TOOL_DEMO_FORMAT (15-30s Short):
                1. Hook (0-1s): Show the AI gadget doing something impressive (voice control, detection)
                2. Hardware reveal (1-3s): "This is a $35 Pi 5 + $15 mic"
                3. Live demo (3-20s): Speak command → local LLM → Home Assistant action
                4. "100% offline, your data never leaves" (20-22s)
                5. CTA (22-30s): "Want my Home Assistant config? Comment 'LOCAL'"
""",
        "Finance & Tech Economy": f"""
            CATEGORY: Finance & Tech Economy (High-CPC personal finance, Fintech, tech economy trends).
            STRATEGY: {base_discovery}
            AUDIENCE: Retail investors, tech professionals, anyone seeking financial growth, wealth protection, and understanding tech markets.
            GOAL: Provide tech-driven personal finance tips, fintech tools, tech market breakdowns, crypto/market updates, or AI wealth hacks.
            HOOK STYLE: 'How to use AI to track your monthly budget automatically.' or 'The fintech tool banks don't want you to know about.'
            EMOTIONAL TRIGGER: Wealth maximization + immediate monetary utility + financial curiosity.
            CONTENT FORMAT: Hook → Specific fintech/investment tip or tool breakdown → Clear step-by-step application → Low-friction engagement CTA.
            
            AI_HACK_STRATEGY: Focus on AI agents for personal finance, Code Interpreter for analysis, and API automation.
                • AI Budgeting Agent: YNAB/Monarch API + LLM = automatic categorization + insights + alerts
                • Tax Optimization: LLM + tax code RAG = personalized deduction finder (not advice, education)
                • Portfolio Analysis: ChatGPT Code Interpreter / Claude Artifacts = risk metrics, correlation, rebalancing
                • Fintech Automation: Plaid/Yodlee + LLM = transaction enrichment, subscription detection, negotiation scripts
                • Crypto/Stock Sentiment: RSS/Reddit/Twitter → embedding → LLM classification → daily briefing
                • Bill Negotiation: LLM generates Comcast/Verizon retention scripts + calls via Twilio
                • RAG on SEC Filings: LlamaIndex + 10-K/10-Q = instant earnings analysis
            
            AI_HACK_HOOKS:
                1. "My AI agent just saved me $400/mo on subscriptions — here's the prompt"
                2. "I fed my 10-K filings into an LLM. It found risks analysts missed."
                3. "Automated portfolio rebalancing with Code Interpreter — no advisor fees"
                4. "This AI reads your bank statements and finds tax deductions automatically"
                5. "Daily crypto/stock sentiment briefing generated by AI while you sleep"
            
            AI_TOOL_DEMO_FORMAT (15-30s Short):
                1. Hook (0-1s): "$400 saved / 15% return / 2hrs saved" — specific number
                2. "The AI workflow" (1-3s): Diagram on screen: Data Source → LLM → Action
                3. Live demo (3-20s): Paste transactions → AI categorizes + finds waste → output
                4. "Cost: $0.03 in API calls" (20-22s)
                5. CTA (22-30s): "Want the prompt? Comment 'FINANCE'"
""",
        "Facts & Trivia": f"""
            CATEGORY: Facts & Trivia (Highly shareable, binge-worthy).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE. People who love learning surprising, weird, or hidden facts.
            GOAL: Share a "Did you know?" tech fact, a weird historical tech story, or debunk a common myth.
            HOOK STYLE: '99% of people don't know what this iPhone button actually does.' or 'The terrifying truth about Google's servers.'
            EMOTIONAL TRIGGER: "Wait, really?!" + curiosity + shareability.
            CONTENT FORMAT: Bold claim/hook → fascinating story/proof → mind-blown payoff.
            
            AI_HACK_STRATEGY: Focus on AI-powered fact discovery, verification, and generation pipelines.
                • Wikipedia/Common Crawl RAG: LlamaIndex + Chroma = "Ask me anything" fact bot with citations
                • Automated Fact-Checking: Claim → Google Search API → LLM verification → confidence score
                • Trivia Generation: LLM + knowledge graph (Wikidata) → infinite unique "Did you know?" cards
                • Hallucination Detection: Self-consistency sampling + retrieval verification = reliability score
                • Historical Tech Stories: LLM + archive.org/API = obscure but verified tech history narratives
                • Myth Busting Pipeline: Common misconception → evidence retrieval → LLM verdict + sources
                • Daily Fact Newsletter: Automated pipeline (RSS → embed → LLM curate → email/Telegram)
            
            AI_HACK_HOOKS:
                1. "I built an AI that fact-checks viral tweets in REAL-TIME. Here's what it caught."
                2. "This AI generates 1000 verified 'Did you know?' facts per minute. Mind blown."
                3. "I asked an LLM to find the WEIRDEST tech fact in Wikipedia. You won't believe #3."
                4. "Automated myth-busting: AI vs 50 common tech myths. The results shocked me."
                5. "This bot reads 10,000 tech papers daily and gives me 3 mind-blowing facts every morning"
            
            AI_TOOL_DEMO_FORMAT (15-30s Short):
                1. Hook (0-1s): "This fact sounds fake but it's 100% verified" — show fact card
                2. "How I verified it" (1-3s): Pipeline diagram: Claim → Search → Sources → Verdict
                3. Live demo (3-20s): Type claim → AI returns confidence + 3 citations
                4. "Zero hallucinations. Here's the source." (20-22s)
                5. CTA (22-30s): "Want the fact-checking prompt? Comment 'VERIFY'"
""",
        "Life Hacks & Productivity": f"""
            CATEGORY: Life Hacks & Productivity (Universal appeal for efficiency).
            STRATEGY: {base_discovery}
            AUDIENCE: EVERYONE. Universal appeal for device optimization and time-saving workflows.
            GOAL: Share a clever phone/computer tip, productivity system, or workflow hack that saves hours.
            HOOK STYLE: 'Turn this setting off immediately to double your battery life.' or 'This 30-second workflow hack saves 5 hours of work.'
            EMOTIONAL TRIGGER: Immediate utility + FOMO.
            CONTENT FORMAT: Fast-paced, result-first. Show the settings/trick step-by-step immediately.
            
            AI_HACK_STRATEGY: Focus on AI-powered capture, organization, summarization, and automation.
                • Voice Capture → Notion/Obsidian: Whisper (local/Cloud) + LLM structuring + API = hands-free second brain
                • Email Triage Agent: Gmail API + LLM = auto-label, draft replies, unsubscribe, summarize threads
                • Meeting Summarizer: Whisper.cpp (local STT) + Llama 3.1 = action items, decisions, transcript
                • Calendar Optimization: LLM analyzes calendar → suggests focus blocks, buffers, meeting prep
                • Obsidian + RAG: Local vault + embeddings = "Ask my notes anything" + auto-linking
                • RSS/Newsletter → AI Digest: Feedparser + LLM = daily 3-min personalized briefing
                • Browser Automation: Browser-use / Stagehand + LLM = form filling, research, data extraction
                • Screenpipe / Rewind.ai alternative: Local screen recording + OCR + LLM = full recall
            
            AI_HACK_HOOKS:
                1. "I haven't typed a note in 6 months. My voice AI organizes everything automatically."
                2. "This AI reads my emails, drafts replies, and unsubscribes me from spam. Zero inbox."
                3. "My meetings summarize THEMSELVES now. Whisper + Llama = magic."
                4. "Obsidian + Local LLM = a second brain that ANSWERS questions from my notes."
                5. "I automated my entire morning briefing. AI reads 50 newsletters, gives me 3 mins of gold."
            
            AI_TOOL_DEMO_FORMAT (15-30s Short):
                1. Hook (0-1s): Show result — clean Notion page / empty inbox / meeting summary
                2. "The 3-tool stack" (1-3s): Whisper → LLM → Notion API (logos on screen)
                3. Live demo (3-20s): Speak → text appears organized in Notion / email drafted
                4. "Runs locally. Private. Free." (20-22s)
                5. CTA (22-30s): "My n8n workflow / Python script? Comment 'CAPTURE'"
""",
        "Agentic AI Facts": f"""
            CATEGORY: Agentic AI Facts (Niche but High CPC, AI/ML Background).
            STRATEGY: Leverage your Senior AI Engineer expertise while keeping it accessible.
            AUDIENCE: Tech enthusiasts, professionals, students, and curious minds.
            GOAL: Explain autonomous AI agents or complex AI facts fast. Make it sound mind-blowing but easy to grasp.
            HOOK STYLE: 'AI agents are about to run the internet. Here is what they actually do.' or 'What is an autonomous AI agent? Explained in 30 seconds.'
            EMOTIONAL TRIGGER: Future-shock + "I'm learning from an expert" + awe.
            CONTENT FORMAT: Hook → Simple everyday analogy for the AI concept → Real-world implication → CTA.
            
            AI_HACK_STRATEGY: Focus on practical agent patterns, frameworks, and debugging techniques.
                • Frameworks: AutoGen (multi-agent chat), CrewAI (role-based), LangGraph (stateful graphs), OpenAI Assistants API
                • Core Patterns: Planning → Execution → Reflection (Reflexion), Tool Use (function calling), Memory (short/long-term), Multi-agent debate
                • Reflexion/Reflexion: Agent critiques own output → revises → improves (Shinn et al.)
                • Tool-Use Schemas: Pydantic models for structured function calling, validation, retries
                • Human-in-the-Loop: Approval gates, clarification requests, escalation policies
                • Debugging Agents: LangSmith / LangFuse / Arize Phoenix = traces, evals, prompt versioning
                • Agent Swarms: Hierarchical (manager + workers), Peer-to-peer (debate), Sequential (pipeline)
                • Production Patterns: Observability, fallback models, cost controls, rate limiting, evals
            
            AI_HACK_HOOKS:
                1. "I built an agent that writes, tests, and deploys code. It just shipped a feature."
                2. "Reflexion: The technique that makes agents 40% smarter by CRITIQUING themselves."
                3. "AutoGen vs CrewAI vs LangGraph — I tested all 3 for a week. Here's the winner."
                4. "Why your agents fail: The 3 debugging tools you're not using (LangSmith, Phoenix, Fuse)"
                5. "Multi-agent debate: Two LLMs argue. The result? 95% accuracy on complex reasoning."
            
            AI_TOOL_DEMO_FORMAT (15-30s Short):
                1. Hook (0-1s): Show agent completing complex task (code + test + PR)
                2. Architecture diagram (1-3s): Planner → Coder → Tester → Critic → Done
                3. Live demo (3-20s): "Build a snake game" → agents collaborate → working game
                4. "Cost: $0.12. Time: 47 seconds." (20-22s)
                5. CTA (22-30s): "Want my agent config? Comment 'AGENT'"
""",
        "Coding & Development Hacks": f"""
            CATEGORY: Coding & Development Hacks (Leverages engineering background, high CPC).
            STRATEGY: Leverage your Senior AI Engineer expertise. Provide extreme programming value.
            AUDIENCE: Software engineers, developers, computer science students, and tech creators.
            GOAL: Share an advanced coding shortcut, Python optimization trick, or AI developer tool workflow.
            HOOK STYLE: 'Stop writing boilerplate code. Use this Python decorator trick instead.' or 'The open-source library that replaces 80% of your backend.'
            EMOTIONAL TRIGGER: "I've been doing it the hard way!" + engineering authority.
            CONTENT FORMAT: Problem setup → The traditional way vs the optimized/AI Hack way → Payoff/performance benchmark.
            
            AI_HACK_STRATEGY: Focus on AI-assisted development workflows, prompts, and tools that ship code faster.
                • AI Code Review: Claude/GPT-4 prompt for security, performance, style, architecture review
                • Test Generation: pytest + LLM = unit tests, property-based tests, mutation testing prompts
                • Refactoring Agents: Legacy code → LLM + AST (LibCST/Rope) → typed, tested, modernized
                • Cursor/Claude Code Workflows: .cursorrules, Cline autonomous, Aider terminal, Continue.dev
                • PR Automation: GitHub Actions + LLM = description, review, changelog, release notes
                • Documentation: LLM reads code → writes docstrings, README, ADR, architecture diagrams (Mermaid)
                • Migration Agents: Python 2→3, JS→TS, REST→GraphQL, monolith→microservices with LLM + codemods
                • Performance Profiling: py-spy / perf + LLM = bottleneck analysis + optimization suggestions
                • Regex/SQL/Config Generation: Natural language → complex patterns with verification
            
            AI_HACK_HOOKS:
                1. "This AI prompt reviews my code BETTER than senior engineers. Here's the exact prompt."
                2. "I migrated 50k lines from JS to TS in 2 hours. AI codemods + AST = magic."
                3. "Stop writing tests. This prompt generates 95% coverage including EDGE CASES."
                4. "Cursor .cursorrules file that makes AI write production-ready code every time."
                5. "Legacy spaghetti → clean architecture. My refactoring agent does it while I sleep."
            
            AI_TOOL_DEMO_FORMAT (15-30s Short):
                1. Hook (0-1s): "1000 lines of tests generated in 3 seconds" / "Security bug found in production code"
                2. Show the prompt/config (1-3s): Exact prompt or .cursorrules on screen
                3. Live demo (3-20s): Paste messy code → AI outputs clean, typed, tested version
                4. Diff view: "Red = before. Green = after." (20-22s)
                5. CTA (22-30s): "Want my prompt library? Comment 'CODE'"
""",
    }
    
    return enhancements.get(category, enhancements.get("AI & Tech Tools", ""))


# ─────────────────────────────────────────────────────────────────────────────
# AI HACKS STRUCTURED DATA — For downstream script/video generation pipelines
# ─────────────────────────────────────────────────────────────────────────────

_AI_HACKS_DATA = {
    "AI & Tech Tools": {
        "strategy": "Focus on immediately accessible, zero-cost AI workflows. Local LLMs (Ollama), free tier APIs (Groq, Together), multi-agent (AutoGen/CrewAI/LangGraph), 5-min RAG (LlamaIndex/Chroma), AI coding (Cursor/Cline/Aider), local voice (Whisper.cpp + Llama + Piper).",
        "hooks": [
            "I ran Llama 3.1 70B on my LAPTOP for FREE — here's how",
            "This 5-minute setup lets you chat with ANY PDF offline",
            "Stop paying for ChatGPT Plus — Groq runs Llama 3.1 INSTANTLY for free",
            "I built an AI agent team that researches/writes/codes while I sleep",
            "Your phone can run an LLM NOW. No internet needed."
        ],
        "demo_format": "Hook (result first) → One-liner install → Live demo (prompt→result) → 'Free/local' emphasis → CTA with comment trigger",
        "tools": ["Ollama", "Groq", "Together AI", "AutoGen", "CrewAI", "LangGraph", "LlamaIndex", "Chroma", "FAISS", "Cursor", "Cline", "Aider", "Whisper.cpp", "Piper TTS"],
        "prompts": [
            "You are a senior engineer. Review this code for security, performance, and maintainability.",
            "Generate comprehensive pytest tests including edge cases and property-based tests.",
            "Refactor this legacy code to modern patterns with type hints and documentation."
        ]
    },
    "Tech Gadgets & Inventions": {
        "strategy": "Focus on AI-powered hardware, local inference on devices, and smart home automation with LLMs. Home Assistant + Local LLM (Ollama + Wyoming + Piper), Raspberry Pi 5 / Jetson Orin for local inference, ESP32-S3 for microcontroller AI, AI wearables (Limitless, Humane, PLAUD), Frame AI glasses, local CV (YOLOv8 on Pi).",
        "hooks": [
            "I replaced Alexa with a LOCAL LLM — zero cloud, total privacy",
            "This $35 Raspberry Pi runs an AI that controls my ENTIRE house",
            "AI glasses that actually work? Brilliant Labs Frame hands-on",
            "ESP32 + Llama.cpp = AI on a $6 microcontroller. Wild.",
            "I built a local AI security camera that detects packages/people — no cloud"
        ],
        "demo_format": "Hook (gadget in action) → Hardware reveal → Live voice→action demo → '100% offline' emphasis → CTA with config share",
        "tools": ["Home Assistant", "Ollama", "Wyoming Satellite", "Piper TTS", "Raspberry Pi 5", "Jetson Orin Nano", "ESP32-S3", "ESP-EYE", "YOLOv8", "Frame AI Glasses", "Limitless Pendant", "PLAUD Note"],
        "prompts": [
            "You are a home automation expert. Convert this voice command into Home Assistant service calls.",
            "Analyze this camera frame and return: person/package/pet/vehicle detected with confidence."
        ]
    },
    "Finance & Tech Economy": {
        "strategy": "Focus on AI agents for personal finance, Code Interpreter for analysis, and API automation. YNAB/Monarch API + LLM budgeting, tax optimization via RAG, portfolio analysis with Code Interpreter, Plaid + LLM transaction enrichment, crypto/stock sentiment via RSS→embedding→LLM, SEC filing RAG with LlamaIndex.",
        "hooks": [
            "My AI agent just saved me $400/mo on subscriptions — here's the prompt",
            "I fed my 10-K filings into an LLM. It found risks analysts missed.",
            "Automated portfolio rebalancing with Code Interpreter — no advisor fees",
            "This AI reads your bank statements and finds tax deductions automatically",
            "Daily crypto/stock sentiment briefing generated by AI while you sleep"
        ],
        "demo_format": "Hook (specific $ saved/return) → Workflow diagram → Live demo (data→AI→action) → Cost breakdown → CTA",
        "tools": ["YNAB API", "Monarch Money API", "Plaid", "Yodlee", "ChatGPT Code Interpreter", "Claude Artifacts", "LlamaIndex", "SEC EDGAR API", "Twilio", "Alpha Vantage", "CoinGecko", "Reddit API", "Twitter API"],
        "prompts": [
            "You are a tax strategist. Given these transactions, identify potential deductions per IRS guidelines.",
            "Analyze this portfolio: compute Sharpe, max drawdown, correlation matrix, and suggest rebalancing.",
            "Categorize these transactions and flag subscriptions, duplicates, and negotiable bills."
        ]
    },
    "Facts & Trivia": {
        "strategy": "Focus on AI-powered fact discovery, verification, and generation pipelines. Wikipedia/Common Crawl RAG with LlamaIndex+Chroma, automated fact-checking (Claim→Search→LLM verification), trivia generation (LLM+Wikidata), hallucination detection (self-consistency+retrieval), historical tech stories (LLM+archive.org), myth-busting pipeline, automated daily fact newsletter.",
        "hooks": [
            "I built an AI that fact-checks viral tweets in REAL-TIME. Here's what it caught.",
            "This AI generates 1000 verified 'Did you know?' facts per minute. Mind blown.",
            "I asked an LLM to find the WEIRDEST tech fact in Wikipedia. You won't believe #3.",
            "Automated myth-busting: AI vs 50 common tech myths. The results shocked me.",
            "This bot reads 10,000 tech papers daily and gives me 3 mind-blowing facts every morning"
        ],
        "demo_format": "Hook (shocking fact card) → Verification pipeline diagram → Live claim→verification demo → Source citations shown → CTA",
        "tools": ["LlamaIndex", "Chroma", "FAISS", "Wikipedia API", "Wikidata", "Google Custom Search API", "Common Crawl", "archive.org", "PubMed API", "arXiv API", "Semantic Scholar API"],
        "prompts": [
            "Verify this claim against reliable sources. Return: verdict (true/false/uncertain), confidence (0-100), citations.",
            "Generate 10 surprising but verified tech facts from Wikipedia. Include article titles.",
            "Is this tech myth true or false? Provide evidence from primary sources."
        ]
    },
    "Life Hacks & Productivity": {
        "strategy": "Focus on AI-powered capture, organization, summarization, and automation. Voice→Notion/Obsidian (Whisper+LLM+API), Email triage (Gmail API+LLM), Meeting summarizer (Whisper.cpp+Llama), Calendar optimization (LLM analysis), Obsidian+RAG (local vault+embeddings), RSS→AI digest, Browser automation (Browser-use/Stagehand+LLM), Screenpipe/Rewind alternative (local screen+OCR+LLM).",
        "hooks": [
            "I haven't typed a note in 6 months. My voice AI organizes everything automatically.",
            "This AI reads my emails, drafts replies, and unsubscribes me from spam. Zero inbox.",
            "My meetings summarize THEMSELVES now. Whisper + Llama = magic.",
            "Obsidian + Local LLM = a second brain that ANSWERS questions from my notes.",
            "I automated my entire morning briefing. AI reads 50 newsletters, gives me 3 mins of gold."
        ],
        "demo_format": "Hook (clean result) → 3-tool stack logos → Live voice→organized demo → 'Local/private/free' → CTA",
        "tools": ["Whisper.cpp", "Whisper API", "Ollama", "Notion API", "Obsidian", "Gmail API", "Calendar API", "n8n", "Zapier", "Browser-use", "Stagehand", "Screenpipe", "Feedparser", "RSS", "Piper TTS"],
        "prompts": [
            "Convert this voice transcript into structured notes with: title, tags, action items, key insights, related topics.",
            "Triage this email: categorize (urgent/action/later/spam), draft reply if needed, extract tasks.",
            "Summarize this meeting transcript: decisions made, action items (owner+due), key discussion points, next steps."
        ]
    },
    "Agentic AI Facts": {
        "strategy": "Focus on practical agent patterns, frameworks, and debugging techniques. AutoGen (multi-agent chat), CrewAI (role-based), LangGraph (stateful graphs), OpenAI Assistants API. Core patterns: Planning→Execution→Reflection (Reflexion), Tool Use (function calling), Memory (short/long-term), Multi-agent debate. Reflexion: agent critiques own output→revises. Tool-use schemas: Pydantic for structured calling. Human-in-loop: approval gates, clarification, escalation. Debugging: LangSmith/LangFuse/Arize Phoenix. Agent swarms: hierarchical, peer-to-peer, sequential.",
        "hooks": [
            "I built an agent that writes, tests, and deploys code. It just shipped a feature.",
            "Reflexion: The technique that makes agents 40% smarter by CRITIQUING themselves.",
            "AutoGen vs CrewAI vs LangGraph — I tested all 3 for a week. Here's the winner.",
            "Why your agents fail: The 3 debugging tools you're not using (LangSmith, Phoenix, Fuse)",
            "Multi-agent debate: Two LLMs argue. The result? 95% accuracy on complex reasoning."
        ],
        "demo_format": "Hook (agent completing complex task) → Architecture diagram → Live demo (goal→agents→result) → Cost/time → CTA",
        "tools": ["AutoGen", "CrewAI", "LangGraph", "OpenAI Assistants API", "LangSmith", "LangFuse", "Arize Phoenix", "Pydantic", "Instructor", "Reflexion", "AutoGPT", "BabyAGI", "MetaGPT", "ChatDev"],
        "prompts": [
            "You are a planner. Break this goal into atomic, ordered steps with success criteria.",
            "You are a critic. Review this output for correctness, completeness, and style. Return specific improvements.",
            "You are a coder. Implement this spec. Write tests. Ensure type safety. Return diff."
        ]
    },
    "Coding & Development Hacks": {
        "strategy": "Focus on AI-assisted development workflows, prompts, and tools that ship code faster. AI code review (Claude/GPT-4 prompts), test generation (pytest+LLM), refactoring agents (legacy→LLM+AST), Cursor/Claude Code workflows (.cursorrules, Cline, Aider, Continue), PR automation (GitHub Actions+LLM), documentation (LLM reads code→docstrings/README/ADR/Mermaid), migration agents (Python 2→3, JS→TS, REST→GraphQL with codemods), performance profiling (py-spy+LLM), regex/SQL/config generation.",
        "hooks": [
            "This AI prompt reviews my code BETTER than senior engineers. Here's the exact prompt.",
            "I migrated 50k lines from JS to TS in 2 hours. AI codemods + AST = magic.",
            "Stop writing tests. This prompt generates 95% coverage including EDGE CASES.",
            "Cursor .cursorrules file that makes AI write production-ready code every time.",
            "Legacy spaghetti → clean architecture. My refactoring agent does it while I sleep."
        ],
        "demo_format": "Hook (tests generated/bug found) → Exact prompt/config shown → Live messy→clean demo → Diff view → CTA",
        "tools": ["Cursor", "Claude Code", "Cline", "Aider", "Continue.dev", "GitHub Copilot", "LibCST", "Rope", "py-spy", "pytest", "Hypothesis", "GitHub Actions", "Mermaid", "AST", "codemods", "ruff", "mypy", "bandit"],
        "prompts": [
            "You are a senior engineer. Review this code for: security vulnerabilities, performance issues, maintainability, type safety, test coverage. Return: severity, line, issue, fix.",
            "Generate comprehensive pytest tests for this function. Include: happy path, edge cases, error handling, property-based tests (Hypothesis), mocks. Target 95%+ coverage.",
            "Refactor this legacy code: add type hints, extract functions, remove duplication, add docstrings, modernize patterns. Preserve exact behavior. Return unified diff."
        ]
    }
}

def get_ai_hacks_for_category(category):
    """
    Returns structured AI hack data for a given category.
    Used by script generators, video pipelines, and teaser selection.
    
    to inject domain-specific AI workflows, hooks, and tool demos.
    
    Returns dict with keys:
        - strategy: Full AI hack strategy description
        - hooks: List of 3-5 viral hook templates
        - demo_format: Short-form demo structure
        - tools: List of specific tools to mention/show
        - prompts: Ready-to-use prompt templates
    """
    return _AI_HACKS_DATA.get(category, _AI_HACKS_DATA["AI & Tech Tools"])


def list_all_categories_with_ai_hacks():
    """Utility: returns all categories that have AI hack data defined."""
    return list(_AI_HACKS_DATA.keys())


def validate_ai_hacks_coverage():
    """Verify all daily categories have AI hack coverage."""
    daily_cats = [
        "AI & Tech Tools",
        "Tech Gadgets & Inventions", 
        "Finance & Tech Economy",
        "Facts & Trivia",
        "Life Hacks & Productivity",
        "Agentic AI Facts",
        "Coding & Development Hacks"
    ]
    covered = list(_AI_HACKS_DATA.keys())
    missing = [c for c in daily_cats if c not in covered]
    extra = [c for c in covered if c not in daily_cats]
    return {
        "total_daily_categories": len(daily_cats),
        "covered": len(covered),
        "coverage_pct": round(len(covered) / len(daily_cats) * 100, 1),
        "missing": missing,
        "extra": extra,
        "all_covered": len(missing) == 0
    }