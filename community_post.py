import os
import random
from datetime import datetime
from config import YOUTUBE_CLIENT_SECRET_FILE

SCOPES = [
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

def get_authenticated_service():
    if not os.path.exists(YOUTUBE_CLIENT_SECRET_FILE):
        print("YouTube client secret file not found.")
        return None
        
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    import google_auth_oauthlib.flow
    
    creds = None
    token_path = "token.json"
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                YOUTUBE_CLIENT_SECRET_FILE, SCOPES
            )
            creds = flow.run_local_server(port=8080, prompt='consent')
            
        with open(token_path, "w") as token:
            token.write(creds.to_json())
            
    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", credentials=creds)
        return youtube
    except Exception as e:
        print(f"YouTube auth failed: {e}")
        return None

COMMUNITY_POST_TEMPLATES = [
    # Template 0: Question/Poll style
    {
        "type": "text",
        "text": """What's the ONE AI tool you can't live without this week?

I've been testing dozens, but keep coming back to [TOOL] for [SPECIFIC USE CASE].

Drop yours below 👇""",
        "tags": ["AITools", "Productivity", "TechStack"]
    },
    # Template 1: Behind the scenes / Process
    {
        "type": "text",
        "text": """Behind the scenes of this week's deep dive:

The topic: [TOPIC]
The rabbit hole: [INTERESTING DISCOVERY]
The verdict: [TL;DR]

Full breakdown drops [DAY] — hit the bell so you don't miss it.

What would YOU have focused on?""",
        "tags": ["BehindTheScenes", "AIResearch", "DeepDive"]
    },
    # Template 2: Controversial take / Hot take
    {
        "type": "text",
        "text": """Unpopular opinion: [CONTROVERSIAL TAKE]

Everyone's hyping [TRENDING THING], but here's what nobody's talking about:
[SPECIFIC COUNTERPOINT]

Am I wrong? Let me know in the comments 👇""",
        "tags": ["HotTake", "AIDebate", "TechOpinion"]
    },
    # Template 3: Resource share
    {
        "type": "text",
        "text": """Free resource drop 🎁

I compiled [SPECIFIC RESOURCE: e.g., "50 prompt templates for code review" / "Local LLM benchmark spreadsheet" / "API cost comparison sheet"]

Grab it here: [TELEGRAM LINK]

What else would be useful to have in a pack like this?""",
        "tags": ["FreeResources", "AITools", "DeveloperProductivity"]
    },
    # Template 4: Weekly recap / Preview
    {
        "type": "text",
        "text": """This week in AI (quick hits):

1️⃣ [MAJOR NEWS 1]
2️⃣ [MAJOR NEWS 2] 
3️⃣ [MAJOR NEWS 3]

My take: [ONE SENTENCE SYNTHESIS]

Next week's deep dive: [TEASE TOPIC]

What did I miss? Drop it below 👇""",
        "tags": ["WeeklyRecap", "AINews", "TechTrends"]
    },
    # Template 5: Question to audience
    {
        "type": "text",
        "text": """Quick question for the engineers here:

[SPECIFIC TECHNICAL QUESTION]

Context: [WHY THIS MATTERS]

Options:
A) [OPTION A]
B) [OPTION B]
C) [OPTION C]

Vote in replies 👇""",
        "tags": ["EngineeringQuestion", "TechDiscussion", "CommunityPoll"]
    },
    # Template 6: Milestone / Channel update
    {
        "type": "text",
        "text": """Milestone update 📈

[CHANNEL MILESTONE: e.g., "Hit 10K subs" / "50 videos published" / "1M total views"]

What started as [ORIGIN STORY] has become [CURRENT STATE].

Biggest lesson: [KEY INSIGHT]

Thank you for being here. What should the next 50 videos cover?""",
        "tags": ["Milestone", "ChannelUpdate", "ThankYou"]
    },
]

# Fallback content for when we need variety
FALLBACK_TOPICS = [
    ("local LLMs", "running models offline without API costs"),
    ("prompt engineering", "getting better results with structured prompts"),
    ("AI coding assistants", "which one actually writes production-ready code"),
    ("vector databases", "when you actually need one vs when you don't"),
    ("RAG implementation", "common mistakes that kill retrieval quality"),
    ("model quantization", "4-bit vs 8-bit: real-world quality tradeoffs"),
    ("AI agents", "where they're useful vs where they're overkill"),
    ("fine-tuning vs RAG", "decision framework for your use case"),
    ("open-source vs API", "cost/privacy/quality decision matrix"),
    ("eval frameworks", "how to measure if your AI actually works"),
]

def _pick_template_and_fill(script_data=None):
    """Pick a template and fill with contextual content."""
    template = random.choice(COMMUNITY_POST_TEMPLATES)
    
    # Context-aware filling
    topic = "AI tooling"
    tool = "Cursor"
    use_case = "code review automation"
    discovery = "the benchmark numbers don't match marketing claims"
    takeaway = "local models are closing the gap faster than expected"
    day = "Thursday"
    telegram = "https://t.me/technewsbyvj"
    milestone = "50 videos published"
    origin = "a weekend experiment"
    current_state = "a daily AI research pipeline"
    lesson = "consistency compounds more than viral hits"
    question = "Would you rather: perfect RAG or fine-tuned model?"
    why_matters = "affects your entire architecture"
    opt_a = "Perfect RAG pipeline"
    opt_b = "Fine-tuned 7B model"
    opt_c = "Hybrid approach"
    news1 = "OpenAI dropped GPT-5 preview"
    news2 = "Meta released Llama 4 weights"
    news3 = "Anthropic launched Claude Code"
    synthesis = "The gap between open and closed is evaporating"
    tease = "Why your RAG is hallucinating (and the fix)"
    
    if script_data:
        topic = script_data.get("original_news_headline", topic)[:60]
        tool = (script_data.get("companies_mentioned", [{}])[0].get("name") if script_data.get("companies_mentioned") else tool)
    
    text = template["text"]
    replacements = {
        "[TOPIC]": topic,
        "[TOOL]": tool,
        "[SPECIFIC USE CASE]": use_case,
        "[INTERESTING DISCOVERY]": discovery,
        "[TL;DR]": takeaway,
        "[DAY]": day,
        "[CONTROVERSIAL TAKE]": "Most AI benchmarks are misleading",
        "[TRENDING THING]": "synthetic data",
        "[SPECIFIC COUNTERPOINT]": "Quality > quantity, but nobody measures quality right",
        "[SPECIFIC RESOURCE]": "Local LLM benchmark comparison sheet",
        "[TELEGRAM LINK]": telegram,
        "[MAJOR NEWS 1]": news1,
        "[MAJOR NEWS 2]": news2,
        "[MAJOR NEWS 3]": news3,
        "[ONE SENTENCE SYNTHESIS]": synthesis,
        "[TEASE TOPIC]": tease,
        "[CHANNEL MILESTONE]": milestone,
        "[ORIGIN STORY]": origin,
        "[CURRENT STATE]": current_state,
        "[KEY INSIGHT]": lesson,
        "[SPECIFIC TECHNICAL QUESTION]": question,
        "[WHY THIS MATTERS]": why_matters,
        "[OPTION A]": opt_a,
        "[OPTION B]": opt_b,
        "[OPTION C]": opt_c,
    }
    
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    
    return text, template["tags"]

def post_community_post(youtube, text, post_type="text"):
    """Post a community post to the channel."""
    try:
        request = youtube.posts().insert(
            part="snippet",
            body={
                "snippet": {
                    "post": {
                        "text": {
                            "content": text
                        }
                    }
                }
            }
        )
        response = request.execute()
        post_id = response.get("id")
        print(f"✅ Community post created: {post_id}")
        return True, post_id
    except Exception as e:
        print(f"❌ Community post failed: {e}")
        return False, str(e)

def create_community_post(script_data=None, dry_run=False):
    """Create and post a community post."""
    if dry_run:
        text, tags = _pick_template_and_fill(script_data)
        print("🧪 [DRY RUN] Community post preview:")
        print(f"Text: {text[:200]}...")
        print(f"Tags: {tags}")
        return True, "MOCK_POST_ID"
    
    youtube = get_authenticated_service()
    if not youtube:
        return False, "Failed to authenticate"
    
    text, tags = _pick_template_and_fill(script_data)
    
    success, result = post_community_post(youtube, text)
    return success, result

def schedule_community_posts():
    """Returns schedule configuration for twice-weekly community posts."""
    # Returns days and times for posting
    return {
        "days": ["Monday", "Thursday"],  # Twice a week
        "times": ["10:00", "16:00"],      # Morning and afternoon IST
        "timezone": "Asia/Kolkata"
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without posting")
    parser.add_argument("--now", action="store_true", help="Post immediately")
    args = parser.parse_args()
    
    if args.now or args.dry_run:
        success, result = create_community_post(dry_run=args.dry_run)
        if not success:
            print(f"Failed: {result}")
            exit(1)
        else:
            print(f"Success: {result}")
    else:
        print("Usage: python community_post.py --now  # Post immediately")
        print("       python community_post.py --dry-run  # Preview only")