"""
Shared LLM Fallback Module
Used by: shorts (gemini_script.py), longform (gemini_script_longform.py), educational images (content/generator.py)
"""

import json
import os
import re
import time
import requests
from typing import Optional, Dict, Any, List


# ─── JSON Normalization ────────────────────────────────────────────────────────

def normalize_llm_response(data: Dict[str, Any], required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Normalize LLM response to match expected schema.
    Handles common mismatches across different models.
    """
    if not isinstance(data, dict):
        return data
    
    # Normalize quiz options: strings -> objects
    if "quiz" in data and isinstance(data["quiz"], dict):
        quiz = data["quiz"]
        if "options" in quiz and isinstance(quiz["options"], list):
            normalized_options = []
            for i, opt in enumerate(quiz["options"]):
                if isinstance(opt, str):
                    text = opt
                    text = re.sub(r'^[A-D][\.\)]\s*', '', text).strip()
                    is_correct = (i == 0)
                    normalized_options.append({"text": text, "is_correct": is_correct})
                elif isinstance(opt, dict):
                    normalized_options.append({
                        "text": opt.get("text", opt.get("option", str(opt))),
                        "is_correct": opt.get("is_correct", opt.get("correct", False))
                    })
                else:
                    normalized_options.append({"text": str(opt), "is_correct": (i == 0)})
            quiz["options"] = normalized_options[:4]
    
    # Normalize infographic points
    if "infographic" in data and isinstance(data["infographic"], dict):
        infographic = data["infographic"]
        if "points" in infographic and isinstance(infographic["points"], list):
            normalized_points = []
            for point in infographic["points"]:
                if isinstance(point, dict):
                    label = point.get("label", point.get("title", point.get("key", "Point")))
                    value = point.get("value", point.get("description", point.get("detail", "")))
                    normalized_points.append({"label": label, "value": value, "icon": point.get("icon")})
                elif isinstance(point, str):
                    normalized_points.append({"label": "Point", "value": point})
            infographic["points"] = normalized_points[:6]
    
    # Normalize flowchart steps
    if "flowchart" in data and isinstance(data["flowchart"], dict):
        fc = data["flowchart"]
        if "steps" in fc and isinstance(fc["steps"], list):
            normalized_steps = []
            for step in fc["steps"]:
                if isinstance(step, dict):
                    label = step.get("label", step.get("title", step.get("step", "Step")))
                    desc = step.get("description", step.get("detail", ""))
                    normalized_steps.append({"label": label, "description": desc})
                elif isinstance(step, str):
                    normalized_steps.append({"label": step, "description": ""})
            fc["steps"] = normalized_steps[:7]
# Normalize comparison table - handle various formats
    # First, ensure comparison is a dict
    if "comparison" in data:
        if isinstance(data["comparison"], list):
            # If comparison is a list, convert to dict with rows
            data["comparison"] = {"rows": data["comparison"], "title": "Comparison"}
        if isinstance(data["comparison"], dict):
            comp = data["comparison"]
            if "rows" in comp and isinstance(comp["rows"], list):
                normalized_rows = []
                for row in comp["rows"]:
                    if isinstance(row, dict):
                        feature = row.get("feature", row.get("criteria", row.get("key", "Feature")))
                        option_a = row.get("option_a", row.get("a", row.get("left", row.get("traditional", row.get("without_rag", row.get("before", ""))))))
                        option_b = row.get("option_b", row.get("b", row.get("right", row.get("modern", row.get("with_rag", row.get("after", ""))))))
                        normalized_rows.append({"feature": feature, "option_a": option_a, "option_b": option_b})
                    elif isinstance(row, (list, tuple)):
                        if len(row) >= 3:
                            # ['Feature', 'OptionA', 'OptionB'] or ['feature', 'optA', 'optB', 'optC']
                            normalized_rows.append({"feature": str(row[0]), "option_a": str(row[1]), "option_b": str(row[2])})
                        elif len(row) == 2:
                            # ['feature', 'option'] - treat as feature with single option
                            normalized_rows.append({"feature": str(row[0]), "option_a": str(row[1]), "option_b": ""})
                        else:
                            continue
                    else:
                        continue
                # Skip header row if it looks like one (first row has generic names)
                if normalized_rows and len(normalized_rows) > 1:
                    first = normalized_rows[0]
                    if first.get("feature", "").lower() in ("feature", "criteria", "key", "parameter", "property", "name", "type"):
                        normalized_rows = normalized_rows[1:]
                comp["rows"] = normalized_rows[:6]
            # Ensure headers have defaults
            if "header_a" not in comp:
                comp["header_a"] = "Traditional"
            if "header_b" not in comp:
                comp["header_b"] = "Modern"
            if "title" not in comp:
                comp["title"] = "Comparison"
    
    # Normalize architecture components
    if "architecture" in data and isinstance(data["architecture"], dict):
        arch = data["architecture"]
        if "components" in arch and isinstance(arch["components"], list):
            normalized_comps = []
            for comp in arch["components"]:
                if isinstance(comp, dict):
                    name = comp.get("name", comp.get("title", comp.get("component", "Component")))
                    desc = comp.get("description", comp.get("detail", ""))
                    icon = comp.get("icon", "⚙️")
                    conns = comp.get("connections", comp.get("connects_to", []))
                    normalized_comps.append({"name": name, "description": desc, "icon": icon, "connections": conns})
                elif isinstance(comp, str):
                    normalized_comps.append({"name": comp, "description": "", "icon": "⚙️", "connections": []})
            arch["components"] = normalized_comps[:8]
    
    # Normalize code snippet
    if "code" in data and isinstance(data["code"], dict):
        code = data["code"]
        if "content" in code and not isinstance(code["content"], str):
            code["content"] = str(code["content"])
        if "language" not in code:
            code["language"] = "python"
    
    # Ensure required fields have defaults
    defaults = {
        "takeaway": "Key concept understood.",
        "cta": "Save this for later!",
        "hook": "Learn this essential concept.",
        "visual_strategy": ["infographic", "flowchart", "quiz"],
    }
    for key, default in defaults.items():
        if key not in data or not data[key]:
            data[key] = default
    
    # Ensure required_fields if specified
    if required_fields:
        for field in required_fields:
            if field not in data or not data[field]:
                data[field] = defaults.get(field, "")
    
    return data


def clean_and_parse_json(content: str) -> Dict[str, Any]:
    """Extract and parse JSON from LLM response (handles markdown code blocks)."""
    raw = content.strip()
    if "```json" in raw:
        raw = raw[raw.find("```json")+7:raw.rfind("```")]
    elif "```" in raw:
        raw = raw[raw.find("```")+3:raw.rfind("```")]
    return json.loads(raw.strip())


def safe_extract_choices(response_or_json, provider_name: str) -> Optional[str]:
    """Safely extract content from OpenAI-compatible response."""
    try:
        data = response_or_json if isinstance(response_or_json, dict) else response_or_json.json()
        choices = data.get("choices")
        if choices and len(choices) > 0:
            msg = choices[0].get("message")
            if msg and "content" in msg and msg["content"]:
                return msg["content"].strip()
        print(f"⚠️ [{provider_name}] Response structure unexpected")
    except Exception as e:
        print(f"⚠️ [{provider_name}] Failed to parse response: {e}")
    return None


def safe_extract_anthropic(response) -> Optional[str]:
    """Safely extract content from Anthropic response."""
    try:
        data = response.json()
        content_list = data.get("content")
        if content_list and len(content_list) > 0:
            text = content_list[0].get("text")
            if text:
                return text.strip()
        print(f"⚠️ [Anthropic] Response structure unexpected")
    except Exception as e:
        print(f"⚠️ [Anthropic] Failed to parse response: {e}")
    return None


# ─── Provider Callers ──────────────────────────────────────────────────────────

import time

def _retry_with_backoff(func, max_retries=3, base_delay=2.0):
    """Retry a function with exponential backoff for transient errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"   ⏳ Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
            raise
    return None


def call_groq(user_prompt: str) -> Optional[Dict[str, Any]]:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None
    
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    
    for model_name in groq_models:
        print(f"🔮 Falling back to Groq ({model_name})...")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            }
            result = _retry_with_backoff(
                lambda: requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            )
            if result and result.status_code == 200:
                content = safe_extract_choices(result.json(), f"Groq {model_name}")
                if content:
                    return clean_and_parse_json(content)
            elif result:
                print(f"⚠️ Groq ({model_name}) failed: {result.status_code}")
        except Exception as e:
            print(f"⚠️ Groq ({model_name}) exception: {e}")
    return None


def call_cloudflare(user_prompt: str) -> Optional[Dict[str, Any]]:
    cf_token = os.getenv("CF_API_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if not cf_token or not cf_account:
        return None
    
    cf_models = [
        "@cf/meta/llama-3.3-70b-instruct",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/zai-org/glm-4.7-flash",
        "@cf/openai/gpt-oss-120b",
        "@cf/nvidia/nemotron-3-120b-a12b",
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        "@cf/qwen/qwq-32b",
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    ]
    
    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json"
    }
    gpt_oss_models = {"@cf/openai/gpt-oss-120b", "@cf/openai/gpt-oss-20b"}
    
    for model_name in cf_models:
        print(f"🔮 Falling back to Cloudflare ({model_name})...")
        try:
            payload = {
                "messages": [{"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
                "max_tokens": 4096
            }
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/{model_name}",
                json=payload, headers=headers, timeout=60
            )
            if r.status_code == 200:
                result = r.json().get("result", {})
                if model_name in gpt_oss_models:
                    content = safe_extract_choices(result, f"Cloudflare {model_name}")
                else:
                    content = result.get("response", "").strip()
                if content:
                    return clean_and_parse_json(content)
        except Exception as e:
            print(f"⚠️ Cloudflare ({model_name}) exception: {e}")
    return None


def call_opencode(user_prompt: str) -> Optional[Dict[str, Any]]:
    opencode_key = os.getenv("OPENCODE_API_KEY")
    if not opencode_key:
        return None
    
    model_name = "nemotron-3-ultra-free"
    print(f"🔮 Falling back to OpenCode Zen ({model_name})...")
    try:
        headers = {
            "Authorization": f"Bearer {opencode_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }
        r = requests.post("https://opencode.ai/zen/v1/chat/completions", json=payload, headers=headers, timeout=60)
        if r.status_code == 200:
            data = r.json()
            # Check for error response
            if "error" in data:
                print(f"⚠️ [OpenCode Zen] API error: {data['error']}")
                return None
            # Try multiple response formats for compatibility
            content = None
            # Format 1: OpenAI standard
            choices = data.get("choices")
            if choices and len(choices) > 0:
                msg = choices[0].get("message")
                if msg and "content" in msg and msg["content"]:
                    content = msg["content"].strip()
            # Format 2: Direct content field
            if not content and "content" in data and data["content"]:
                content = data["content"].strip()
            # Format 3: Response object with text
            if not content and "response" in data and isinstance(data["response"], dict):
                content = data["response"].get("text", "").strip()
            # Format 4: Raw text in data
            if not content and "text" in data and data["text"]:
                content = data["text"].strip()
            
            if content:
                return clean_and_parse_json(content)
            else:
                print(f"⚠️ [OpenCode Zen] Response structure unexpected: {list(data.keys())}")
    except Exception as e:
        print(f"⚠️ OpenCode Zen exception: {e}")
    return None


def call_cerebras(user_prompt: str) -> Optional[Dict[str, Any]]:
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if not cerebras_key:
        return None
    
    cerebras_models = ["gpt-oss-120b", "zai-glm-4.7", "gemma-4-31b"]
    headers = {"Authorization": f"Bearer {cerebras_key}", "Content-Type": "application/json"}
    
    for model_name in cerebras_models:
        print(f"🔮 Falling back to Cerebras ({model_name})...")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            }
            r = requests.post("https://api.cerebras.ai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = safe_extract_choices(r.json(), "Cerebras")
                if content:
                    return clean_and_parse_json(content)
        except Exception as e:
            print(f"⚠️ Cerebras ({model_name}) exception: {e}")
    return None


def call_nvidia(user_prompt: str) -> Optional[Dict[str, Any]]:
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_key:
        return None
    
    nvidia_models = ["nvidia/llama-3.1-nemotron-70b-instruct", "meta/llama3-70b-instruct"]
    headers = {"Authorization": f"Bearer {nvidia_key}", "Content-Type": "application/json"}
    
    for model_name in nvidia_models:
        print(f"🔮 Falling back to NVIDIA NIM ({model_name})...")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.3,
            }
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = safe_extract_choices(r.json(), "NVIDIA NIM")
                if content:
                    return clean_and_parse_json(content)
        except Exception as e:
            print(f"⚠️ NVIDIA NIM ({model_name}) exception: {e}")
    return None


def call_mistral(user_prompt: str) -> Optional[Dict[str, Any]]:
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        return None
    
    mistral_models = ["mistral-large-latest", "pixtral-large-latest", "codestral-latest"]
    headers = {"Authorization": f"Bearer {mistral_key}", "Content-Type": "application/json"}
    
    for model_name in mistral_models:
        print(f"🔮 Falling back to Mistral ({model_name})...")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            }
            r = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = safe_extract_choices(r.json(), "Mistral")
                if content:
                    return clean_and_parse_json(content)
        except Exception as e:
            print(f"⚠️ Mistral ({model_name}) exception: {e}")
    return None


def call_github_models(user_prompt: str) -> Optional[Dict[str, Any]]:
    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not github_token:
        return None
    
    github_models = ["gpt-4o-mini", "meta-llama-3.1-405b-instruct"]
    headers = {"Authorization": f"Bearer {github_token}", "Content-Type": "application/json"}
    
    for model_name in github_models:
        print(f"🔮 Falling back to GitHub Models ({model_name})...")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            }
            r = requests.post("https://models.inference.ai.azure.com/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = safe_extract_choices(r.json(), "GitHub Models")
                if content:
                    return clean_and_parse_json(content)
        except Exception as e:
            print(f"⚠️ GitHub Models ({model_name}) exception: {e}")
    return None


def call_openai(user_prompt: str) -> Optional[Dict[str, Any]]:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None
    
    print("🔮 Falling back to OpenAI (gpt-4o-mini)...")
    try:
        headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            content = safe_extract_choices(r.json(), "OpenAI")
            if content:
                return clean_and_parse_json(content)
    except Exception as e:
        print(f"⚠️ OpenAI exception: {e}")
    return None


def call_anthropic(user_prompt: str) -> Optional[Dict[str, Any]]:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return None
    
    print("🔮 Falling back to Anthropic (claude-3-5-haiku-20241022)...")
    try:
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": user_prompt}]
        }
        r = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            content = safe_extract_anthropic(r)
            if content:
                return clean_and_parse_json(content)
    except Exception as e:
        print(f"⚠️ Anthropic exception: {e}")
    return None


def call_deepseek(user_prompt: str) -> Optional[Dict[str, Any]]:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        return None
    
    print("🔮 Falling back to DeepSeek (deepseek-chat)...")
    try:
        headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }
        r = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            content = safe_extract_choices(r.json(), "DeepSeek")
            if content:
                return clean_and_parse_json(content)
    except Exception as e:
        print(f"⚠️ DeepSeek exception: {e}")
    return None


def call_openrouter(user_prompt: str) -> Optional[Dict[str, Any]]:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return None
    
    openrouter_models = ["openrouter/free", "meta-llama/llama-3.3-70b-instruct"]
    headers = {"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"}
    
    for model_name in openrouter_models:
        print(f"🔮 Falling back to OpenRouter ({model_name})...")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.3,
            }
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                content = safe_extract_choices(r.json(), "OpenRouter")
                if content:
                    return clean_and_parse_json(content)
        except Exception as e:
            print(f"⚠️ OpenRouter ({model_name}) exception: {e}")
    return None


def generate_local_fallback_script(user_prompt: str, required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Generate a minimal valid script structure when all LLM providers fail."""
    import re
    
    # Extract topic from prompt
    topic_match = re.search(r'(?:STORY|TOPIC|HEADLINE)[:\s]+([^\n]+)', user_prompt, re.IGNORECASE)
    headline = topic_match.group(1).strip() if topic_match else "AI Tech Breakthrough"
    
    url_match = re.search(r'(?:SOURCE|URL)[:\s]+(https?://\S+)', user_prompt, re.IGNORECASE)
    url = url_match.group(1).strip() if url_match else ""
    
    # Generate a basic but valid script
    script = f"Did you know {headline}? This changes everything for developers. " \
             f"The tool automates complex workflows that used to take hours. " \
             f"Here's why it matters: it runs locally, no cloud costs, full privacy. " \
             f"But here's the twist — it's completely open source. " \
             f"What does this mean for your production apps? You can customize everything. " \
             f"Save this for your next project. Link in bio."
    
    base = {
        "script": script,
        "original_news_headline": headline,
        "original_news_url": url,
        "title": headline[:80],
        "ai_description": f"Discover {headline.lower()} — a game-changing open-source alternative.",
        "hashtags": ["#AI", "#OpenSource", "#DevTools", "#TechNews"],
        "retention_map": {
            "open_loops": [{"text": "changes everything for developers", "planted_at_word": 5, "resolved_at_word": 25}],
            "pattern_interrupts": [{"type": "contradiction", "text": "But here's the twist", "at_word": 30}],
            "rhetorical_questions": [{"text": "What does this mean for your production apps?", "at_word": 40}],
            "direct_address_count": 3,
            "curiosity_gap_ratio": 0.6,
            "hook_word_count": 6,
            "payoff_zone_start_word": 50,
            "retention_risk_zones": []
        },
        "script_format": "Result-First",
        "comment_trigger_keyword": "TOOL",
        "incentive_cta_type": "save",
        "digital_asset_offer": "GitHub repo link",
        "editorial_perspective": "Cost Optimizer",
        "editorial_angle": "Focus on free alternatives to expensive proprietary tools",
        "content_fingerprint": f"local_fallback_{hash(headline) % 10000}"
    }
    
    # Ensure required fields are present
    if required_fields:
        for field in required_fields:
            if field not in base:
                base[field] = ""
    
    return base


# ─── Main Fallback Chain ───────────────────────────────────────────────────────

FALLBACK_CHAIN = [
    ("OpenCode Zen", call_opencode),
    ("Cerebras", call_cerebras),
    ("NVIDIA NIM", call_nvidia),
    ("Mistral", call_mistral),
    ("Groq", call_groq),
    ("Cloudflare", call_cloudflare),
    ("OpenRouter", call_openrouter),
    # ("GitHub Models", call_github_models),  # DNS fails in GitHub Actions
    ("OpenAI", call_openai),
    ("Anthropic", call_anthropic),
    ("DeepSeek", call_deepseek),
]


def call_fallback_chain(user_prompt: str, normalize: bool = True, required_fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Try all fallback providers in order until one succeeds.
    
    Args:
        user_prompt: The prompt to send to LLMs
        normalize: Whether to apply JSON normalization
        required_fields: List of required field names to ensure in response
    
    Returns:
        Normalized dict or None if all providers fail
    """
    for name, func in FALLBACK_CHAIN:
        for attempt in range(2):  # Retry once on transient failures
            try:
                result = func(user_prompt)
                if result:
                    print(f"✅ {name} succeeded")
                    if normalize:
                        return normalize_llm_response(result, required_fields)
                    return result
            except Exception as e:
                if attempt == 0:
                    print(f"⚠️ {name} fallback failed (attempt 1/2): {e}. Retrying...")
                    time.sleep(2)
                else:
                    print(f"⚠️ {name} fallback failed after retries: {e}")
    
    # Last resort: Generate a minimal valid script structure locally
    print("🔧 All providers failed. Generating minimal local fallback script...")
    return generate_local_fallback_script(user_prompt, required_fields)