"""
Shared LLM Fallback Module
Used by: shorts (gemini_script.py), longform (gemini_script_longform.py), educational images (content/generator.py)
"""

import json
import os
import re
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
    """Extract and parse JSON from LLM response (handles markdown code blocks and conversational wrappers)."""
    raw = content.strip()
    if "```json" in raw:
        raw = raw[raw.find("```json")+7:raw.rfind("```")]
    elif "```" in raw:
        raw = raw[raw.find("```")+3:raw.rfind("```")]
    raw = raw.strip()
    if not (raw.startswith("{") or raw.startswith("[")):
        start_brace = raw.find("{")
        start_bracket = raw.find("[")
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            end_brace = raw.rfind("}")
            if end_brace != -1:
                raw = raw[start_brace:end_brace+1]
        elif start_bracket != -1:
            end_bracket = raw.rfind("]")
            if end_bracket != -1:
                raw = raw[start_bracket:end_bracket+1]
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
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
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
                print(f"⚠️ Groq ({model_name}) returned HTTP {result.status_code}: {result.text[:150]}")
        except Exception as e:
            print(f"⚠️ Groq ({model_name}) exception: {e}")
    return None


def call_cloudflare(user_prompt: str) -> Optional[Dict[str, Any]]:
    cf_token = os.getenv("CF_API_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN")
    cf_account = os.getenv("CF_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if not cf_token or not cf_account:
        return None
    
    cf_models = [
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/meta/llama-3.1-8b-instruct",
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "@cf/qwen/qwen2.5-72b-instruct",
        "@cf/mistral/mistral-7b-instruct-v0.2",
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
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Always output valid JSON."},
                    {"role": "user", "content": user_prompt}
                ],
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
                    response_val = result.get("response", "")
                    if isinstance(response_val, dict):
                        # Some CF models return {"response": {"choices": [...]}} or raw JSON object
                        content = safe_extract_choices(response_val, f"Cloudflare {model_name}")
                        if not content:
                            content = json.dumps(response_val)
                    elif isinstance(response_val, str):
                        content = response_val.strip()
                    else:
                        content = str(response_val).strip()
                if content:
                    return clean_and_parse_json(content)
            else:
                print(f"⚠️ Cloudflare ({model_name}) returned HTTP {r.status_code}: {r.text[:150]}")
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
        r = requests.post("https://opencode.ai/zen/v1/chat/completions", json=payload, headers=headers, timeout=90)
        if r.status_code == 200:
            content = safe_extract_choices(r.json(), "OpenCode Zen")
            if content:
                return clean_and_parse_json(content)
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
    
    github_models = ["openai/gpt-4o-mini", "meta-llama-3.1-70b-instruct"]
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
            r = requests.post("https://models.github.ai/inference/chat/completions", json=payload, headers=headers, timeout=30)
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
        else:
            print(f"⚠️ DeepSeek returned HTTP {r.status_code}: {r.text[:150]}")
    except Exception as e:
        print(f"⚠️ DeepSeek exception: {e}")
    return None


# ─── OpenRouter Model Priorities ───────────────────────────────────────────────
# Priority 1: nvidia/nemotron-3-ultra-550b-a55b:free (Main content generation / reasoning)
# Priority 2: poolside/laguna-s-2.1:free (Coding + technical topics)
# Priority 3: nvidia/nemotron-3.5-lightning:free (Fast high-volume fallback)
# Priority 4: inclusionai/ling-3.0-flash-fin:free (Finance/business topics)

OPENROUTER_PRIORITY_MODELS = {
    "main_reasoning": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "coding_technical": "poolside/laguna-s-2.1:free",
    "fast_fallback": "nvidia/nemotron-3.5-lightning:free",
    "finance_business": "inclusionai/ling-3.0-flash-fin:free",
}

OPENROUTER_ADDITIONAL_FREE_MODELS = [
    "qwen/qwen3.8-27b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "poolside/laguna-s-2.1:free",
    "z-ai/glm-5.2:free",
    "meta-llama/llama-3.1-8b-instruct:free",
]


def get_openrouter_models_by_priority(topic_category: Optional[str] = None, context_text: str = "") -> List[str]:
    """Return OpenRouter models ordered according to user-specified priority & topic specialization."""
    text_lower = f"{topic_category or ''} {context_text}".lower()
    
    # Coding / Technical topic detection -> prioritize poolside/laguna-s-2.1
    if any(k in text_lower for k in ["code", "coding", "developer", "ide", "programming", "python", "typescript", "rust", "ast", "cli", "git", "api"]):
        prioritized = [
            OPENROUTER_PRIORITY_MODELS["coding_technical"],   # Priority 2 specialized
            OPENROUTER_PRIORITY_MODELS["main_reasoning"],    # Priority 1 main
            OPENROUTER_PRIORITY_MODELS["fast_fallback"],     # Priority 3 fallback
            OPENROUTER_PRIORITY_MODELS["finance_business"],  # Priority 4
        ]
    # Finance / Business topic detection -> prioritize inclusionai/ling-3.0-flash-fin
    elif any(k in text_lower for k in ["finance", "funding", "valuation", "venture", "market", "revenue", "investor", "acquisition", "stock", "business", "industry_trend", "industry", "economy", "startup"]):
        prioritized = [
            OPENROUTER_PRIORITY_MODELS["finance_business"],  # Priority 4 specialized
            OPENROUTER_PRIORITY_MODELS["main_reasoning"],    # Priority 1 main
            OPENROUTER_PRIORITY_MODELS["coding_technical"],  # Priority 2
            OPENROUTER_PRIORITY_MODELS["fast_fallback"],     # Priority 3 fallback
        ]
    else:
        # Default order: Main reasoning (P1) -> Coding (P2) -> Fast fallback (P3) -> Finance (P4)
        prioritized = [
            OPENROUTER_PRIORITY_MODELS["main_reasoning"],        # Priority 1
            OPENROUTER_PRIORITY_MODELS["coding_technical"],      # Priority 2
            OPENROUTER_PRIORITY_MODELS["fast_fallback"],         # Priority 3
            OPENROUTER_PRIORITY_MODELS["finance_business"],      # Priority 4
        ]
    
    # Append fallback free models without duplicates to survive rate-limits
    for m in OPENROUTER_ADDITIONAL_FREE_MODELS:
        if m not in prioritized:
            prioritized.append(m)
    return prioritized


def call_openrouter(user_prompt: str, topic_category: Optional[str] = None, context_text: str = "", timeout: int = 40) -> Optional[Dict[str, Any]]:
    """Call OpenRouter with prioritized models: Nemotron 3 Ultra, Laguna, Nemotron Lightning, Ling Flash.
    
    Args:
        timeout: HTTP timeout in seconds. Default 40s for shorts; pass 90s for longform prompts
                 where free-tier 550B models need more generation time.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return None
    
    models_to_try = get_openrouter_models_by_priority(topic_category=topic_category, context_text=context_text)
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vjaab/YtDidYouKnowByVJ",
        "X-Title": "YtDidYouKnowByVJ Video & Carousel Pipeline",
    }
    
    for model_name in models_to_try:
        print(f"🔮 Calling OpenRouter priority model ({model_name})...")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.4,
            }
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                content = safe_extract_choices(r.json(), f"OpenRouter:{model_name}")
                if content:
                    parsed = clean_and_parse_json(content)
                    if parsed:
                        print(f"✅ OpenRouter ({model_name}) succeeded")
                        return parsed
            else:
                print(f"⚠️ OpenRouter ({model_name}) returned HTTP {r.status_code}: {r.text[:150]}")
                if r.status_code == 429 and ("free-models-per-day" in r.text or "per-day" in r.text):
                    print("⏳ OpenRouter account daily limit reached. Fast-falling back to next provider.")
                    break
        except Exception as e:
            print(f"⚠️ OpenRouter ({model_name}) exception: {e}")
    return None


# ─── Main Fallback Chain ───────────────────────────────────────────────────────
# Priority 1: OpenRouter prioritized free models
# Priority 2: Groq high-speed free tier
# Priority 3: Cloudflare Workers AI
# Priority 4+: Specialized/Paid providers
FALLBACK_CHAIN = [
    ("OpenRouter", call_openrouter),
    ("Groq", call_groq),
    ("Cloudflare", call_cloudflare),
    ("OpenCode Zen", call_opencode),
    ("Cerebras", call_cerebras),
    ("NVIDIA NIM", call_nvidia),
    ("Mistral", call_mistral),
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
        try:
            result = func(user_prompt)
            if result:
                print(f"✅ {name} succeeded")
                if normalize:
                    return normalize_llm_response(result, required_fields)
                return result
        except Exception as e:
            print(f"⚠️ {name} fallback failed: {e}")
    
    print("🚨 All fallback providers failed or not configured")
    return None