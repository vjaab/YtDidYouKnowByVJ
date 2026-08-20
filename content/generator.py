import json
import os
import requests
from typing import Optional, List
from google import genai
from google.genai import types

from content.schemas import (
    EducationalContent, TopicCategory, DifficultyLevel, VisualType,
    Quiz, QuizOption, Flowchart, FlowchartStep, Infographic, InfographicPoint,
    CodeSnippet, ArchitectureDiagram, ArchitectureComponent, ComparisonTable, ComparisonRow
)


CATEGORY_KEYWORDS = {
    TopicCategory.PYTHON: ["python", "decorator", "async", "generator", "context manager", "dataclass", "typing"],
    TopicCategory.JAVA: ["java", "spring", "stream", "optional", "completablefuture", "jpa", "hibernate"],
    TopicCategory.AWS: ["aws", "lambda", "dynamodb", "s3", "api gateway", "cloudformation", "ecs", "rds"],
    TopicCategory.AI: ["ai", "llm", "transformer", "embedding", "fine-tuning", "prompt engineering", "rag"],
    TopicCategory.MACHINE_LEARNING: ["machine learning", "training", "inference", "model", "feature", "pipeline"],
    TopicCategory.RAG: ["rag", "retrieval", "embedding", "vector database", "chunking", "reranking"],
    TopicCategory.KUBERNETES: ["kubernetes", "pod", "service", "deployment", "ingress", "configmap", "helm"],
    TopicCategory.DOCKER: ["docker", "container", "image", "dockerfile", "compose", "registry"],
    TopicCategory.GIT: ["git", "branch", "merge", "rebase", "commit", "pull request"],
    TopicCategory.SYSTEM_DESIGN: ["system design", "scalability", "load balancer", "cache", "database", "microservices"],
    TopicCategory.SQL: ["sql", "query", "join", "index", "transaction", "normalization"],
    TopicCategory.DEVOPS: ["devops", "ci/cd", "pipeline", "deployment", "monitoring", "terraform"],
    TopicCategory.CYBERSECURITY: ["security", "authentication", "authorization", "encryption", "vulnerability", "owasp"],
    TopicCategory.CLOUD: ["cloud", "serverless", "kubernetes", "microservices", "infrastructure"],
}


VISUAL_STRATEGY_MAP = {
    TopicCategory.PYTHON: [VisualType.CODE, VisualType.FLOWCHART, VisualType.QUIZ],
    TopicCategory.JAVA: [VisualType.CODE, VisualType.ARCHITECTURE, VisualType.QUIZ],
    TopicCategory.AWS: [VisualType.ARCHITECTURE, VisualType.FLOWCHART, VisualType.QUIZ],
    TopicCategory.AI: [VisualType.PIPELINE, VisualType.DATA_FLOW, VisualType.QUIZ],
    TopicCategory.MACHINE_LEARNING: [VisualType.PIPELINE, VisualType.DATA_FLOW, VisualType.QUIZ],
    TopicCategory.RAG: [VisualType.ARCHITECTURE, VisualType.PIPELINE, VisualType.QUIZ],
    TopicCategory.KUBERNETES: [VisualType.ARCHITECTURE, VisualType.CLUSTER_DIAGRAM, VisualType.QUIZ],
    TopicCategory.DOCKER: [VisualType.CONTAINER_DIAGRAM, VisualType.FLOWCHART, VisualType.QUIZ],
    TopicCategory.GIT: [VisualType.BRANCH_DIAGRAM, VisualType.FLOWCHART, VisualType.QUIZ],
    TopicCategory.SYSTEM_DESIGN: [VisualType.ARCHITECTURE, VisualType.COMPARISON, VisualType.QUIZ],
    TopicCategory.SQL: [VisualType.QUERY_TABLE, VisualType.FLOWCHART, VisualType.QUIZ],
    TopicCategory.DEVOPS: [VisualType.CI_CD_PIPELINE, VisualType.FLOWCHART, VisualType.QUIZ],
    TopicCategory.CYBERSECURITY: [VisualType.ATTACK_DEFENSE_FLOW, VisualType.FLOWCHART, VisualType.QUIZ],
    TopicCategory.CLOUD: [VisualType.SERVICE_ARCHITECTURE, VisualType.ARCHITECTURE, VisualType.QUIZ],
    TopicCategory.GENERIC: [VisualType.INFOGRAPHIC, VisualType.FLOWCHART, VisualType.QUIZ],
}


CATEGORY_COLORS = {
    TopicCategory.PYTHON: "#3776AB",
    TopicCategory.JAVA: "#ED8B00",
    TopicCategory.AWS: "#FF9900",
    TopicCategory.AI: "#8B5CF6",
    TopicCategory.MACHINE_LEARNING: "#F59E0B",
    TopicCategory.RAG: "#EC4899",
    TopicCategory.KUBERNETES: "#326CE5",
    TopicCategory.DOCKER: "#2496ED",
    TopicCategory.GIT: "#F05032",
    TopicCategory.SYSTEM_DESIGN: "#06B6D4",
    TopicCategory.SQL: "#4479A1",
    TopicCategory.DEVOPS: "#10B981",
    TopicCategory.CYBERSECURITY: "#EF4444",
    TopicCategory.CLOUD: "#0EA5E9",
    TopicCategory.GENERIC: "#6366F1",
}


SYSTEM_PROMPT = """You are an expert technical educator creating structured educational content for social media infographics.

Your job is to generate a JSON object following the exact EducationalContent schema. The content will be rendered as:
- Facebook: Single 1080x1350 "cheat sheet" poster
- Instagram: 5-6 slide carousel (1080x1350 each)

RULES:
1. Generate ACCURATE, technically correct content. No invented concepts.
2. Keep text SHORT - optimized for mobile reading in seconds.
3. Choose visual_strategy appropriate to the topic category.
4. Quiz must have exactly 4 options with exactly 1 correct answer.
5. Flowchart steps: 3-7 items, each with clear label.
6. Infographic points: 3-6 key concepts.
6. Code snippets: Real, working syntax. Include language.
7. Architecture: 3-8 components with connections.
8. Comparison: 3-6 rows, fair comparison.
9. Hook: One compelling sentence that makes someone stop scrolling.
10. Takeaway: One memorable sentence summarizing the core learning.
11. Output ONLY valid JSON. No markdown, no explanation, no extra text.

DIFFICULTY ADAPTATION:
- beginner: Define terms, explain "what" and "why", simple examples
- intermediate: Explain "how", show patterns, practical usage
- advanced: Deep dive, trade-offs, optimization, edge cases

VISUAL STRATEGY BY CATEGORY:
- python/java: code + flowchart + quiz
- aws/kubernetes/system_design/cloud: architecture + flowchart + quiz
- ai/ml/rag: pipeline + data_flow + quiz
- docker: container_diagram + flowchart + quiz
- git: branch_diagram + flowchart + quiz
- sql: query_table + flowchart + quiz
- devops: ci_cd_pipeline + flowchart + quiz
- cybersecurity: attack_defense_flow + flowchart + quiz
- generic: infographic + flowchart + quiz"""


def build_user_prompt(topic: str, category: TopicCategory, difficulty: DifficultyLevel, audience: list) -> str:
    strategy = VISUAL_STRATEGY_MAP.get(category, [VisualType.INFOGRAPHIC, VisualType.FLOWCHART, VisualType.QUIZ])
    strategy_names = [s.value for s in strategy]
    
    difficulty_guidance = {
        DifficultyLevel.BEGINNER: "Focus on definitions, analogies, and 'what/why'. Avoid jargon. Use simple examples.",
        DifficultyLevel.INTERMEDIATE: "Focus on 'how it works', common patterns, and practical usage. Show real code/config.",
        DifficultyLevel.ADVANCED: "Focus on internals, trade-offs, optimization, and edge cases. Assume strong prior knowledge.",
    }
    
    return f"""Generate educational content for: "{topic}"

Category: {category.value}
Difficulty: {difficulty.value}
Audience: {', '.join(audience)}

Difficulty guidance: {difficulty_guidance[difficulty]}

Required visual strategy: {strategy_names}

Generate JSON with these fields:
- topic (string)
- category (string: {category.value})
- audience (array)
- difficulty (string: {difficulty.value})
- hook (string: one compelling sentence)
- infographic (object with title, points[]) - IF infographic in strategy
- flowchart (object with title, steps[]) - IF flowchart in strategy
- architecture (object with title, components[]) - IF architecture in strategy
- code (object with language, title, content) - IF code in strategy
- comparison (object with title, header_a, header_b, rows[]) - IF comparison in strategy
- quiz (object with question, options[4], explanation) - REQUIRED
- takeaway (string: one memorable sentence)
- cta (string: default "Save this for later!")
- visual_strategy (array of strings)

Make content technically accurate, concise, and educational. Output ONLY JSON."""


def clean_and_parse_json(content: str) -> dict:
    """Extract and parse JSON from LLM response."""
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


class ContentGenerator:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.gemini_client = genai.Client(api_key=api_key)
        self.gemini_models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
        ]

    def _call_gemini(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try Gemini models in order."""
        for model_name in self.gemini_models_to_try:
            try:
                print(f"🤖 Trying Gemini ({model_name})...")
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=[types.Content(role="user", parts=[types.Part(text=user_prompt)])],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.3,
                        response_mime_type="application/json",
                    )
                )
                if response.text:
                    content_json = json.loads(response.text)
                    return content_json
            except Exception as e:
                err_str = str(e).lower()
                print(f"⚠️ Gemini ({model_name}) failed: {e}")
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    print(f"   Quota exceeded, trying next model...")
                    continue
                break
        return None

    def _call_groq(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try Groq models."""
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            return None
        
        groq_models = [
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "openai/gpt-oss-20b",
            "llama-3.1-8b-instant",
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
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    content = safe_extract_choices(r.json(), f"Groq {model_name}")
                    if content:
                        return clean_and_parse_json(content)
                else:
                    print(f"⚠️ Groq ({model_name}) failed: {r.status_code} - {r.text}")
            except Exception as e:
                print(f"⚠️ Groq ({model_name}) exception: {e}")
        return None

    def _call_cloudflare(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try Cloudflare Workers AI models."""
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
                else:
                    print(f"⚠️ Cloudflare ({model_name}) failed: {r.status_code}")
            except Exception as e:
                print(f"⚠️ Cloudflare ({model_name}) exception: {e}")
        return None

    def _call_opencode(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try OpenCode Zen."""
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
                "temperature": 0.3,
            }
            r = requests.post("https://opencode.ai/zen/v1/chat/completions", json=payload, headers=headers, timeout=60)
            if r.status_code == 200:
                content = safe_extract_choices(r.json(), "OpenCode Zen")
                if content:
                    return clean_and_parse_json(content)
        except Exception as e:
            print(f"⚠️ OpenCode Zen exception: {e}")
        return None

    def _call_cerebras(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try Cerebras models."""
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

    def _call_nvidia(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try NVIDIA NIM."""
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

    def _call_mistral(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try Mistral AI."""
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

    def _call_github_models(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try GitHub Models."""
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

    def _call_openai(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try OpenAI."""
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

    def _call_anthropic(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try Anthropic Claude."""
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

    def _call_deepseek(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try DeepSeek."""
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

    def _call_openrouter(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try OpenRouter."""
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

    def _generate_with_fallback(self, user_prompt: str) -> Optional[EducationalContent]:
        """Try all providers in order until one succeeds."""
        
        # 1. Try Gemini first
        result = self._call_gemini(user_prompt)
        if result:
            print("✅ Gemini succeeded")
            return result
        
        print("🚨 Gemini failed all models. Attempting fallback providers...")
        
        # 2. Fallback chain
        fallbacks = [
            ("Groq", self._call_groq),
            ("Cloudflare", self._call_cloudflare),
            ("OpenCode Zen", self._call_opencode),
            ("Cerebras", self._call_cerebras),
            ("NVIDIA NIM", self._call_nvidia),
            ("Mistral", self._call_mistral),
            ("GitHub Models", self._call_github_models),
            ("OpenAI", self._call_openai),
            ("Anthropic", self._call_anthropic),
            ("DeepSeek", self._call_deepseek),
            ("OpenRouter", self._call_openrouter),
        ]
        
        for name, func in fallbacks:
            try:
                result = func(user_prompt)
                if result:
                    print(f"✅ {name} succeeded")
                    return result
            except Exception as e:
                print(f"⚠️ {name} fallback failed: {e}")
        
        print("🚨 All fallback providers failed or not configured")
        return None

    def detect_category(self, topic: str) -> TopicCategory:
        topic_lower = topic.lower()
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in topic_lower:
                    return cat
        return TopicCategory.GENERIC

    def get_theme_color(self, category: TopicCategory) -> str:
        return CATEGORY_COLORS.get(category, CATEGORY_COLORS[TopicCategory.GENERIC])

    def get_visual_strategy(self, category: TopicCategory) -> list[VisualType]:
        return VISUAL_STRATEGY_MAP.get(category, [VisualType.INFOGRAPHIC, VisualType.FLOWCHART, VisualType.QUIZ])

    def generate(self, topic: str, audience: list = None, difficulty: DifficultyLevel = None) -> EducationalContent:
        if audience is None:
            audience = ["students", "developers"]
        if difficulty is None:
            difficulty = DifficultyLevel.INTERMEDIATE

        category = self.detect_category(topic)
        theme_color = self.get_theme_color(category)
        visual_strategy = self.get_visual_strategy(category)

        user_prompt = build_user_prompt(topic, category, difficulty, audience)

        content_json = self._generate_with_fallback(user_prompt)
        if not content_json:
            raise RuntimeError(f"All LLM providers failed for topic: {topic}")
        
        content_json["visual_strategy"] = [s.value for s in visual_strategy]
        return EducationalContent(**content_json)

    def generate_batch(self, topics: list, audience: list = None, difficulty: DifficultyLevel = None) -> list[EducationalContent]:
        results = []
        for topic in topics:
            try:
                content = self.generate(topic, audience, difficulty)
                results.append(content)
            except Exception as e:
                print(f"Failed to generate content for '{topic}': {e}")
        return results