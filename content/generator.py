import json
import os
import re
from typing import Optional, List, Any, Dict
from google import genai
from google.genai import types

from content.schemas import (
    EducationalContent, TopicCategory, DifficultyLevel, VisualType,
    Quiz, QuizOption, Flowchart, FlowchartStep, Infographic, InfographicPoint,
    CodeSnippet, ArchitectureDiagram, ArchitectureComponent, ComparisonTable, ComparisonRow
)

from content.validator import validate_content
from content.schemas import EducationalContent, TopicCategory, DifficultyLevel, VisualType, ComparisonTable


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
5. Flowchart steps: 3-7 items. Label: MAX 40 chars. Description optional.
6. Infographic points: 3-6 key concepts.
7. Code snippets: Real, working syntax. MAX 12 lines, MAX 500 chars. Include language.
8. Architecture: 3-8 components. Description: 10-100 chars.
9. Comparison: 3-6 rows. Feature: MAX 40 chars. Option A/B: MAX 50 chars each.
10. Hook: One compelling sentence that makes someone stop scrolling.
11. Takeaway: One memorable sentence summarizing the core learning.
12. Output ONLY valid JSON. No markdown, no explanation, no extra text.

HARD LIMITS (will be validated - content exceeding limits will be REJECTED):
- Flowchart step.label: ≤ 40 characters
- Flowchart step.description: keep brief
- CodeSnippet.content: ≤ 500 characters, ≤ 12 lines
- ArchitectureComponent.description: 10-100 characters
- ComparisonRow.feature: ≤ 40 characters
- ComparisonRow.option_a: ≤ 50 characters
- ComparisonRow.option_b: ≤ 50 characters
- CodeSnippet: ≤ 12 lines total

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
- generic: infographic + flowchart + quiz

CRITICAL for comparison table: Each row MUST have feature, option_a, option_b.
  Example: {"feature": "Data Model", "option_a": "Key-value/document", "option_b": "Relational tables"}"""


def build_user_prompt(topic: str, category: TopicCategory, difficulty: DifficultyLevel, audience: list) -> str:
    strategy = VISUAL_STRATEGY_MAP.get(category, [VisualType.INFOGRAPHIC, VisualType.FLOWCHART, VisualType.QUIZ])
    strategy_names = [s.value for s in strategy]
    
    difficulty_guidance = {
        DifficultyLevel.BEGINNER: "Focus on definitions, analogies, and 'what/why'. Avoid jargon. Use simple examples.",
        DifficultyLevel.INTERMEDIATE: "Focus on 'how it works', common patterns, and practical usage. Show real code/config.",
        DifficultyLevel.ADVANCED: "Focus on internals, trade-offs, optimization, and edge cases. Assume strong prior knowledge.",
    }
    
    comparison_instruction = """
COMPARISON TABLE FORMAT (if comparison in strategy):
  Each row MUST have exactly: feature, option_a, option_b
  Example rows:
    {"feature": "Data Model", "option_a": "Key-value/document", "option_b": "Relational tables"}
    {"feature": "Scaling", "option_a": "Horizontal (auto)", "option_b": "Vertical + read replicas"}
    {"feature": "Query Flexibility", "option_a": "Key-based only", "option_b": "SQL joins, aggregations"}
  Do NOT use "left/right", "traditional/modern", "before/after" - use option_a/option_b exactly."""
    
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

{comparison_instruction if 'comparison' in strategy_names else ''}

STRICT LIMITS - EXCEEDING THESE WILL CAUSE VALIDATION FAILURE:
- Flowchart step.label ≤ 40 chars
- CodeSnippet.content ≤ 500 chars, ≤ 12 lines
- ArchitectureComponent.description 10-100 chars
- ComparisonRow.feature ≤ 40 chars, option_a ≤ 50 chars, option_b ≤ 50 chars

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

    def _call_gemini(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Try Gemini models in order. Returns None immediately on quota exhaustion."""
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
                    print(f"   🚫 Quota exhausted - skipping remaining Gemini models, falling back to other providers...")
                    return None  # Immediately skip to fallback providers
                break
        return None

    def _generate_with_fallback(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Try all providers in order until one succeeds. Returns normalized dict."""
        
        # 1. Try Gemini first
        result = self._call_gemini(user_prompt)
        if result:
            print("✅ Gemini succeeded")
            return normalize_llm_response(result)
        
        print("🚨 Gemini failed all models. Attempting fallback providers...")
        
        # 2. Use shared fallback chain
        result = call_fallback_chain(user_prompt, normalize=True)
        if result:
            return result
        
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

    def _regenerate_invalid_section(self, content_json: dict, validation_errors: list, user_prompt: str) -> dict:
        """Regenerate only the invalid sections based on validation errors."""
        invalid_sections = set()
        for error in validation_errors:
            if "comparison" in error.lower():
                invalid_sections.add("comparison")
            elif "flowchart" in error.lower():
                invalid_sections.add("flowchart")
            elif "architecture" in error.lower():
                invalid_sections.add("architecture")
            elif "code" in error.lower():
                invalid_sections.add("code")
            elif "infographic" in error.lower():
                invalid_sections.add("infographic")
        
        if not invalid_sections:
            return content_json
        
        print(f"🔄 Regenerating invalid sections: {invalid_sections}")
        
        sections_to_fix = []
        if "comparison" in invalid_sections:
            sections_to_fix.append("comparison table with feature/option_a/option_b")
        if "flowchart" in invalid_sections:
            sections_to_fix.append("flowchart with short labels (≤40 chars)")
        if "architecture" in invalid_sections:
            sections_to_fix.append("architecture with descriptions (10-100 chars)")
        if "code" in invalid_sections:
            sections_to_fix.append("code snippet (≤12 lines, ≤500 chars)")
        if "infographic" in invalid_sections:
            sections_to_fix.append("infographic points")
        
        fix_prompt = f"""The previous response had validation errors in: {', '.join(sections_to_fix)}.

Generate ONLY the corrected JSON for these sections. Output a partial JSON object with ONLY the corrected fields.

Example format for comparison:
  "comparison": {{"title": "X vs Y", "header_a": "A", "header_b": "B", "rows": [{{"feature": "Feature", "option_a": "A", "option_b": "B"}}]}}

Example for flowchart:
  "flowchart": {{"title": "Flow", "steps": [{{"label": "Step 1"}, {{"label": "Step 2"}}]}}"""

        focused_prompt = user_prompt + "\n\n" + fix_prompt + "\n\nOutput ONLY the corrected JSON fields."
        
        for name, func in [
            ("Gemini", lambda p: self._call_gemini(p)),
            ("OpenCode", lambda p: self._call_opencode(p)),
            ("Groq", lambda p: self._call_groq(p)),
        ]:
            try:
                print(f"   🔄 Attempting fix with {name}...")
                result = func(focused_prompt)
                if result:
                    for key in invalid_sections:
                        if key in result:
                            content_json[key] = result[key]
                            print(f"   ✅ Fixed {key} with {name}")
                    break
            except Exception as e:
                print(f"   ⚠️ Fix failed with {name}: {e}")
        
        return content_json

    def generate(self, topic: str, audience: list = None, difficulty: DifficultyLevel = None) -> EducationalContent:
        if audience is None:
            audience = ["students", "developers"]
        if difficulty is None:
            difficulty = DifficultyLevel.INTERMEDIATE

        category = self.detect_category(topic)
        theme_color = self.get_theme_color(category)
        visual_strategy = self.get_visual_strategy(category)

        user_prompt = build_user_prompt(topic, category, difficulty, audience)

        for attempt in range(2):
            content_json = self._generate_with_fallback(user_prompt)
            if not content_json:
                raise RuntimeError(f"All LLM providers failed for topic: {topic}")
            
            content_json["visual_strategy"] = [s.value for s in visual_strategy]
            content_json["category"] = category.value
            content_json["difficulty"] = difficulty.value
            content_json["audience"] = audience
            
            is_valid, errors = validate_content(EducationalContent(**content_json))
            if is_valid:
                return EducationalContent(**content_json)
            
            print(f"⚠️ Validation errors on attempt {attempt + 1}:")
            for err in errors:
                print(f"  - {err}")
            
            if attempt < 1:
                print("🔄 Attempting targeted regeneration...")
                content_json = self._regenerate_invalid_section(content_json, errors, user_prompt)
            else:
                print("⚠️ Validation failed after regeneration, proceeding anyway")
                return EducationalContent(**content_json)
        
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