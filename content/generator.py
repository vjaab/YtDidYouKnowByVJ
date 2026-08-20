import json
import os
from typing import Optional
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


class ContentGenerator:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

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

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(role="user", parts=[types.Part(text=user_prompt)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
            )
        )

        content_json = json.loads(response.text)
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