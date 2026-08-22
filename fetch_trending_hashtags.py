#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_trending_hashtags.py — Fetch trending hashtags for tech category.
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

# Hashtag categories for different tech topics
HASHTAG_CATEGORIES = {
    "python": [
        "#Python", "#PythonProgramming", "#PythonTips", "#PythonTricks",
        "#PythonDeveloper", "#CodingInPython", "#Python3", "#PyCon",
        "#Django", "#FastAPI", "#Flask", "#Pandas", "#NumPy"
    ],
    "aws": [
        "#AWS", "#AWSCertified", "#CloudComputing", "#Serverless",
        "#Lambda", "#EC2", "#S3", "#DynamoDB", "#CloudFormation",
        "#AWSArchitecture", "#DevOps", "#InfrastructureAsCode"
    ],
    "kubernetes": [
        "#Kubernetes", "#K8s", "#ContainerOrchestration", "#CloudNative",
        "#Docker", "#Helm", "#Microservices", "#DevOps", "#CNCF",
        "#KubernetesCertified", "#ContainerSecurity"
    ],
    "ai_ml": [
        "#MachineLearning", "#DeepLearning", "#AI", "#ArtificialIntelligence",
        "#LLM", "#GenerativeAI", "#RAG", "#LangChain", "#PyTorch",
        "#TensorFlow", "#HuggingFace", "#MLOps", "#DataScience"
    ],
    "devops": [
        "#DevOps", "#CI/CD", "#GitHubActions", "#GitLabCI", "#Jenkins",
        "#InfrastructureAsCode", "#Terraform", "#Ansible", "#Prometheus",
        "#Grafana", "#Observability", "#SRE", "#PlatformEngineering"
    ],
    "javascript": [
        "#JavaScript", "#TypeScript", "#React", "#NodeJS", "#NextJS",
        "#VueJS", "#FrontendDevelopment", "#WebDevelopment", "#Vite",
        "#Webpack", "#ESLint", "#Prettier"
    ],
    "general_tech": [
        "#TechNews", "#SoftwareEngineering", "#Programming", "#Coding",
        "#DeveloperLife", "#CodeQuality", "#CleanCode", "#BestPractices",
        "#TechTrends", "#Innovation", "#DigitalTransformation"
    ],
}

# Cross-platform trending hashtags (always relevant)
CROSS_PLATFORM_HASHTAGS = [
    "#Tech", "#Technology", "#Innovation", "#Engineering",
    "#Developer", "#Coding", "#Programming", "#Software",
]

def fetch_github_trending_topics() -> list:
    """Fetch trending topics from GitHub."""
    try:
        resp = requests.get(
            "https://github-trending-api.now.sh/repositories?language=python&since=daily",
            timeout=10,
        )
        if resp.status_code == 200:
            repos = resp.json()
            topics = []
            for repo in repos[:5]:
                if repo.get("description"):
                    # Extract keywords from description
                    words = repo["description"].lower().split()
                    for word in words:
                        if len(word) > 4 and word.isalpha():
                            topics.append(f"#{word.capitalize()}")
            return list(set(topics))[:10]
    except Exception:
        pass
    return []

def fetch_reddit_trending(programming_subs: list = None) -> list:
    """Fetch trending topics from Reddit programming subreddits."""
    if programming_subs is None:
        programming_subs = ["programming", "python", "javascript", "golang", "rust"]
    
    try:
        topics = []
        for sub in programming_subs[:3]:
            resp = requests.get(
                f"https://www.reddit.com/r/{sub}/hot.json?limit=5",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            if resp.status_code == 200:
                posts = resp.json()["data"]["children"]
                for post in posts[:3]:
                    title = post["data"]["title"].lower()
                    # Extract tech keywords
                    keywords = ["python", "javascript", "react", "kubernetes", "docker", 
                               "aws", "ai", "ml", "llm", "rust", "go", "typescript"]
                    for kw in keywords:
                        if kw in title:
                            topics.append(f"#{kw.capitalize()}")
        return list(set(topics))[:10]
    except Exception:
        return []

def get_hashtags_for_category(category: str, count: int = 15) -> list:
    """Get hashtags for a specific category."""
    category = category.lower()
    
    # Map category to our defined categories
    cat_map = {
        "python": "python",
        "aws": "aws",
        "cloud": "aws",
        "kubernetes": "kubernetes",
        "k8s": "kubernetes",
        "ai": "ai_ml",
        "ml": "ai_ml",
        "llm": "ai_ml",
        "devops": "devops",
        "javascript": "javascript",
        "typescript": "javascript",
        "react": "javascript",
        "node": "javascript",
    }
    
    mapped_cat = cat_map.get(category, "general_tech")
    hashtags = HASHTAG_CATEGORIES.get(mapped_cat, HASHTAG_CATEGORIES["general_tech"])
    
    # Add cross-platform tags
    all_tags = hashtags + CROSS_PLATFORM_HASHTAGS
    
    # Add trending from GitHub/Reddit
    all_tags += fetch_github_trending_topics()
    all_tags += fetch_reddit_trending()
    
    # Deduplicate and return
    seen = set()
    unique_tags = []
    for tag in all_tags:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_tags.append(tag)
    
    return unique_tags[:count]

def main():
    parser = argparse.ArgumentParser(description="Fetch trending hashtags")
    parser.add_argument("--category", default="general_tech", help="Topic category")
    parser.add_argument("--count", type=int, default=15, help="Number of hashtags")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()
    
    hashtags = get_hashtags_for_category(args.category, args.count)
    hashtag_str = " ".join(hashtags)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(hashtag_str)
    
    print(hashtag_str)

if __name__ == "__main__":
    main()