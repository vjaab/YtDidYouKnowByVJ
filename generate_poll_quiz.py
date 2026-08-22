#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_poll_quiz.py — Generate interactive polls/quizzes for Instagram Stories.
"""

import os
import sys
import json
import argparse
from pathlib import Path

POLL_TEMPLATES = {
    "python_decorators": {
        "question": "What does @decorator do?",
        "options": [
            "Wraps a function",
            "Deletes a function", 
            "Creates a class",
            "Imports a module"
        ],
        "correct": 0,
        "explanation": "@decorator syntax wraps a function to add behavior without modifying its source."
    },
    "aws_lambda": {
        "question": "What triggers AWS Lambda?",
        "options": [
            "API Gateway, S3, DynamoDB",
            "Only manual invocation",
            "Only CloudWatch events",
            "Only SNS topics"
        ],
        "correct": 0,
        "explanation": "Lambda can be triggered by API Gateway, S3, DynamoDB, EventBridge, SNS, SQS, and more."
    },
    "rag_pipeline": {
        "question": "What does RAG stand for?",
        "options": [
            "Retrieval-Augmented Generation",
            "Random Access Generation",
            "Recursive Algorithm Generation",
            "Real-time AI Generation"
        ],
        "correct": 0,
        "explanation": "RAG = Retrieval-Augmented Generation. It retrieves relevant docs before generating answers."
    },
    "kubernetes_pods": {
        "question": "What comes after 'Pending' in pod lifecycle?",
        "options": [
            "ContainerCreating → Running",
            "Running → Terminated",
            "Failed → Succeeded",
            "Succeeded → Failed"
        ],
        "correct": 0,
        "explanation": "Pod lifecycle: Pending → ContainerCreating → Running → Terminating → Succeeded/Failed"
    },
    "docker_multistage": {
        "question": "What's the main benefit of multi-stage builds?",
        "options": [
            "Smaller final image size",
            "Faster build time",
            "More layers",
            "Larger attack surface"
        ],
        "correct": 0,
        "explanation": "Multi-stage builds copy only artifacts to final stage, reducing image size by 90%+."
    },
}

def generate_quiz_poll(topic_key: str) -> dict:
    """Generate a quiz poll for Instagram Stories."""
    template = POLL_TEMPLATES.get(topic_key, POLL_TEMPLATES["python_decorators"])
    
    return {
        "type": "quiz",
        "question": template["question"],
        "options": template["options"],
        "correct_option": template["correct"],
        "explanation": template["explanation"],
    }

def generate_opinion_poll(topic_key: str) -> dict:
    """Generate an opinion poll for Instagram Stories."""
    polls = {
        "python_decorators": {
            "question": "How often do you use decorators?",
            "options": ["Daily", "Weekly", "Rarely", "Never heard of them"],
        },
        "aws_lambda": {
            "question": "What's your Lambda experience?",
            "options": ["Production use", "Learning", "Tried once", "Never used"],
        },
        "rag_pipeline": {
            "question": "Have you built a RAG system?",
            "options": ["Yes, in production", "Experimenting", "Planning to", "No"],
        },
        "kubernetes_pods": {
            "question": "How do you debug pod issues?",
            "options": ["kubectl logs", "kubectl describe", "Events", "All of the above"],
        },
        "docker_multistage": {
            "question": "What's your Docker image size?",
            "options": ["<100MB", "100-500MB", "500MB-1GB", ">1GB"],
        },
    }
    
    template = polls.get(topic_key, polls["python_decorators"])
    return {
        "type": "poll",
        "question": template["question"],
        "options": template["options"],
    }

def generate_slider_poll(topic_key: str) -> dict:
    """Generate an emoji slider poll for Instagram Stories."""
    sliders = {
        "python_decorators": {
            "question": "How useful are decorators?",
            "emoji": "🐍",
        },
        "aws_lambda": {
            "question": "Serverless love level?",
            "emoji": "☁️",
        },
        "rag_pipeline": {
            "question": "RAG effectiveness?",
            "emoji": "🤖",
        },
        "kubernetes_pods": {
            "question": "K8s complexity rating?",
            "emoji": "☸️",
        },
        "docker_multistage": {
            "question": "Docker optimization priority?",
            "emoji": "🐳",
        },
    }
    
    template = sliders.get(topic_key, sliders["python_decorators"])
    return {
        "type": "emoji_slider",
        "question": template["question"],
        "emoji": template["emoji"],
    }

def generate_quiz_sticker_data(topic: str) -> dict:
    """Generate all interactive sticker data for a topic."""
    topic_key = topic.lower().replace(" ", "_").replace("-", "_")
    
    return {
        "topic": topic,
        "quiz_poll": generate_quiz_poll(topic_key),
        "opinion_poll": generate_opinion_poll(topic_key),
        "slider_poll": generate_slider_poll(topic_key),
    }

def save_poll_data(topic: str, output_dir: Path):
    """Save poll data as JSON for Instagram Story creation."""
    data = generate_quiz_sticker_data(topic)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"poll_{topic.lower().replace(' ', '_')}.json"
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Poll data saved: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate interactive polls/quizzes")
    parser.add_argument("--topic", required=True, help="Topic name")
    parser.add_argument("--output-dir", default="output/social_images", help="Output directory")
    parser.add_argument("--type", choices=["quiz", "poll", "slider", "all"], default="all")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    topic_key = args.topic.lower().replace(" ", "_").replace("-", "_")
    
    if args.type in ["quiz", "all"]:
        quiz = generate_quiz_poll(topic_key)
        print(f"Quiz: {quiz['question']}")
        for i, opt in enumerate(quiz["options"]):
            marker = " ✓" if i == quiz["correct_option"] else ""
            print(f"  {i+1}. {opt}{marker}")
    
    if args.type in ["poll", "all"]:
        poll = generate_opinion_poll(topic_key)
        print(f"\nPoll: {poll['question']}")
        for i, opt in enumerate(poll["options"]):
            print(f"  {i+1}. {opt}")
    
    if args.type in ["slider", "all"]:
        slider = generate_slider_poll(topic_key)
        print(f"\nSlider: {slider['question']} {slider['emoji']}")
    
    if args.type == "all":
        save_poll_data(args.topic, output_dir)

if __name__ == "__main__":
    main()