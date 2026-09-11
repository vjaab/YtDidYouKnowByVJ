"""
topic_classifier.py — Embedding-based deduplication & Zero-shot classification
Uses local sentence-transformers & transformers models (no API keys, no rate limits)
"""

import os
import json
from typing import List, Dict, Optional, Tuple
import numpy as np

from config import (
    EMBEDDING_MODEL, ZERO_SHOT_MODEL,
    EMBEDDING_SIMILARITY_THRESHOLD, ZERO_SHOT_THRESHOLD,
    HAS_EMBEDDING_MODEL, HAS_ZERO_SHOT_MODEL
)

# Category labels for zero-shot classification
CATEGORY_LABELS = [
    "AI & Tech Tools",
    "Tech Gadgets & Inventions",
    "Finance & Tech Economy",
    "Facts & Trivia",
    "Coding & Development Hacks",
    "Science & Research Breakthroughs",
    "Cybersecurity & Privacy",
    "Robotics & Automation",
    "Developer Productivity",
    "Open Source Projects",
]

# Initialize models lazily
_embedding_model = None
_zero_shot_classifier = None


def _get_embedding_model():
    """Lazy load sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None and HAS_EMBEDDING_MODEL:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"✅ Embedding model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load embedding model: {e}")
    return _embedding_model


def _get_zero_shot_classifier():
    """Lazy load zero-shot classification pipeline."""
    global _zero_shot_classifier
    if _zero_shot_classifier is None and HAS_ZERO_SHOT_MODEL:
        try:
            from transformers import pipeline
            print(f"🔄 Loading zero-shot classifier: {ZERO_SHOT_MODEL}")
            _zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=ZERO_SHOT_MODEL,
                device=-1  # CPU
            )
            print(f"✅ Zero-shot classifier loaded")
        except Exception as e:
            print(f"⚠️ Failed to load zero-shot classifier: {e}")
    return _zero_shot_classifier


def get_text_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding vector for a text string."""
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    except Exception as e:
        print(f"⚠️ Embedding generation failed: {e}")
        return None


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized embeddings."""
    return float(np.dot(emb1, emb2))


def check_topic_uniqueness(
    candidate_title: str,
    candidate_description: str,
    recent_topics: List[Dict],
    threshold: float = EMBEDDING_SIMILARITY_THRESHOLD
) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a candidate topic is unique compared to recent topics using embeddings.
    
    Returns:
        (is_unique, max_similarity, most_similar_topic_title)
    """
    model = _get_embedding_model()
    if model is None:
        print("⚠️ Embedding model not available, skipping uniqueness check")
        return True, 0.0, None
    
    candidate_text = f"{candidate_title}. {candidate_description}"
    candidate_emb = get_text_embedding(candidate_text)
    
    if candidate_emb is None:
        return True, 0.0, None
    
    max_similarity = 0.0
    most_similar_title = None
    
    for topic in recent_topics:
        if not isinstance(topic, dict):
            continue
        
        # Use title + description for comparison
        recent_text = f"{topic.get('title', '')}. {topic.get('description', '')}"
        if not recent_text.strip():
            continue
            
        recent_emb = get_text_embedding(recent_text)
        if recent_emb is None:
            continue
        
        similarity = compute_similarity(candidate_emb, recent_emb)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_title = topic.get('title', 'Unknown')
    
    is_unique = max_similarity < threshold
    
    if not is_unique:
        print(f"🔄 Topic rejected (similarity {max_similarity:.2f} >= {threshold}): '{candidate_title}' ~ '{most_similar_title}'")
    else:
        print(f"✅ Topic unique (max similarity {max_similarity:.2f} < {threshold}): '{candidate_title}'")
    
    return is_unique, max_similarity, most_similar_title


def classify_topic_category(
    title: str,
    description: str,
    candidate_labels: List[str] = None,
    threshold: float = ZERO_SHOT_THRESHOLD
) -> Tuple[str, float]:
    """
    Classify a topic into a category using zero-shot classification.
    
    Returns:
        (predicted_category, confidence)
    """
    classifier = _get_zero_shot_classifier()
    if classifier is None:
        print("⚠️ Zero-shot classifier not available, defaulting to 'AI & Tech Tools'")
        return "AI & Tech Tools", 0.0
    
    labels = candidate_labels or CATEGORY_LABELS
    text = f"{title}. {description}"
    
    try:
        result = classifier(text, candidate_labels=labels, multi_label=False)
        predicted = result["labels"][0]
        confidence = result["scores"][0]
        
        if confidence >= threshold:
            print(f"🏷️ Zero-shot classified: '{title[:50]}...' → {predicted} ({confidence:.2f})")
            return predicted, confidence
        else:
            print(f"⚠️ Low confidence ({confidence:.2f} < {threshold}) for '{title[:50]}...', defaulting to 'AI & Tech Tools'")
            return "AI & Tech Tools", confidence
    except Exception as e:
        print(f"⚠️ Zero-shot classification failed: {e}")
        return "AI & Tech Tools", 0.0


def filter_and_classify_candidates(
    candidates: List[Dict],
    recent_topics: List[Dict],
    max_candidates: int = 10
) -> List[Dict]:
    """
    Filter candidates by uniqueness and classify them by category.
    Returns filtered and classified candidates, sorted by engagement score.
    """
    if not candidates:
        return []
    
    filtered = []
    
    for candidate in candidates:
        title = candidate.get("title", "")
        description = candidate.get("description", "")
        
        if not title:
            continue
        
        # Check uniqueness
        is_unique, similarity, similar_to = check_topic_uniqueness(
            title, description, recent_topics
        )
        
        if not is_unique:
            # Skip this candidate, but continue to next
            candidate["_rejected_reason"] = f"Similar to: {similar_to} (score: {similarity:.2f})"
            continue
        
        # Classify category
        category, confidence = classify_topic_category(title, description)
        candidate["_predicted_category"] = category
        candidate["_category_confidence"] = confidence
        candidate["_embedding_similarity"] = similarity
        
        filtered.append(candidate)
        
        if len(filtered) >= max_candidates:
            break
    
    print(f"📊 Filtered {len(candidates)} → {len(filtered)} unique candidates")
    return filtered


def get_recent_topic_embeddings(tracker_file: str = "data/topic_tracker.json", limit: int = 20) -> List[Dict]:
    """Load recent topics from tracker for deduplication."""
    if not os.path.exists(tracker_file):
        return []
    
    try:
        with open(tracker_file, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        history = tracker.get("history", [])
        # Return last N topics with title and description
        recent = []
        for entry in history[-limit:]:
            if isinstance(entry, dict):
                recent.append({
                    "title": entry.get("title", ""),
                    "description": entry.get("news_headline", ""),
                })
        return recent
    except Exception as e:
        print(f"⚠️ Failed to load recent topics: {e}")
        return []


# CLI for testing
if __name__ == "__main__":
    # Test embedding
    test_text = "New AI tool for code generation"
    emb = get_text_embedding(test_text)
    if emb is not None:
        print(f"Embedding shape: {emb.shape}")
    
    # Test zero-shot
    cat, conf = classify_topic_category(
        "New open-source LLM beats GPT-4",
        "A new open-source language model has been released that outperforms GPT-4 on benchmarks"
    )
    print(f"Category: {cat}, Confidence: {conf}")