import unittest
import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trending_engine import (
    fetch_github_trending_ai,
    fetch_hacker_news_trending,
    fetch_huggingface_trending,
    fetch_arxiv_ai_papers,
    fetch_tldr_ai_newsletters,
    fetch_reddit_hot_ai,
    compute_engagement_score,
    fetch_all_trending_signals
)
from fetch_research_papers import fetch_tech_news, fetch_ai_tools

class TestMultiSourcePipeline(unittest.TestCase):
    def test_github_trending(self):
        print("\n--- Testing GitHub Trending (Languages + Topics + Awesome Lists) ---")
        repos = fetch_github_trending_ai()
        self.assertIsInstance(repos, list)
        print(f"Retrieved {len(repos)} GitHub repos.")

    def test_hacker_news_trending(self):
        print("\n--- Testing Hacker News Trending ---")
        hn_articles = fetch_hacker_news_trending()
        self.assertIsInstance(hn_articles, list)
        print(f"Retrieved {len(hn_articles)} Hacker News items.")

    def test_huggingface_trending(self):
        print("\n--- Testing Hugging Face Papers & Models ---")
        hf_articles = fetch_huggingface_trending()
        self.assertIsInstance(hf_articles, list)
        print(f"Retrieved {len(hf_articles)} Hugging Face items.")

    def test_arxiv_ai_papers(self):
        print("\n--- Testing ArXiv AI Research Papers ---")
        arxiv_articles = fetch_arxiv_ai_papers()
        self.assertIsInstance(arxiv_articles, list)
        print(f"Retrieved {len(arxiv_articles)} ArXiv papers.")

    def test_tldr_ai_newsletters(self):
        print("\n--- Testing TLDR AI Newsletters ---")
        tldr_articles = fetch_tldr_ai_newsletters()
        self.assertIsInstance(tldr_articles, list)
        print(f"Retrieved {len(tldr_articles)} Newsletter items.")

    def test_compute_engagement_score(self):
        print("\n--- Testing Unified Engagement Scoring ---")
        mock_hn = {
            "title": "Hacker News: DeepSeek V3 release",
            "type": "hacker_news",
            "_engagement": {"points": 450, "comments": 220}
        }
        mock_hf = {
            "title": "Hugging Face Paper: FlashAttention-3",
            "type": "huggingface_trending",
            "_engagement": {"upvotes": 85}
        }
        mock_arxiv = {
            "title": "ArXiv: Lossy Quantization and VRAM optimization",
            "description": "VRAM memory reduction for LLM inference",
            "type": "arxiv_papers",
            "_engagement": {}
        }
        hn_score = compute_engagement_score(mock_hn)
        hf_score = compute_engagement_score(mock_hf)
        arxiv_score = compute_engagement_score(mock_arxiv)
        
        self.assertGreaterEqual(hn_score, 50)
        self.assertGreaterEqual(hf_score, 45)
        self.assertGreaterEqual(arxiv_score, 40)
        print(f"HN Score: {hn_score}, HF Score: {hf_score}, ArXiv Score: {arxiv_score}")

    def test_fetch_all_trending_signals(self):
        print("\n--- Testing Master Aggregator across all 9 data feeds ---")
        all_signals = fetch_all_trending_signals(target_country="US")
        self.assertIsInstance(all_signals, list)
        self.assertGreater(len(all_signals), 0)
        print(f"Master Aggregator returned {len(all_signals)} total scored signals.")

if __name__ == "__main__":
    unittest.main()
