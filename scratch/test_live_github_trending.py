import os
import sys

# Add workspace directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trending_engine import fetch_github_trending_ai

if __name__ == "__main__":
    print("Running live GitHub trending fetch...")
    repos = fetch_github_trending_ai()
    print(f"Total parsed trending repos: {len(repos)}")
    for r in repos[:5]:
        print(f"- {r['title']} ({r['_engagement']['stars_per_day']} stars/day, relevance: {r.get('_relevance_score', 0)})")
