import os
import sys
import json
import time
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Add workspace directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trending_engine
from trending_engine import (
    scrape_github_trending,
    load_cached_github_trending,
    save_github_trending_cache,
    _parse_github_repos,
    fetch_github_trending_ai,
    GITHUB_TRENDING_CACHE_FILE
)

class TestGithubTrending(unittest.TestCase):

    @patch("requests.get")
    def test_scrape_github_trending_parsing(self, mock_get):
        """Verify scrape_github_trending correctly parses repository structures from GitHub HTML."""
        sample_html = """
        <ol class="repo-list">
          <article class="Box-row">
            <h2 class="h3 lh-condensed">
              <a href="/google/antigravity">
                <span class="text-normal">google / </span>antigravity
              </a>
            </h2>
            <p class="col-9 color-fg-muted my-1 pr-4">
              Gravity-defying Python module.
            </p>
            <div class="f6 color-fg-muted mt-2">
              <span class="d-inline-block ml-0 mr-3">
                <span itemprop="programmingLanguage">Python</span>
              </span>
              <a href="/google/antigravity/stargazers" class="d-inline-block Link--muted mr-3">
                12,345
              </a>
              <a href="/google/antigravity/forks" class="d-inline-block Link--muted mr-3">
                567
              </a>
              <span class="d-inline-block float-sm-right">
                123 stars today
              </span>
            </div>
          </article>
        </ol>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sample_html
        mock_get.return_value = mock_resp

        repos = scrape_github_trending()
        self.assertEqual(len(repos), 1)
        repo = repos[0]
        self.assertEqual(repo["repo"], "google/antigravity")
        self.assertEqual(repo["full_name"], "google/antigravity")
        self.assertEqual(repo["description"], "Gravity-defying Python module.")
        self.assertEqual(repo["language"], "Python")
        self.assertEqual(repo["stargazers_count"], 12345)
        self.assertEqual(repo["forks_count"], 567)
        self.assertEqual(repo["stars_in_period"], 123)
        self.assertEqual(repo["url"], "https://github.com/google/antigravity")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getmtime")
    @patch("builtins.open")
    @patch("json.load")
    def test_load_cached_github_trending_ttl(self, mock_json_load, mock_open, mock_getmtime, mock_exists):
        """Verify load_cached_github_trending detects stale cache files (>48 hours)."""
        # 1. Test stale cache warning
        mock_getmtime.return_value = time.time() - (50 * 3600)  # 50 hours ago
        mock_json_load.return_value = [{"repo": "cached/repo"}]
        
        with patch('sys.stdout') as mock_stdout:
            repos = load_cached_github_trending()
            mock_stdout.write.assert_any_call("\n⚠️ CRITICAL: GitHub Trending cache is stale (>48 hours)! Age: 50.0 hours.\n")
            self.assertEqual(len(repos), 1)
            self.assertEqual(repos[0]["repo"], "cached/repo")

        # 2. Test fresh cache
        mock_getmtime.return_value = time.time() - (10 * 3600)  # 10 hours ago
        with patch('sys.stdout') as mock_stdout:
            repos = load_cached_github_trending()
            # Verify no warning printed
            written_outputs = [call.args[0] for call in mock_stdout.write.call_args_list]
            self.assertFalse(any("CRITICAL" in out for out in written_outputs))
            self.assertEqual(len(repos), 1)

    def test_parse_github_repos_velocity(self):
        """Verify velocity calculation correctly normalizes stars depending on source."""
        # 1. Scraped repository with stars_in_period
        scraped_repo = {
            "full_name": "scraped/repo",
            "stargazers_count": 1000,
            "stars_in_period": 85,
            "description": "Scraped description",
            "language": "Python",
            "html_url": "https://github.com/scraped/repo"
        }
        parsed = _parse_github_repos([scraped_repo], min_stars=50)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["_engagement"]["stars_per_day"], 85)

        # 2. API search repository with created_at (10 days ago, 500 stars -> 50 stars/day)
        created_at_str = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        api_repo = {
            "full_name": "api/repo",
            "stargazers_count": 500,
            "stars_in_period": 0,
            "description": "API description",
            "language": "Go",
            "html_url": "https://github.com/api/repo",
            "created_at": created_at_str
        }
        parsed_api = _parse_github_repos([api_repo], min_stars=50)
        self.assertEqual(len(parsed_api), 1)
        self.assertEqual(parsed_api[0]["_engagement"]["stars_per_day"], 50)

    @patch("trending_engine.scrape_github_trending")
    @patch("requests.get")
    def test_fetch_github_trending_deduplication(self, mock_api_get, mock_scrape):
        """Verify case-insensitive deduplication of repositories during merging."""
        # Setup scraper to return mixed casing duplicate + 9 other unique items (to exceed threshold of 10)
        mock_scrape.return_value = [
            {"full_name": "google/Antigravity", "repo": "google/Antigravity", "stars_in_period": 100, "stargazers_count": 1000}
        ] + [
            {"full_name": f"user/repo{i}", "repo": f"user/repo{i}", "stars_in_period": 10 + i, "stargazers_count": 500}
            for i in range(9)
        ]
        # Setup API search to return same slug in different casing
        mock_api_resp = MagicMock()
        mock_api_resp.status_code = 200
        mock_api_resp.json.return_value = {
            "items": [
                {"full_name": "GOOGLE/antigravity", "stargazers_count": 1000, "description": "API desc", "html_url": "url"}
            ]
        }
        mock_api_get.return_value = mock_api_resp

        parsed_results = fetch_github_trending_ai()
        
        # Verify deduplicated to exactly 10 results (Google/Antigravity and google/antigravity merged)
        self.assertEqual(len(parsed_results), 10)
        self.assertEqual(parsed_results[0]["title"].lower(), "github trending: google/antigravity — ")

if __name__ == "__main__":
    unittest.main()
