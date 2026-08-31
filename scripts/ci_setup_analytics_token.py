#!/usr/bin/env python3
"""
CI script to set up YouTube Analytics token from GitHub Secrets.
"""
import json
import os
import sys


def main():
    val = os.environ.get('YOUTUBE_ANALYTICS_TOKEN_JSON', '').strip()
    if not val:
        print('YOUTUBE_ANALYTICS_TOKEN_JSON not set, skipping')
        sys.exit(0)
    try:
        data = json.loads(val)
        if isinstance(data, str):
            data = json.loads(data)
        with open('token_youtube_analytics.json', 'w') as f:
            json.dump(data, f, indent=2)
        print('Successfully wrote token_youtube_analytics.json')
    except Exception as e:
        print(f'ERROR writing token_youtube_analytics.json: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()