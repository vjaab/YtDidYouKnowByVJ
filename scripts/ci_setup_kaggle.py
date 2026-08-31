#!/usr/bin/env python3
"""
CI script to set up Kaggle credentials from GitHub Secrets.
"""
import json
import os
import sys


def main():
    username = os.environ.get('KAGGLE_USERNAME', '').strip()
    key = os.environ.get('KAGGLE_KEY', '').strip()
    if not username or not key:
        print('WARNING: KAGGLE_USERNAME or KAGGLE_KEY not set')
        sys.exit(0)
    
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
        json.dump({'username': username, 'key': key}, f, indent=2)
    print('Successfully wrote kaggle.json')


if __name__ == '__main__':
    main()