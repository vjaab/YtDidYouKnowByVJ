#!/usr/bin/env python3
"""
CI script to set up YouTube auth files from GitHub Secrets.
"""
import json
import os
import sys


def write_json_file(filename, env_var):
    val = os.environ.get(env_var, '').strip()
    if not val:
        print(f'WARNING: {env_var} not set, skipping {filename}')
        return False
    try:
        data = json.loads(val)
        if isinstance(data, str):
            data = json.loads(data)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Successfully wrote {filename}')
        return True
    except Exception as e:
        print(f'ERROR writing {filename}: {e}')
        return False


def main():
    for filename, env_var in [('client_secret.json', 'CLIENT_SECRET_JSON'), ('token.json', 'TOKEN_JSON')]:
        if not write_json_file(filename, env_var):
            sys.exit(1)


if __name__ == '__main__':
    main()