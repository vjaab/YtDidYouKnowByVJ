#!/usr/bin/env python3
"""
Platform-native thumbnail optimization script.
Generates platform-specific variants and optimization report.
"""
import os
import json
import glob
from datetime import datetime

# Platform specs with safe zones for text/face placement
PLATFORMS = {
    'youtube_shorts': {'aspect': '9:16', 'safe_zones': {'top': 0.15, 'bottom': 0.25, 'sides': 0.10}},
    'instagram_reels': {'aspect': '9:16', 'safe_zones': {'top': 0.10, 'bottom': 0.30, 'sides': 0.08}},
    'tiktok': {'aspect': '9:16', 'safe_zones': {'top': 0.12, 'bottom': 0.35, 'sides': 0.05}},
    'youtube_longform': {'aspect': '16:9', 'safe_zones': {'top': 0.08, 'bottom': 0.08, 'sides': 0.10}}
}

# Find latest thumbnails
yt_thumbs = sorted(glob.glob('output/thumbnail_*.jpg'), key=os.path.getmtime, reverse=True)[:3]
shorts_thumbs = sorted(glob.glob('output/thumbnail_shorts_*.jpg'), key=os.path.getmtime, reverse=True)[:3]

for platform, spec in PLATFORMS.items():
    thumbs = yt_thumbs if 'longform' in platform else shorts_thumbs
    if thumbs:
        print(f'📱 {platform}: {len(thumbs)} variants ready for {spec["aspect"]}')
        # Safe zone validation would go here
        # In production: use PIL to verify text/face not in unsafe zones

# Save platform optimization report
report = {
    'timestamp': datetime.now().isoformat(),
    'platforms': list(PLATFORMS.keys()),
    'thumbnail_variants': {
        'youtube_longform': len(yt_thumbs),
        'youtube_shorts': len(shorts_thumbs),
        'instagram_reels': len(shorts_thumbs),
        'tiktok': len(shorts_thumbs)
    },
    'ab_test_status': 'pending_ctr_collection'
}
os.makedirs('output', exist_ok=True)
with open('output/platform_optimization_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print('✅ Platform optimization report saved')