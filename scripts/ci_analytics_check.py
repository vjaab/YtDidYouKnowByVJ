#!/usr/bin/env python3
"""
CI script to run analytics sync and health check.
"""
import sys
sys.path.insert(0, '.')

import json

# Sync with YouTube Analytics (last 7 days)
print('🔄 Syncing hook analytics with YouTube...')
try:
    from hook_analytics_sync import sync_hook_analytics_with_youtube
    sync_hook_analytics_with_youtube(days_back=7)
except Exception as e:
    print(f'⚠️ Analytics sync failed: {e}')

# Check metrics against targets
print('\n📈 Checking metrics against targets...')
try:
    from hook_analytics import get_hook_analytics, print_analytics_summary
    analytics = get_hook_analytics()

    # Targets from strategy
    TARGETS = {
        'ctr': 0.05,           # >5% CTR
        'retention': 0.50,     # >50% AVD
        'engagement': 0.03,    # >3% engagement rate
    }

    alerts = []
    for category, cat_data in analytics.get('categories', {}).items():
        for pattern_id, pattern_data in cat_data.items():
            if pattern_id.startswith('_'):  # Skip metadata keys
                continue
            avg_retention = pattern_data.get('avg_retention', 0)
            avg_engagement = pattern_data.get('avg_engagement', 0)
            total_views = pattern_data.get('total_views', 0)
            
            if total_views > 10:  # Only alert if enough data
                if avg_retention < TARGETS['retention']:
                    alerts.append(f'🔴 LOW RETENTION: {category}/{pattern_id} = {avg_retention:.1%} (target >{TARGETS["retention"]:.0%})')
                if avg_engagement < TARGETS['engagement']:
                    alerts.append(f'🔴 LOW ENGAGEMENT: {category}/{pattern_id} = {avg_engagement:.1%} (target >{TARGETS["engagement"]:.0%})')

    if alerts:
        print('\n⚠️ METRICS BELOW TARGET:')
        for alert in alerts:
            print(f'  {alert}')
        
        # Send Telegram alert if configured
        import os
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if bot_token and chat_id:
            import requests
            msg = '📊 *Analytics Alert - Metrics Below Target*\n\n' + '\n'.join(alerts[:10])
            try:
                requests.post(f'https://api.telegram.org/bot{bot_token}/sendMessage', 
                             json={'chat_id': chat_id, 'text': msg, 'parse_mode': 'Markdown'})
                print('📱 Telegram alert sent')
            except Exception as e:
                print(f'⚠️ Failed to send Telegram alert: {e}')
    else:
        print('✅ All tracked metrics above target thresholds')

    print('\n📋 Full analytics summary:')
    print_analytics_summary()
except Exception as e:
    print(f'⚠️ Analytics summary check failed: {e}')