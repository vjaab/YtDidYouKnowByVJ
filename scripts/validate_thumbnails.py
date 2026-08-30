#!/usr/bin/env python3
"""
Thumbnail quality validation script.
Checks A/B test metadata against quality gates.
"""
import json
import glob
import os

# Quality gate thresholds
MIN_CONTRAST = 4.5      # WCAG AA
MIN_RULE_OF_THIRDS = 0.6
MAX_WORDS = 3
FACE_REQUIRED = True

# Check A/B test metadata for quality gates
abtest_files = glob.glob('output/*_abtest.json')
for f in abtest_files:
    try:
        with open(f) as fp:
            data = json.load(fp)
        for v in data.get('variants', []):
            issues = []
            if v.get('contrast_score', 0) < MIN_CONTRAST:
                issues.append(f'Low contrast: {v["contrast_score"]}')
            if v.get('rule_of_thirds_score', 0) < MIN_RULE_OF_THIRDS:
                issues.append(f'Poor rule of thirds: {v["rule_of_thirds_score"]}')
            if v.get('text_word_count', 99) > MAX_WORDS:
                issues.append(f'Too many words: {v["text_word_count"]}')
            if FACE_REQUIRED and not v.get('face_detected', False):
                issues.append('No face detected')
            if issues:
                print(f'⚠️ {v["variant_id"]}: {", ".join(issues)}')
            else:
                print(f'✅ {v["variant_id"]}: All quality gates passed')
    except Exception as e:
        print(f'⚠️ Could not validate {f}: {e}')