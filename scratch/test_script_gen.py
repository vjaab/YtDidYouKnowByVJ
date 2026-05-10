import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemini_script import pick_and_generate_script
from unittest.mock import patch
from ecosystem_logic import get_slot_info

original_get_slot_info = get_slot_info

def mock_get_slot_info():
    return ("Wednesday", "Slot C", "General")

with patch('gemini_script.get_slot_info', side_effect=mock_get_slot_info):
    data = pick_and_generate_script(topic_type="tools")
    if data:
        print("SUCCESS! Generated Script length:", len(data.get("script", "").split()), "words")
        print("Keys:", data.keys())
    else:
        print("FAILED to generate script")
