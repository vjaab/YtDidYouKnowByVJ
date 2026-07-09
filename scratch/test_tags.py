import sys
import os

# Adjust path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tags_helper import get_optimized_metadata

def run_tests():
    print("🧪 Running Tags & Hashtags Optimizer Tests...")
    
    # ── Test Scenario 1: Sam Altman & OpenAI ──
    print("\n--- Test 1: Sam Altman & OpenAI ---")
    test1 = get_optimized_metadata(
        title="Sam Altman's secret plan for GPT-5 leaked! 🤯",
        script="OpenAI CEO Sam Altman just hinted that GPT-5 is coming sooner than you think. He claims it will achieve full AGI and change software development forever. But critics like Gary Marcus are skeptical.",
        sub_category="AI News",
        initial_keywords=["leak", "AGI"],
        initial_companies=["OpenAI"],
        initial_people=["Sam Altman", "Gary Marcus"],
        initial_hashtags=["#AINews"]
    )
    print("Generated Tags:", test1["tags"])
    print("Generated Hashtags:", test1["hashtags"])
    
    # Assertions
    assert len(test1["tags"]) >= 8 and len(test1["tags"]) <= 15, f"Tag count {len(test1['tags'])} out of bounds"
    assert test1["hashtags"] == ["#SamAltman", "#GaryMarcus", "#GPT5", "#OpenAICEO"], f"Unexpected hashtags: {test1['hashtags']}"
    assert "Sam Altman" in test1["tags"]
    assert "OpenAI CEO" in test1["tags"]
    
    # ── Test Scenario 2: Jensen Huang & NVIDIA GPUs ──
    print("\n--- Test 2: Jensen Huang & NVIDIA GPUs ---")
    test2 = get_optimized_metadata(
        title="NVIDIA's new AI chip is terrifyingly fast!",
        script="Jensen Huang just unveiled the new Rubin GPU architecture at Computex. This new chip uses advanced semiconductors to power next-generation agentic AI workflows. The demand for cloud computing resources is soaring.",
        sub_category="AI Hardware",
        initial_keywords=["computex", "hardware"],
        initial_companies=["NVIDIA"],
        initial_people=["Jensen Huang"],
        initial_hashtags=[]
    )
    print("Generated Tags:", test2["tags"])
    print("Generated Hashtags:", test2["hashtags"])
    
    # Assertions
    assert len(test2["tags"]) >= 8 and len(test2["tags"]) <= 15, f"Tag count {len(test2['tags'])} out of bounds"
    assert test2["hashtags"] == ["#JensenHuang", "#NVIDIA", "#RubinGPU", "#Computex"], f"Unexpected hashtags: {test2['hashtags']}"
    assert "Jensen Huang" in test2["tags"]
    assert "NVIDIA" in test2["tags"]
    
    # ── Test Scenario 3: General Tech Fact ──
    print("\n--- Test 3: General Tech Fact ---")
    test3 = get_optimized_metadata(
        title="Did you know this hidden phone setting?",
        script="Here is a simple tech hack. You can disable location tracking on your phone by turning off this simple setting. It prevents apps from watching your daily moves.",
        sub_category="Mobile Tips",
        initial_keywords=["tips", "hacks"],
        initial_companies=[],
        initial_people=[],
        initial_hashtags=[]
    )
    print("Generated Tags:", test3["tags"])
    print("Generated Hashtags:", test3["hashtags"])
    
    # Assertions
    assert len(test3["tags"]) >= 8 and len(test3["tags"]) <= 15, f"Tag count {len(test3['tags'])} out of bounds"
    assert test3["hashtags"] == ["#MobileTips", "#TechHack", "#DisableLocation", "#LocationTracking"], f"Unexpected hashtags: {test3['hashtags']}"
    
    print("\n✅ All Optimizer Tests Passed Successfully!")
 
if __name__ == "__main__":
    run_tests()
