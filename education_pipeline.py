import asyncio
import os
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from content.schemas import EducationalContent, DifficultyLevel, RenderConfig
from content.generator import ContentGenerator
from content.validator import validate_content, auto_fix_content
from render.renderer import PlaywrightRenderer, render_educational_content


class EducationPipeline:
    def __init__(
        self,
        gemini_api_key: str,
        output_dir: str = "output",
        model: str = "gemini-2.5-flash"
    ):
        self.generator = ContentGenerator(gemini_api_key, model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_content(
        self,
        topic: str,
        audience: List[str] = None,
        difficulty: DifficultyLevel = None,
        auto_fix: bool = True
    ) -> EducationalContent:
        """Generate educational content for a topic."""
        content = self.generator.generate(topic, audience, difficulty)
        
        # Validate
        is_valid, issues = validate_content(content)
        if not is_valid:
            print(f"⚠️ Validation issues for '{topic}':")
            for issue in issues:
                print(f"  - {issue}")
            
            if auto_fix:
                print("  Attempting auto-fix...")
                content = auto_fix_content(content)
                is_valid, issues = validate_content(content)
                if is_valid:
                    print("  ✅ Auto-fix successful")
                else:
                    print(f"  ❌ Auto-fix failed, remaining issues: {issues}")
        
        return content

    def generate_batch(
        self,
        topics: List[str],
        audience: List[str] = None,
        difficulty: DifficultyLevel = None
    ) -> List[EducationalContent]:
        """Generate content for multiple topics."""
        return self.generator.generate_batch(topics, audience, difficulty)

    async def render_content(
        self,
        content: EducationalContent,
        topic_slug: str = None
    ) -> dict:
        """Render content for Facebook and Instagram."""
        config = RenderConfig(
            platform="facebook",
            theme_color=self.generator.get_theme_color(content.category),
            brand_handle="@Vijayakumarj_ai",
        )
        
        renderer = PlaywrightRenderer(str(self.output_dir))
        try:
            return await renderer.render_both(content, config, topic_slug)
        finally:
            await renderer.close()

    async def process_topic(
        self,
        topic: str,
        audience: List[str] = None,
        difficulty: DifficultyLevel = None,
        topic_slug: str = None
    ) -> dict:
        """Full pipeline: generate content -> validate -> render."""
        print(f"\n{'='*60}")
        print(f"📚 Processing: {topic}")
        print(f"{'='*60}")
        
        # Generate content
        print("🤖 Generating educational content...")
        content = self.generate_content(topic, audience, difficulty)
        print(f"  ✅ Generated: {content.category.value} | {content.difficulty.value}")
        print(f"  🎯 Hook: {content.hook[:80]}...")
        
        # Render
        print("🎨 Rendering images...")
        results = await self.render_content(content, topic_slug)
        
        print(f"  ✅ Facebook: {results['facebook']}")
        print(f"  ✅ Instagram: {len(results['instagram'])} slides")
        for i, path in enumerate(results['instagram'], 1):
            print(f"     Slide {i}: {path}")
        
        return {
            "topic": topic,
            "content": content.model_dump(),
            "outputs": results,
            "timestamp": datetime.now().isoformat(),
        }

    async def process_batch(
        self,
        topics: List[str],
        audience: List[str] = None,
        difficulty: DifficultyLevel = None
    ) -> List[dict]:
        """Process multiple topics."""
        results = []
        for topic in topics:
            try:
                slug = "".join(c if c.isalnum() else "_" for c in topic)[:30]
                result = await self.process_topic(topic, audience, difficulty, slug)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to process '{topic}': {e}")
                results.append({"topic": topic, "error": str(e)})
        return results

    def save_results(self, results: List[dict], filename: str = None):
        """Save pipeline results to JSON."""
        if filename is None:
            filename = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {filepath}")
        return filepath


async def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Educational Content Pipeline")
    parser.add_argument("--topic", type=str, help="Single topic to process")
    parser.add_argument("--topics", nargs="+", help="Multiple topics to process")
    parser.add_argument("--topics-file", type=str, help="JSON file with topics list")
    parser.add_argument("--audience", nargs="+", default=["students", "developers"])
    parser.add_argument("--difficulty", choices=["beginner", "intermediate", "advanced"], default="intermediate")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--api-key", type=str, help="Gemini API key (or use GEMINI_API_KEY env)")
    parser.add_argument("--dry-run", action="store_true", help="Generate content only, don't render")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY required")
        return 1
    
    # Get topics
    topics = []
    if args.topic:
        topics = [args.topic]
    elif args.topics:
        topics = args.topics
    elif args.topics_file:
        with open(args.topics_file) as f:
            data = json.load(f)
            topics = data if isinstance(data, list) else data.get("topics", [])
    else:
        # Default educational topics
        topics = [
            "Python Decorators",
            "AWS Lambda",
            "Java Streams",
            "RAG Pipeline",
            "Kubernetes Pods",
            "Docker Containers",
            "Git Branching",
            "System Design Basics",
            "SQL Joins",
            "CI/CD Pipelines",
        ]
    
    print(f"🚀 Starting Education Pipeline")
    print(f"   Topics: {len(topics)}")
    print(f"   Audience: {args.audience}")
    print(f"   Difficulty: {args.difficulty}")
    print(f"   Output: {args.output_dir}")
    
    pipeline = EducationPipeline(
        gemini_api_key=api_key,
        output_dir=args.output_dir,
        model=args.model
    )
    
    if args.dry_run:
        print("\n🧪 DRY RUN - Generating content only")
        contents = pipeline.generate_batch(topics, args.audience, DifficultyLevel(args.difficulty))
        for content in contents:
            print(f"\n📝 {content.topic} ({content.category.value})")
            print(f"   Hook: {content.hook}")
            print(f"   Strategy: {[v.value for v in content.visual_strategy]}")
    else:
        results = await pipeline.process_batch(topics, args.audience, DifficultyLevel(args.difficulty))
        pipeline.save_results(results)
        
        # Summary
        success = sum(1 for r in results if "error" not in r)
        failed = len(results) - success
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY: {success} succeeded, {failed} failed")
        print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))