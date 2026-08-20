import asyncio
import os
from pathlib import Path
from typing import List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

from content.schemas import EducationalContent, RenderConfig, TopicCategory
from templates.config import FACEBOOK_CONFIG, INSTAGRAM_CONFIG


class TemplateRenderer:
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Add custom filters
        self.env.filters['int_to_letter'] = lambda i: chr(65 + i) if 0 <= i < 26 else '?'
        
        # Add enumerate as global for templates
        self.env.globals['enumerate'] = enumerate

    def _get_theme_vars(self, config: RenderConfig) -> dict:
        base = FACEBOOK_CONFIG.to_dict() if config.platform == "facebook" else INSTAGRAM_CONFIG.to_dict()
        base.update({
            "theme_color": config.theme_color,
            "font_family": config.font_family,
            "brand_handle": config.brand_handle,
            "width": config.width,
            "height": config.height,
        })
        return base

    def render_facebook(self, content: EducationalContent, config: RenderConfig) -> str:
        template = self.env.get_template("facebook/educational-poster.html")
        
        # Determine category display name
        category_display = content.category.value.upper().replace("_", " ")
        difficulty_display = content.difficulty.value.capitalize()
        
        return template.render(
            topic=content.topic,
            category_display=category_display,
            difficulty_display=difficulty_display,
            reading_time=max(1, len(content.hook) // 100),
            hook=content.hook,
            infographic=content.infographic,
            code=content.code,
            architecture=content.architecture,
            flowchart=content.flowchart,
            comparison=content.comparison,
            quiz=content.quiz,
            takeaway=content.takeaway,
            cta=content.cta,
            **self._get_theme_vars(config)
        )

    def render_instagram_slides(self, content: EducationalContent, config: RenderConfig) -> List[str]:
        slides = []
        total_slides = 5
        
        # Slide 1: Hook
        template = self.env.get_template("instagram/slide-hook.html")
        slides.append(template.render(
            topic=content.topic,
            category_display=content.category.value.upper().replace("_", " "),
            hook=content.hook,
            total_slides=total_slides,
            
            **self._get_theme_vars(config)
        ))
        
        # Slide 2: Concept
        template = self.env.get_template("instagram/slide-concept.html")
        slides.append(template.render(
            topic=content.topic,
            category_display=content.category.value.upper().replace("_", " "),
            hook=content.hook,
            infographic=content.infographic,
            code=content.code,
            total_slides=total_slides,
            
            **self._get_theme_vars(config)
        ))
        
        # Slide 3: Flow/Architecture
        template = self.env.get_template("instagram/slide-flow.html")
        slides.append(template.render(
            topic=content.topic,
            category_display=content.category.value.upper().replace("_", " "),
            flowchart=content.flowchart,
            architecture=content.architecture,
            total_slides=total_slides,
            
            **self._get_theme_vars(config)
        ))
        
        # Slide 4: Quiz
        template = self.env.get_template("instagram/slide-quiz.html")
        slides.append(template.render(
            topic=content.topic,
            category_display=content.category.value.upper().replace("_", " "),
            quiz=content.quiz,
            total_slides=total_slides,
            
            **self._get_theme_vars(config)
        ))
        
        # Slide 5: Answer
        template = self.env.get_template("instagram/slide-answer.html")
        slides.append(template.render(
            topic=content.topic,
            category_display=content.category.value.upper().replace("_", " "),
            quiz=content.quiz,
            takeaway=content.takeaway,
            cta=content.cta,
            total_slides=total_slides,
            
            **self._get_theme_vars(config)
        ))
        
        return slides


class PlaywrightRenderer:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_renderer = TemplateRenderer()
        self._browser = None
        self._context = None
        self._playwright = None

    async def _get_browser(self):
        if self._browser is None:
            from playwright.async_api import async_playwright
            try:
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox']
                )
                self._context = await self._browser.new_context(
                    viewport={'width': 1080, 'height': 1350},
                    device_scale_factor=2,
                )
            except Exception as e:
                # Clean up on failure
                if self._playwright:
                    await self._playwright.stop()
                self._playwright = None
                self._browser = None
                raise
        return self._context

    async def close(self):
        if getattr(self, '_browser', None):
            await self._browser.close()
        if getattr(self, '_playwright', None):
            await self._playwright.stop()

    async def render_html_to_image(self, html: str, output_path: str, width: int = 1080, height: int = 1350) -> str:
        context = await self._get_browser()
        page = await context.new_page()
        
        try:
            await page.set_content(html, wait_until='networkidle')
            await page.wait_for_timeout(500)  # Let fonts/styles settle
            
            # Set viewport to exact size
            await page.set_viewport_size({"width": width, "height": height})
            
            # Take screenshot
            await page.screenshot(
                path=output_path,
                full_page=True,
                type='png'
            )
            return output_path
        finally:
            await page.close()

    async def render_facebook_post(self, content: EducationalContent, config: RenderConfig, output_name: str = None) -> str:
        html = self.template_renderer.render_facebook(content, config)
        
        if output_name is None:
            safe_topic = "".join(c if c.isalnum() else "_" for c in content.topic)[:30]
            output_name = f"fb_{safe_topic}.png"
        
        output_path = self.output_dir / "facebook" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return await self.render_html_to_image(html, str(output_path), config.width, config.height)

    async def render_instagram_carousel(self, content: EducationalContent, config: RenderConfig, output_prefix: str = None) -> List[str]:
        slides_html = self.template_renderer.render_instagram_slides(content, config)
        
        if output_prefix is None:
            safe_topic = "".join(c if c.isalnum() else "_" for c in content.topic)[:30]
            output_prefix = f"ig_{safe_topic}"
        
        output_dir = self.output_dir / "instagram" / output_prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        for i, slide_html in enumerate(slides_html):
            output_path = output_dir / f"slide_{i+1}.png"
            await self.render_html_to_image(slide_html, str(output_path), config.width, config.height)
            output_paths.append(str(output_path))
        
        return output_paths

    async def render_both(self, content: EducationalContent, config: RenderConfig, topic_slug: str = None) -> dict:
        if topic_slug is None:
            topic_slug = "".join(c if c.isalnum() else "_" for c in content.topic)[:30]
        
        fb_path = await self.render_facebook_post(content, config, f"{topic_slug}.png")
        ig_paths = await self.render_instagram_carousel(content, config, topic_slug)
        
        return {
            "facebook": fb_path,
            "instagram": ig_paths,
            "topic": content.topic,
        }


async def render_educational_content(content: EducationalContent, output_dir: str = "output") -> dict:
    """Convenience function to render content for both platforms."""
    config = RenderConfig(
        platform="facebook",
        theme_color=getattr(content, 'theme_color', '#3B82F6'),
    )
    
    renderer = PlaywrightRenderer(output_dir)
    try:
        return await renderer.render_both(content, config)
    finally:
        await renderer.close()