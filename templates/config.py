from pydantic import BaseModel
from typing import Dict, Any


class TemplateConfig(BaseModel):
    # Canvas
    width: int = 1080
    height: int = 1350
    
    # Typography
    font_family: str = "Inter"
    font_size_base: int = 16
    font_size_sm: int = 14
    font_size_lg: int = 20
    font_size_xl: int = 24
    font_size_2xl: int = 32
    
    # Spacing
    spacing_xs: int = 4
    spacing_sm: int = 8
    spacing_md: int = 16
    spacing_lg: int = 24
    spacing_xl: int = 32
    
    # Border radius
    radius_sm: int = 4
    radius_md: int = 8
    radius_lg: int = 16
    
    # Brand
    brand_handle: str = "@Vijayakumarj_ai"
    logo_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        # Generate theme color variations
        import colorsys
        
        # Parse theme color (assuming hex)
        theme_color = self.theme_color if hasattr(self, 'theme_color') else "#3B82F6"
        r = int(theme_color[1:3], 16) / 255
        g = int(theme_color[3:5], 16) / 255
        b = int(theme_color[5:7], 16) / 255
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # Light version
        r_light, g_light, b_light = colorsys.hls_to_rgb(h, min(l + 0.35, 0.95), s)
        theme_color_light = f"#{int(r_light*255):02x}{int(g_light*255):02x}{int(b_light*255):02x}"
        
        # Dark version
        r_dark, g_dark, b_dark = colorsys.hls_to_rgb(h, max(l - 0.2, 0.15), min(s + 0.1, 1.0))
        theme_color_dark = f"#{int(r_dark*255):02x}{int(g_dark*255):02x}{int(b_dark*255):02x}"
        
        # Background version (very light)
        r_bg, g_bg, b_bg = colorsys.hls_to_rgb(h, 0.97, 0.3)
        theme_color_bg = f"#{int(r_bg*255):02x}{int(g_bg*255):02x}{int(b_bg*255):02x}"
        
        return {
            "width": self.width,
            "height": self.height,
            "font_family": self.font_family,
            "font_size_base": self.font_size_base,
            "font_size_sm": self.font_size_sm,
            "font_size_lg": self.font_size_lg,
            "font_size_xl": self.font_size_xl,
            "font_size_2xl": self.font_size_2xl,
            "spacing_xs": self.spacing_xs,
            "spacing_sm": self.spacing_sm,
            "spacing_md": self.spacing_md,
            "spacing_lg": self.spacing_lg,
            "spacing_xl": self.spacing_xl,
            "radius_sm": self.radius_sm,
            "radius_md": self.radius_md,
            "radius_lg": self.radius_lg,
            "brand_handle": self.brand_handle,
            "theme_color": theme_color,
            "theme_color_light": theme_color_light,
            "theme_color_dark": theme_color_dark,
            "theme_color_bg": theme_color_bg,
        }


# Default configs for each platform
FACEBOOK_CONFIG = TemplateConfig(
    width=1080,
    height=1350,
    brand_handle="@Vijayakumarj_ai"
)

INSTAGRAM_CONFIG = TemplateConfig(
    width=1080,
    height=1350,
    brand_handle="@Vijayakumarj_ai"
)